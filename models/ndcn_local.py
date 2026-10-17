#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDCN-Local : decoupling with reuse, applied to LOCAL descriptors.

Motivation
    The pooled formulation collapses L local descriptors into one C-dimensional
    vector per class. On Conv-4 that is 64 numbers per class, against the
    441 x 64 that DN4 keeps, and it discards exactly the local detail that
    fine-grained recognition depends on. Measured prototype separability under
    pooling is poor (mean off-diagonal cosine ~0.89), and the accuracy ceiling
    sits at ProtoNet level rather than DN4 level.

    NDCN-Local keeps every descriptor and uses the soft mask as a weight in an
    image-to-class k-NN measure. The decomposition, the soft mask, the
    foreground/background split and the background-reuse objective are all
    unchanged; only the way the mask is consumed changes.

Three image-to-class similarities, all in [0, 1]:

    S_fg(q, n)      query FOREGROUND descriptors vs class-n FOREGROUND descriptors
    S_bgsep(q, n)   query FOREGROUND descriptors vs class-n BACKGROUND descriptors
    S_qbg(q, n)     query BACKGROUND descriptors vs class-n BACKGROUND descriptors

    logit_n = gamma_fg * S_fg - gamma_bg * S_bgsep + beta_bg * S_qbg

Each S is a mask-weighted average, over query positions, of the mask-weighted
mean cosine to that position's k nearest support descriptors. Writing
d = 1 - S puts the three quantities on exactly the same scale as the pooled
distances, so the loss functions of the paper carry over unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ndcn import NDCN_Metric, make_nonneg


class NDCN_Local_Metric(NDCN_Metric):

    def __init__(self, *args, neighbor_k=3, chunk_size=0, **kwargs):
        super(NDCN_Local_Metric, self).__init__(*args, **kwargs)
        self.neighbor_k = neighbor_k
        self.chunk_size = chunk_size          # 0 = no chunking

    # ------------------------------------------------------------------
    @staticmethod
    def _local_knn(cos, w_sup, k, eps):
        """
        cos   : nq x way x Lq x Ms   cosine between every query and support descriptor
        w_sup :      way x Ms        support-side weight (mask, or 1-mask)
        returns nq x way x Lq        weighted mean cosine to the k nearest neighbours

        The support weight steers the neighbour search: a background descriptor
        that happens to look similar is down-weighted before the top-k, so it is
        unlikely to be selected and contributes little if it is.
        """
        ws = w_sup.unsqueeze(0).unsqueeze(2)                       # 1 x way x 1 x Ms
        idx = (cos * ws).topk(k, dim=3).indices                    # nq x way x Lq x k
        c_sel = cos.gather(3, idx)
        w_sel = ws.expand(cos.shape).gather(3, idx)
        return (c_sel * w_sel).sum(3) / (w_sel.sum(3) + eps)       # in [0, 1]

    @staticmethod
    def _aggregate(local, w_qry, eps):
        """local : nq x way x Lq ; w_qry : nq x Lq  ->  nq x way"""
        num = (local * w_qry.unsqueeze(1)).sum(2)
        den = w_qry.sum(1, keepdim=True) + eps
        return num / den

    def _auto_chunk(self, NQ, Lq, Ms):
        """Keep the cosine tensor under roughly 256 MB per chunk."""
        if self.chunk_size > 0:
            return self.chunk_size
        per_query = self.way_num * Lq * Ms * 4          # bytes, fp32
        budget = 256 * 1024 * 1024
        return max(1, min(NQ, budget // max(per_query, 1)))

    def _scores(self, Qn, Sn, Mq, Ms):
        """
        Qn : NQ x Lq x C   (l2-normalised query descriptors)
        Sn : way x Ms x C  (l2-normalised support descriptors, class-major)
        Mq : NQ x Lq       query soft mask
        Ms : way x Ms      support soft mask
        """
        eps, k = self.eps, self.neighbor_k
        NQ, Lq = Qn.size(0), Qn.size(1)
        step = self._auto_chunk(NQ, Lq, Sn.size(1))
        need_bg = self.use_bg_sep or self.use_qbg
        Ms_bg = 1.0 - Ms

        S_fg, S_sep, S_qbg = [], [], []
        for i in range(0, NQ, step):
            q = Qn[i:i + step]
            cos = torch.einsum('qlc,wmc->qwlm', q, Sn)             # nq x way x Lq x Ms
            mq = Mq[i:i + step]

            S_fg.append(self._aggregate(self._local_knn(cos, Ms, k, eps), mq, eps))
            if need_bg:
                # bg-separation and bg-consistency share the same neighbour
                # search; only the query-side weighting differs.
                local_bg = self._local_knn(cos, Ms_bg, k, eps)
                if self.use_bg_sep:
                    S_sep.append(self._aggregate(local_bg, mq, eps))
                if self.use_qbg:
                    S_qbg.append(self._aggregate(local_bg, 1.0 - mq, eps))
                del local_bg
            del cos

        cat = lambda x: torch.cat(x, 0) if x else None
        return cat(S_fg), cat(S_sep), cat(S_qbg)

    # ------------------------------------------------------------------
    def forward(self, x_query, x_support, target=None):
        eps, way, shot = self.eps, self.way_num, self.shot_num

        Zq = make_nonneg(self._flatten(x_query), self.nonneg_mode, eps)     # NQ x Lq x C
        Zs = make_nonneg(self._flatten(x_support), self.nonneg_mode, eps)   # (way*shot) x Ls x C
        NQ, Lq, C = Zq.shape
        Ls = Zs.size(1)

        # ---- reference directions (identical to the pooled model) ----------
        ref_q = Zq.mean(dim=1)
        Zs_cls = Zs.view(way, shot, Ls, C)
        ref_s = Zs_cls.mean(dim=(1, 2)).unsqueeze(1).expand(way, shot, C).reshape(way * shot, C)

        # ---- soft masks -----------------------------------------------------
        Mq, alpha_q, _ = self._mask(Zq, ref_q)                              # NQ x Lq
        Ms, alpha_s, _ = self._mask(Zs, ref_s)                              # (way*shot) x Ls

        # ---- image-to-class similarities ------------------------------------
        Qn = F.normalize(Zq, dim=2)
        Sn = F.normalize(Zs, dim=2).view(way, shot * Ls, C)
        Ms_c = Ms.view(way, shot * Ls)

        S_fg, S_sep, S_qbg = self._scores(Qn, Sn, Mq, Ms_c)
        self._cache = (Qn, Sn, Mq, Ms_c)

        logit = self.gamma_fg * S_fg
        if self.use_bg_sep:
            logit = logit - self.gamma_bg * S_sep
        if self.use_qbg:
            logit = logit + self.beta_bg * S_qbg

        # ---- pooled prototypes, used only by the diversity regulariser ------
        Zs_c = Zs.view(way, shot * Ls, C)
        p_fg = (Ms_c.unsqueeze(2) * Zs_c).sum(1) / (Ms_c.sum(1, keepdim=True) + eps)

        self._diag = {
            'logit_spread': (logit.max(1).values - logit.min(1).values).mean().detach(),
            'mask_mean_q': Mq.mean().detach(),
            'mask_mean_s': Ms.mean().detach(),
            'alpha_hi': (alpha_q > 0.7).float().sum(1).mean().detach()
                        if alpha_q is not None else torch.tensor(0.),
            'proto_offdiag': self._offdiag_cos(p_fg).detach(),
        }

        if (not self.training) or target is None:
            return logit

        d_fg = 1.0 - S_fg
        d_sep = (1.0 - S_sep) if self.use_bg_sep else None
        d_qbg = (1.0 - S_qbg) if self.use_qbg else None
        aux = self._aux_from_dists(d_fg, d_sep, d_qbg, Mq, p_fg, target)
        return logit, aux

    def _loss_scores(self, Qn, Sn, Mq, Ms):
        """Background similarities for L_bg, independent of the logit switches."""
        eps, k = self.eps, self.neighbor_k
        NQ, Lq = Qn.size(0), Qn.size(1)
        step = self._auto_chunk(NQ, Lq, Sn.size(1))
        Ms_bg = 1.0 - Ms
        sep, qbg = [], []
        for i in range(0, NQ, step):
            cos = torch.einsum('qlc,wmc->qwlm', Qn[i:i + step], Sn)
            local_bg = self._local_knn(cos, Ms_bg, k, eps)
            mq = Mq[i:i + step]
            sep.append(self._aggregate(local_bg, mq, eps))
            qbg.append(self._aggregate(local_bg, 1.0 - mq, eps))
            del cos, local_bg
        return torch.cat(sep, 0), torch.cat(qbg, 0)

    # ------------------------------------------------------------------
    def _aux_from_dists(self, d_fg, d_sep, d_qbg, Mq, p_fg, target):
        """
        Same objectives as the pooled model, expressed on image-to-class
        distances instead of prototype distances.
        """
        D = self.margin
        NQ = d_fg.size(0)
        idx = torch.arange(NQ, device=d_fg.device)
        d_pos = d_fg[idx, target]
        out = {}

        # (1) hardest inter-class foreground negative
        if self.use_triplet:
            masked = d_fg.clone()
            masked[idx, target] = float('inf')
            d_neg = masked.min(dim=1).values
            out['triplet'] = F.relu(D + d_pos - d_neg).mean()
        else:
            out['triplet'] = d_fg.new_zeros(())

        # (2) intra-class background negative
        #
        # These distances are recomputed rather than reused from the logit, so
        # that --no_bg_sep / --no_qbg switch off logit terms ONLY. Reusing
        # d_sep / d_qbg would silently disable L_bg as well and the ablation
        # would no longer isolate the logit.
        if self.use_bgloss:
            S_sep_l, S_qbg_l = self._loss_scores(*self._cache)
            t1 = F.relu(D + d_pos - (1.0 - S_sep_l)[idx, target])
            t2 = 0.5 * (1.0 - S_qbg_l)[idx, target]
            out['bg'] = (t1 + t2).mean()
        else:
            out['bg'] = d_fg.new_zeros(())

        # (3) prototype diversity (on the pooled foreground prototypes)
        if self.use_div:
            Pn = F.normalize(p_fg, dim=1)
            G = torch.mm(Pn, Pn.t())
            n = G.size(0)
            off = ~torch.eye(n, dtype=torch.bool, device=G.device)
            if self.div_margin > 0:
                out['div'] = (F.relu(G[off] - self.div_margin) ** 2).mean()
            else:
                I = torch.eye(n, device=G.device, dtype=G.dtype)
                out['div'] = ((G - I) ** 2).mean()
        else:
            out['div'] = d_fg.new_zeros(())

        # (4) mask sparsity
        out['mask'] = Mq.mean() if self.use_mask else d_fg.new_zeros(())

        out['total'] = (out['triplet']
                        + self.lambda_bg * out['bg']
                        + self.lambda_div * out['div']
                        + self.lambda_mask * out['mask'])
        return out

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask-weighted reconstruction classifier.

Idea
    The pooled model compresses L descriptors into one C-dimensional vector
    per class before comparing. A reconstruction classifier instead keeps
    every support descriptor, treats the class's descriptors as a subspace,
    and scores a query by how well that subspace reconstructs it. The mask
    enters as a diagonal weighting on the support descriptors, so background
    descriptors contribute less to the subspace.

Formulation
    For class n, let S_n be its (K*L) x C support descriptor matrix and
    w_n the corresponding mask values. With W = diag(w_n),

        min_A  || Q - A (W^{1/2} S_n) ||_F^2 + lam || A ||_F^2

    has the closed form  A* = Q X^T (X X^T + lam I)^{-1}  with X = W^{1/2} S_n,
    and the reconstruction is  Q_hat = A* X.  The score is the negative
    residual, temperature-scaled:

        logit_n = -gamma * mean_l || q_l - q_hat_l ||^2 / || q_l ||^2

    The inverse is taken in the (K*L) x (K*L) Gram space when that is smaller
    than C, and in C x C otherwise, so the cost stays manageable for both
    Conv-4 (L=441, C=64) and ResNet-12 (L=25, C=640).

    lam is stored in log space and learned, so it stays positive.

Cost
    Conv-4 1-shot : Gram is 441x441 per class -> 5 solves of 441x441
    ResNet-12 5-shot: Gram is 125x125 -> cheap
    Use --recon_chunk to split the query batch if memory is tight.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ndcn import NDCN_Metric, make_nonneg


def _inv_softplus(y, eps=1e-12):
    return math.log(math.expm1(max(y, eps)))


class NDCN_Recon_Metric(NDCN_Metric):

    def __init__(self, *args, recon_lam=1.0, recon_chunk=0,
                 recon_mask_power=1.0, **kwargs):
        super(NDCN_Recon_Metric, self).__init__(*args, **kwargs)
        # learnable ridge coefficient, kept positive by softplus
        self._raw_lam = nn.Parameter(torch.tensor(_inv_softplus(float(recon_lam))))
        self.recon_chunk = recon_chunk
        self.recon_mask_power = recon_mask_power

    @property
    def recon_lam(self):
        return F.softplus(self._raw_lam)

    # ------------------------------------------------------------------
    def _residual(self, Q, X, lam):
        """
        Q   : nq x L x C     query descriptors (l2-normalised)
        X   : way x M x C    mask-weighted support descriptors (l2-normalised rows)
        returns nq x way     mean relative squared residual in [0, ~1]
        """
        nq, L, C = Q.shape
        way, M, _ = X.shape
        eps = self.eps

        if M <= C:
            # Gram space: (X X^T + lam I)^{-1}, M x M
            G = torch.bmm(X, X.transpose(1, 2))                        # way x M x M
            G = G + lam * torch.eye(M, device=X.device, dtype=X.dtype).unsqueeze(0)
            Ginv = torch.linalg.inv(G)
            P = torch.bmm(X.transpose(1, 2), torch.bmm(Ginv, X))       # way x C x C
        else:
            # feature space: (X^T X + lam I)^{-1}, C x C
            S = torch.bmm(X.transpose(1, 2), X)                        # way x C x C
            S = S + lam * torch.eye(C, device=X.device, dtype=X.dtype).unsqueeze(0)
            P = torch.bmm(torch.linalg.inv(S), torch.bmm(X.transpose(1, 2), X))

        # Q_hat[i, n] = Q[i] @ P[n]
        Qh = torch.einsum('ilc,ncd->inld', Q, P)                       # nq x way x L x C
        num = ((Q.unsqueeze(1) - Qh) ** 2).sum(-1)                     # nq x way x L
        den = (Q ** 2).sum(-1).unsqueeze(1) + eps                      # nq x 1   x L
        return (num / den).mean(-1)                                    # nq x way

    # ------------------------------------------------------------------
    def forward(self, x_query, x_support, target=None):
        eps, way, shot = self.eps, self.way_num, self.shot_num

        Zq = make_nonneg(self._flatten(x_query), self.nonneg_mode, eps)      # NQ x Lq x C
        Zs = make_nonneg(self._flatten(x_support), self.nonneg_mode, eps)    # (way*shot) x Ls x C
        NQ, Lq, C = Zq.shape
        Ls = Zs.size(1)

        ref_q = Zq.mean(dim=1)
        Zs_cls = Zs.view(way, shot, Ls, C)
        ref_s = Zs_cls.mean(dim=(1, 2)).unsqueeze(1).expand(way, shot, C).reshape(way * shot, C)

        Mq, alpha_q, _ = self._mask(Zq, ref_q)                               # NQ x Lq
        Ms, alpha_s, _ = self._mask(Zs, ref_s)                               # (way*shot) x Ls

        # ---- mask-weighted support subspace ---------------------------------
        Sn = F.normalize(Zs, dim=2).view(way, shot * Ls, C)
        w = Ms.view(way, shot * Ls).clamp_min(eps) ** self.recon_mask_power
        X = Sn * w.sqrt().unsqueeze(2)                                       # way x M x C

        Qn = F.normalize(Zq, dim=2)
        step = self.recon_chunk if self.recon_chunk > 0 else NQ
        res = torch.cat([self._residual(Qn[i:i + step], X, self.recon_lam)
                         for i in range(0, NQ, step)], 0)                    # NQ x way

        logit = -self.gamma_fg * res

        # pooled prototypes are still needed by the diversity regulariser
        Zs_c = Zs.view(way, shot * Ls, C)
        Ms_c = Ms.view(way, shot * Ls)
        _M = Ms_c.size(1)
        p_fg = ((Ms_c.unsqueeze(2) * Zs_c).sum(1)
                / Ms_c.sum(1, keepdim=True).clamp_min(self.pool_floor * _M))

        self._diag = {
            'logit_spread': (logit.max(1).values - logit.min(1).values).mean().detach(),
            'mask_mean_q': Mq.mean().detach(),
            'mask_mean_s': Ms.mean().detach(),
            'alpha_hi': (alpha_q > 0.7).float().sum(1).mean().detach()
                        if alpha_q is not None else torch.tensor(0.),
            'proto_offdiag': self._offdiag_cos(p_fg).detach(),
            'recon_lam': self.recon_lam.detach(),
            'residual': res.mean().detach(),
        }

        if (not self.training) or target is None:
            return logit

        return logit, self._aux_recon(res, Mq, p_fg, target)

    # ------------------------------------------------------------------
    def _aux_recon(self, res, Mq, p_fg, target):
        """Same objectives as the pooled model, on reconstruction residuals."""
        D = self.margin
        NQ = res.size(0)
        idx = torch.arange(NQ, device=res.device)
        d_pos = res[idx, target]
        out = {}

        if self.use_triplet:
            masked = res.clone()
            masked[idx, target] = float('inf')
            d_neg = masked.min(dim=1).values
            out['triplet'] = F.relu(D + d_pos - d_neg).mean()
        else:
            out['triplet'] = res.new_zeros(())

        out['bg'] = res.new_zeros(())          # background reuse is not used

        if self.use_div:
            Pn = F.normalize(p_fg, dim=1)
            G = torch.mm(Pn, Pn.t())
            n = G.size(0)
            off = ~torch.eye(n, dtype=torch.bool, device=G.device)
            out['div'] = (F.relu(G[off] - self.div_margin) ** 2).mean() \
                if self.div_margin > 0 else ((G - torch.eye(n, device=G.device)) ** 2).mean()
        else:
            out['div'] = res.new_zeros(())

        if not self.use_mask:
            out['mask'] = res.new_zeros(())
        elif self.mask_budget > 0:
            out['mask'] = F.relu(Mq.mean() - self.mask_budget)
        else:
            out['mask'] = Mq.mean()

        out['total'] = (out['triplet']
                        + self.lambda_bg * out['bg']
                        + self.lambda_div * out['div']
                        + self.lambda_mask * out['mask'])
        return out

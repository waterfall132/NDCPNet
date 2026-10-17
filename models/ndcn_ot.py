#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimal-transport matching with the NDM mask as transport marginals.

Why this and not the two local-matching heads we already discarded
    NDCN_Local (DN4-style top-k) failed because top-k asks "does this query
    patch resemble ANY support patch", and a sky patch resembles a sky patch in
    every class, so background matches carry no class information and there is
    nothing to stop them dominating.

    NDCN_Comp (component matching) failed because splitting L descriptors into
    r component prototypes divides the effective sample size by r, and because
    the max over class components inflates every class's score.

    Optimal transport avoids both. It never splits the descriptor set, so the
    sample size per prototype is unchanged. And the transport plan is subject to
    mass conservation: giving mass to one pair takes it away from another, so a
    class cannot score highly just by containing one convenient match.

What the mask does here
    DeepEMD generates node weights with a cross-reference mechanism: the weight
    of a node is its dot product with the MEAN node feature of the other image.
    Their ablation shows those weights matter more than the transport itself
    (56.95 with uniform weights, 61.13 with cross-reference). We replace that
    mechanism with the NDM mask, which is computed per image, does not depend on
    which images the episode happens to contain, and does not rely on a mean
    feature direction.

Solver
    We use entropy-regularised OT (Sinkhorn) in the log domain rather than an
    exact LP. It is batched, differentiable, needs no external solver, and
    keeps the head parameter-free. sinkhorn_eps controls how soft the plan is;
    smaller values approach the exact transport at the cost of more iterations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ndcn import NDCN_Metric, make_nonneg


class NDCN_OT_Metric(NDCN_Metric):

    def __init__(self, *args, sinkhorn_eps=0.05, sinkhorn_iters=20,
                 ot_chunk=0, ot_uniform=False, **kwargs):
        super(NDCN_OT_Metric, self).__init__(*args, **kwargs)
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.ot_chunk = ot_chunk
        # ot_uniform reproduces DeepEMD's "equal weights" ablation row, i.e.
        # transport without the mask, so the mask's contribution is isolable.
        self.ot_uniform = ot_uniform

    # ------------------------------------------------------------------
    def _sinkhorn(self, C, a, b):
        """
        C : B x n x m   cost matrix
        a : B x n       source marginal (sums to 1)
        b : B x m       target marginal (sums to 1)
        returns (B,) transport cost and the B x n x m plan

        Log-domain Sinkhorn. The plan is computed without gradient and the cost
        is then read off as <pi, C>, so gradients flow through the cost matrix
        only. At the optimum the derivative with respect to the plan vanishes
        (envelope theorem), and this keeps memory independent of the iteration
        count -- backpropagating through 20 iterations of an L x L plan does not
        fit for Conv-4, where L = 441.
        """
        eps = self.sinkhorn_eps
        log_a = (a + self.eps).log()
        log_b = (b + self.eps).log()

        with torch.no_grad():
            Cd = C.detach()
            f = torch.zeros_like(log_a)
            g = torch.zeros_like(log_b)
            for _ in range(self.sinkhorn_iters):
                Mm = (-Cd + f.unsqueeze(2) + g.unsqueeze(1)) / eps
                f = eps * (log_a - torch.logsumexp(Mm, dim=2)) + f
                Mm = (-Cd + f.unsqueeze(2) + g.unsqueeze(1)) / eps
                g = eps * (log_b - torch.logsumexp(Mm, dim=1)) + g
            pi = ((-Cd + f.unsqueeze(2) + g.unsqueeze(1)) / eps).exp()

        return (pi * C).sum(dim=(1, 2)), pi

    def _auto_chunk(self, NQ, Lq, Ms):
        if self.ot_chunk > 0:
            return self.ot_chunk
        per_query = self.way_num * Lq * Ms * 4
        return max(1, min(NQ, (256 * 1024 * 1024) // max(per_query, 1)))

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

        # the K support images of a class form one descriptor set
        Sn = F.normalize(Zs, dim=2).view(way, shot * Ls, C)                  # way x M x C
        Ms_c = Ms.view(way, shot * Ls)
        Qn = F.normalize(Zq, dim=2)
        M = Sn.size(1)

        # transport marginals
        if self.ot_uniform:
            a_all = torch.full_like(Mq, 1.0 / Lq)
            b_all = torch.full_like(Ms_c, 1.0 / M)
        else:
            a_all = Mq / (Mq.sum(1, keepdim=True) + eps)                     # NQ x Lq
            b_all = Ms_c / (Ms_c.sum(1, keepdim=True) + eps)                 # way x M

        step = self._auto_chunk(NQ, Lq, M)
        costs, plans = [], []
        for i in range(0, NQ, step):
            q = Qn[i:i + step]
            nq = q.size(0)
            cos = torch.einsum('qlc,nmc->qnlm', q, Sn)                       # nq x way x Lq x M
            Cm = (1.0 - cos).reshape(nq * way, Lq, M)
            a = a_all[i:i + step].unsqueeze(1).expand(nq, way, Lq).reshape(-1, Lq)
            b = b_all.unsqueeze(0).expand(nq, way, M).reshape(-1, M)
            c, pi = self._sinkhorn(Cm, a, b)
            costs.append(c.view(nq, way))
            # how concentrated the plan is: 1 / (effective number of pairs used)
            plans.append((pi ** 2).sum(dim=(1, 2)).view(nq, way).detach())
            del cos, Cm, pi

        d = torch.cat(costs, 0)                                              # NQ x way
        conc = torch.cat(plans, 0).mean()
        logit = -self.gamma_fg * d

        # pooled prototypes are still needed by the diversity regulariser
        Zs_c = Zs.view(way, shot * Ls, C)
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
            'ot_cost': d.mean().detach(),
            'plan_conc': conc,
        }

        if (not self.training) or target is None:
            return logit
        return logit, self._aux_ot(d, Mq, p_fg, target)

    # ------------------------------------------------------------------
    def _aux_ot(self, d, Mq, p_fg, target):
        D = self.margin
        NQ = d.size(0)
        idx = torch.arange(NQ, device=d.device)
        d_pos = d[idx, target]
        out = {}

        if self.use_triplet:
            masked = d.clone()
            masked[idx, target] = float('inf')
            d_neg = masked.min(dim=1).values
            out['triplet'] = F.relu(D + d_pos - d_neg).mean()
        else:
            out['triplet'] = d.new_zeros(())

        out['bg'] = d.new_zeros(())

        if self.use_div:
            Pn = F.normalize(p_fg, dim=1)
            G = torch.mm(Pn, Pn.t())
            n = G.size(0)
            off = ~torch.eye(n, dtype=torch.bool, device=G.device)
            out['div'] = (F.relu(G[off] - self.div_margin) ** 2).mean() \
                if self.div_margin > 0 else ((G - torch.eye(n, device=G.device)) ** 2).mean()
        else:
            out['div'] = d.new_zeros(())

        if not self.use_mask:
            out['mask'] = d.new_zeros(())
        elif self.mask_budget > 0:
            out['mask'] = F.relu(Mq.mean() - self.mask_budget)
        else:
            out['mask'] = Mq.mean()

        out['total'] = (out['triplet']
                        + self.lambda_bg * out['bg']
                        + self.lambda_div * out['div']
                        + self.lambda_mask * out['mask'])
        return out

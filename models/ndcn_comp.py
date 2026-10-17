#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Component-level multi-prototype matching.

Motivation
    The pooled head collapses every descriptor of a class into ONE vector
    before comparing. In fine-grained recognition the parts that separate two
    species (beak shape, wing bar, plumage texture) land in DIFFERENT NMF
    components, and averaging them together dilutes exactly the evidence that
    matters. This is the same "averaging washes out detail" problem ProtoNet
    has, moved from spatial averaging to mask-weighted averaging.

Method
    The decomposition already gives, per image, an assignment map
    a_{l,v} (which component each descriptor belongs to). Instead of one
    prototype per class we build one prototype PER COMPONENT:

        p_n^u = sum_l a_{l,u} x_l / sum_l a_{l,u}     over class n's support
        q^v   = sum_l a_{l,v} x_l / sum_l a_{l,v}     for the query

    and score by matching query components against class components.

Permutation
    NMF component indices carry no identity across images: component 2 may be
    the beak in one image and the wing in another. Comparing q^v to p_n^v by
    index is therefore meaningless. We avoid any alignment step by scoring
    each query component against its BEST match, which is invariant to how
    either side is ordered:

        score_n = sum_v w_v * max_u cos(q^v, p_n^u),     w_v = alpha_v / sum alpha

    Weighting by alpha means background components (alpha ~ 0) drop out of the
    score on their own; no threshold is needed.

    --match_mode softmax replaces the hard max with a temperature-weighted
    average over class components, which is smoother early in training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ndcn import NDCN_Metric, make_nonneg


class NDCN_Comp_Metric(NDCN_Metric):

    def __init__(self, *args, match_mode='max', match_tau=10.0,
                 comp_blend=0.0, **kwargs):
        super(NDCN_Comp_Metric, self).__init__(*args, **kwargs)
        self.match_mode = match_mode          # 'max' | 'softmax'
        self.match_tau = match_tau
        # comp_blend > 0 mixes in the pooled score, so the component score can
        # be ablated against the pooled baseline on a single axis.
        self.comp_blend = comp_blend

    # ------------------------------------------------------------------
    @staticmethod
    def _component_prototypes(Z, A, eps, floor):
        """
        Z : B x L x C      descriptors
        A : B x L x r      row-normalised assignment maps
        returns B x r x C  one prototype per component
        """
        num = torch.einsum('blr,blc->brc', A, Z)          # B x r x C
        den = A.sum(dim=1).clamp_min(floor * A.size(1))   # B x r
        return num / den.unsqueeze(2)

    def _match(self, Qv, Pu):
        """
        Qv : NQ x r x C     query component prototypes (l2-normalised)
        Pu : way x r x C    class component prototypes (l2-normalised)
        returns NQ x way x r   best-match cosine of each query component
        """
        # NQ x way x r_q x r_p
        S = torch.einsum('qvc,nuc->qnvu', Qv, Pu)
        if self.match_mode == 'max':
            return S.max(dim=3).values
        elif self.match_mode == 'softmax':
            w = torch.softmax(self.match_tau * S, dim=3)
            return (w * S).sum(dim=3)
        raise ValueError('unknown match_mode: %s' % self.match_mode)

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

        Mq, alpha_q, Aq = self._mask(Zq, ref_q)                              # NQ x Lq, NQ x r, NQ x Lq x r
        Ms, alpha_s, As = self._mask(Zs, ref_s)

        if Aq is None:            # the mu-sigma frontend produces no components
            raise RuntimeError('NDCN_Comp requires the ndm frontend (assignment maps)')

        r = Aq.size(2)

        # ---- component prototypes ------------------------------------------
        Qv = self._component_prototypes(Zq, Aq, eps, self.pool_floor)        # NQ x r x C
        # class side: treat the K support images of a class as one descriptor set
        As_c = As.view(way, shot * Ls, r)
        Zs_c = Zs.view(way, shot * Ls, C)
        Pu = self._component_prototypes(Zs_c, As_c, eps, self.pool_floor)    # way x r x C

        Qv_n = F.normalize(Qv, dim=2)
        Pu_n = F.normalize(Pu, dim=2)

        best = self._match(Qv_n, Pu_n)                                       # NQ x way x r

        # weight components by how object-like they are; background drops out
        w = alpha_q / (alpha_q.sum(dim=1, keepdim=True) + eps)               # NQ x r
        score = (best * w.unsqueeze(1)).sum(dim=2)                           # NQ x way

        # ---- optional blend with the pooled score, for ablation -------------
        Ms_c = Ms.view(way, shot * Ls)
        _M = Ms_c.size(1)
        p_fg = ((Ms_c.unsqueeze(2) * Zs_c).sum(1)
                / Ms_c.sum(1, keepdim=True).clamp_min(self.pool_floor * _M))
        if self.comp_blend > 0:
            q_fg, _ = self._pool(Zq, Mq, eps, self.pool_floor)
            pooled = 1.0 - self._cos_dist(q_fg, p_fg)                        # cosine similarity
            score = (1.0 - self.comp_blend) * score + self.comp_blend * pooled

        d = 1.0 - score                                                      # NQ x way, in [0, 2]
        logit = -self.gamma_fg * d

        self._diag = {
            'logit_spread': (logit.max(1).values - logit.min(1).values).mean().detach(),
            'mask_mean_q': Mq.mean().detach(),
            'mask_mean_s': Ms.mean().detach(),
            'alpha_hi': (alpha_q > 0.7).float().sum(1).mean().detach(),
            'proto_offdiag': self._offdiag_cos(p_fg).detach(),
            # how concentrated the component matching is: 1.0 would mean every
            # query component matches the same class component
            'match_conc': best.max(dim=2).values.mean().detach(),
        }

        if (not self.training) or target is None:
            return logit

        return logit, self._aux_comp(d, Mq, p_fg, target)

    # ------------------------------------------------------------------
    def _aux_comp(self, d, Mq, p_fg, target):
        """Same objectives as the pooled model, on the component-matching distance."""
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

        out['bg'] = d.new_zeros(())          # background reuse is not used

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

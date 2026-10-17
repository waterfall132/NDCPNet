#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDCN: Non-negative Decoupling and Contrast Network
--------------------------------------------------
Drop-in module for the DN4 (2023) codebase.

Contents
    NDM              : parameter-free non-negative decomposition (NNDSVD + MU)
    NDCN_Metric      : soft mask -> fg/bg decoupling -> three-way logit -> losses
    make_nonneg      : enforce non-negativity of backbone descriptors

IMPORTANT - non-negativity
    This codebase's Conv64F_Local uses LeakyReLU(0.2) and ResNet12's BasicBlock
    uses LeakyReLU(0.1). Their outputs are NOT non-negative, so the assumption
    the method relies on does not hold out of the box. Use one of:
        --nonneg_mode relu    (default; Z <- max(Z, 0))
        --nonneg_mode shift   (Z <- Z - min(Z) per image)
        --encoder_model Conv64F_Local_ReLU / ResNet12_ReLU  (see backbone_relu.py)
    See README_NDCN.md for the trade-offs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y, eps=1e-12):
    """x such that softplus(x) == y, for y > 0."""
    return math.log(math.expm1(max(y, eps)))


# =====================================================================
# 0. Non-negativity
# =====================================================================
def make_nonneg(Z, mode='relu', eps=1e-8):
    """
    Z : B x L x C  (may contain negatives if the backbone uses LeakyReLU/GELU)
    """
    if mode == 'none':
        return Z
    elif mode == 'relu':
        return F.relu(Z)
    elif mode == 'shift':
        zmin = Z.amin(dim=(1, 2), keepdim=True)
        return Z - torch.clamp(zmin, max=0.0)
    elif mode == 'abs':
        return Z.abs()
    else:
        raise ValueError('unknown nonneg_mode: %s' % mode)


# =====================================================================
# 1. NDM : Non-negative Decomposition Module   (zero learnable parameters)
# =====================================================================
class NDM(nn.Module):
    """
    Factorise Z (B x L x C, Z >= 0) into W (B x L x r) and H (B x r x C),
    both non-negative, by NNDSVD initialisation + T multiplicative updates.

    Returns
        A       : B x L x r   row-normalised assignment maps  (rows sum to 1)
        H       : B x r x C   component basis vectors
        rec_err : B           relative reconstruction error ||Z-WH||_F/||Z||_F
    """

    def __init__(self, rank=5, n_iter=3, eps=1e-8, grad_last_only=True):
        super(NDM, self).__init__()
        self.r = rank
        self.T = n_iter
        self.eps = eps
        self.grad_last_only = grad_last_only

    # ---------------- NNDSVD (deterministic, seed-free, prototype-free) ------
    @torch.no_grad()
    def _nndsvd_init(self, Z):
        B, L, C = Z.shape
        r, eps = self.r, self.eps

        # economy SVD; r <= min(L, C) is guaranteed for the backbones we use
        U, S, Vh = torch.linalg.svd(Z.float(), full_matrices=False)
        U, S, Vh = U[:, :, :r], S[:, :r], Vh[:, :r, :]              # B x L x r, B x r, B x r x C

        W = torch.zeros(B, L, r, device=Z.device, dtype=Z.dtype)
        H = torch.zeros(B, r, C, device=Z.device, dtype=Z.dtype)

        # ---- leading component: guaranteed non-negative by Perron-Frobenius
        s0 = torch.sqrt(S[:, 0].clamp_min(eps))                     # B
        W[:, :, 0] = s0.unsqueeze(1) * U[:, :, 0].abs()
        H[:, 0, :] = s0.unsqueeze(1) * Vh[:, 0, :].abs()

        # ---- remaining components: keep the dominant positive/negative section
        for j in range(1, r):
            x, y = U[:, :, j], Vh[:, j, :]                          # B x L, B x C
            xp, xn = x.clamp_min(0.0), (-x).clamp_min(0.0)
            yp, yn = y.clamp_min(0.0), (-y).clamp_min(0.0)

            xpn = xp.norm(dim=1) + eps
            xnn = xn.norm(dim=1) + eps
            ypn = yp.norm(dim=1) + eps
            ynn = yn.norm(dim=1) + eps

            mp, mn = xpn * ypn, xnn * ynn                            # B
            use_p = (mp >= mn).float().unsqueeze(1)                  # B x 1

            u = use_p * (xp / xpn.unsqueeze(1)) + (1 - use_p) * (xn / xnn.unsqueeze(1))
            v = use_p * (yp / ypn.unsqueeze(1)) + (1 - use_p) * (yn / ynn.unsqueeze(1))
            sigma = torch.where(mp >= mn, mp, mn)                    # B

            lbd = torch.sqrt((S[:, j] * sigma).clamp_min(eps)).unsqueeze(1)
            W[:, :, j] = lbd * u
            H[:, j, :] = lbd * v

        return W.clamp_min(eps), H.clamp_min(eps)

    # ---------------- Lee & Seung multiplicative updates ---------------------
    def _mu_step(self, Z, W, H):
        eps = self.eps
        # H <- H * (W^T Z) / (W^T W H)
        num_h = torch.bmm(W.transpose(1, 2), Z)                      # B x r x C
        den_h = torch.bmm(torch.bmm(W.transpose(1, 2), W), H) + eps
        H = H * (num_h / den_h)

        # W <- W * (Z H^T) / (W H H^T)
        num_w = torch.bmm(Z, H.transpose(1, 2))                      # B x L x r
        den_w = torch.bmm(W, torch.bmm(H, H.transpose(1, 2))) + eps
        W = W * (num_w / den_w)
        return W, H

    def forward(self, Z, return_recon=False):
        eps = self.eps
        W, H = self._nndsvd_init(Z)

        if self.grad_last_only:
            # first T-1 iterations carry no gradient -> shallow graph
            with torch.no_grad():
                for _ in range(max(self.T - 1, 0)):
                    W, H = self._mu_step(Z.detach(), W, H)
            W, H = W.detach(), H.detach()
            if self.T >= 1:
                W, H = self._mu_step(Z, W, H)     # gradient flows to Z here
        else:
            for _ in range(self.T):
                W, H = self._mu_step(Z, W, H)

        # row-normalise W -> each descriptor distributes one unit of mass
        A = W / (W.sum(dim=2, keepdim=True) + eps)                   # B x L x r

        rec_err = None
        if return_recon:
            with torch.no_grad():
                rec = torch.bmm(W, H)
                rec_err = (Z - rec).flatten(1).norm(dim=1) / (Z.flatten(1).norm(dim=1) + eps)
        return A, H, rec_err

    @torch.no_grad()
    def recon_trajectory(self, Z):
        """Diagnostic: relative reconstruction error after 0..T updates."""
        eps = self.eps
        W, H = self._nndsvd_init(Z)
        zn = Z.flatten(1).norm(dim=1) + eps
        traj = [((Z - torch.bmm(W, H)).flatten(1).norm(dim=1) / zn).mean().item()]
        for _ in range(self.T):
            W, H = self._mu_step(Z, W, H)
            traj.append(((Z - torch.bmm(W, H)).flatten(1).norm(dim=1) / zn).mean().item())
        return traj


# =====================================================================
# 2. Alternative frontends (for the frontend-generality table)
# =====================================================================
class MuSigmaFrontend(nn.Module):
    """Parameter-free statistical (mu - sigma) threshold, softened by a sigmoid."""

    def __init__(self, lam=1.0, temp=10.0):
        super(MuSigmaFrontend, self).__init__()
        self.lam, self.temp = lam, temp

    def forward(self, Z, ref):
        # relevance of each descriptor to the reference direction
        s = F.cosine_similarity(F.normalize(Z, dim=2), F.normalize(ref, dim=1).unsqueeze(1), dim=2)
        mu = s.mean(dim=1, keepdim=True)
        sd = s.std(dim=1, keepdim=True)
        return torch.sigmoid(self.temp * (s - (mu - self.lam * sd)))


# =====================================================================
# 3. NDCN head
# =====================================================================
class NDCN_Metric(nn.Module):
    """
    Soft-mask decoupling + three-way logit + auxiliary losses.

    forward(x_query, x_support) -> logits (NQ x way)
                                   and, in training mode, an aux-loss dict.
    """

    def __init__(self, way_num=5, shot_num=1, neighbor_k=3,
                 rank=5, n_iter=3, beta=10.0, margin=0.3,
                 lambda_bg=0.05, lambda_div=0.01, lambda_mask=0.001,
                 score_mode='ref', frontend='ndm', nonneg_mode='relu',
                 alpha_norm='minmax', alpha_tau=8.0, div_margin=0.0,
                 pool_floor=0.05, mask_budget=0.0, uniform_mask=False,
                 use_bg_sep=True, use_qbg=True,
                 use_triplet=True, use_bgloss=True,
                 use_div=True, use_mask=True,
                 grad_last_only=True, eps=1e-8):
        super(NDCN_Metric, self).__init__()
        self.way_num, self.shot_num = way_num, shot_num
        self.eps, self.margin = eps, margin
        self.lambda_bg, self.lambda_div, self.lambda_mask = lambda_bg, lambda_div, lambda_mask
        self.score_mode, self.frontend, self.nonneg_mode = score_mode, frontend, nonneg_mode
        self.alpha_norm, self.alpha_tau, self.div_margin = alpha_norm, alpha_tau, div_margin
        self.pool_floor = pool_floor
        # set per episode by forward(); initialised here so that _mask() can be
        # called on its own, e.g. by the diagnostics in Test_NDCN.py
        self._alpha_range = None
        self.mask_budget = mask_budget
        self.uniform_mask = uniform_mask
        self.use_bg_sep, self.use_qbg = use_bg_sep, use_qbg
        self.use_triplet, self.use_bgloss = use_triplet, use_bgloss
        self.use_div, self.use_mask = use_div, use_mask

        if frontend == 'ndm':
            self.ndm = NDM(rank=rank, n_iter=n_iter, eps=eps, grad_last_only=grad_last_only)
        elif frontend == 'mustd':
            self.ndm = MuSigmaFrontend()
        else:
            raise ValueError('unknown frontend: %s' % frontend)

        # The model's ONLY learnable parameters.
        #
        # gamma_fg and gamma_bg must stay positive to act as temperatures.
        # Taking |x| does that but has zero subgradient at x = 0 in PyTorch, so
        # a parameter that reaches 0 stops receiving gradient and can never
        # recover; gamma_bg died this way in three of four runs. We store the
        # pre-activation instead and read the temperature through softplus,
        # whose gradient is 1/2 at 0 and never vanishes.
        self._raw_gamma_fg = nn.Parameter(torch.tensor(_inv_softplus(float(beta))))
        self._raw_gamma_bg = nn.Parameter(torch.tensor(_inv_softplus(float(beta) / 2.0)))
        self.beta_bg = nn.Parameter(torch.tensor(0.5))   # sign-free by design

        self._diag = {}

    # ---------------- learnable temperatures ---------------------------------
    @property
    def gamma_fg(self):
        return F.softplus(self._raw_gamma_fg)

    @property
    def gamma_bg(self):
        return F.softplus(self._raw_gamma_bg)

    # ---------------- helpers ------------------------------------------------
    @staticmethod
    def _flatten(x):
        """B x C x H x W  ->  B x L x C"""
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1).contiguous()

    def _rescale_alpha(self, a):
        """
        Spread the component scores over [0, 1].

        Both H and ref are non-negative, so cos(h_v, ref) >= 0 by construction
        and Eq.(7)'s 0.5*(cos+1) can only reach [0.5, 1]. Measured on real
        Conv-4 features the scores land in [0.66, 1.00], which forces the soft
        mask above 0.71 everywhere and effectively disables the decoupling.
        Only the ranking of the components carries information, so we rescale.
        """
        if self.alpha_norm == 'none':
            return a
        if self.alpha_norm == 'minmax':
            lo = a.amin(dim=1, keepdim=True)
            hi = a.amax(dim=1, keepdim=True)
            rng = hi - lo
            flat = rng < 1e-6                              # all components tied
            return torch.where(flat, torch.full_like(a, 0.5), (a - lo) / (rng + self.eps))
        if self.alpha_norm == 'zsig':
            return torch.sigmoid(self.alpha_tau * (a - a.mean(dim=1, keepdim=True)))
        if self.alpha_norm in ('setminmax', 'supminmax'):
            # Episode-level variants. The per-image form rescales each image
            # against its own components, so an image whose components are all
            # weak looks identical to one whose components are all strong. A
            # shared range instead lets the mask reflect how strong an image is
            # relative to its context.
            #
            # 'setminmax' takes the range from whatever batch is passed in, so a
            # query's mask depends on the other queries in the same batch. That
            # is TRANSDUCTIVE inference and is not comparable to inductive
            # baselines. 'supminmax' takes the range from the support set only,
            # which is legitimately available at test time, so each query is
            # still classified independently.
            if self.alpha_norm == 'supminmax' and self._alpha_range is not None:
                lo, hi = self._alpha_range
            else:
                lo, hi = a.min(), a.max()
            rng = hi - lo
            if float(rng.detach()) < 1e-6:
                return torch.full_like(a, 0.5)
            return ((a - lo) / (rng + self.eps)).clamp(0.0, 1.0)
        raise ValueError('unknown alpha_norm: %s' % self.alpha_norm)

    def _alpha(self, H, ref, A=None, Z=None):
        """
        H   : B x r x C   component basis
        ref : B x C       reference direction (ignored when score_mode='free')
        """
        if self.score_mode == 'ref':
            cos = F.cosine_similarity(F.normalize(H, dim=2),
                                      F.normalize(ref, dim=1).unsqueeze(1), dim=2)   # B x r
            self._raw_s = 0.5 * (cos + 1.0)
            return self._rescale_alpha(self._raw_s)
        elif self.score_mode == 'centered':
            # Every component shares a large DC part with the reference, because
            # for a non-negative matrix the leading NMF factor IS the mean
            # direction (measured: cos(h_0, ref) = 0.997 vs 0.37-0.50 for the
            # rest, in 100% of images). Scoring the raw vectors therefore ranks
            # components by NMF index, not by object-ness. We remove the shared
            # component from both sides before scoring.
            Hc = H - H.mean(dim=1, keepdim=True)                                     # B x r x C
            rc = ref - ref.mean(dim=0, keepdim=True)                                 # B x C
            cos = F.cosine_similarity(F.normalize(Hc, dim=2),
                                      F.normalize(rc, dim=1).unsqueeze(1), dim=2)
            return self._rescale_alpha(0.5 * (cos + 1.0))
        elif self.score_mode == 'free':
            # activation-norm mass per component; no reference at all.
            # This branch used to hard-code a per-image min-max rescaling and
            # ignore --alpha_norm entirely. It now routes through the same
            # rescaling as the other modes, so the flag applies here too.
            nrm = Z.norm(dim=2)                                                       # B x L
            num = (A * nrm.unsqueeze(2)).sum(dim=1)                                   # B x r
            den = A.sum(dim=1) + self.eps
            raw = num / den
            self._raw_s = raw            # kept so supminmax can reuse the support range
            return self._rescale_alpha(raw)
        else:
            raise ValueError('unknown score_mode: %s' % self.score_mode)

    def _mask(self, Z, ref):
        """Z : B x L x C  ->  soft mask B x L  (and diagnostics)"""
        if self.uniform_mask:
            # Baseline: every position carries the same weight, so the masked
            # pooling of Eq.(pool) reduces exactly to average pooling. Used to
            # measure what the mask contributes, independently of how the
            # component scores are normalised.
            return Z.new_ones(Z.size(0), Z.size(1)), None, None
        if getattr(self, 'mask_mode', 'ndm') == 'pointwise':
            # No NMF/SVD, no giant identity assignment matrix.
            self._raw_s = Z.norm(dim=2)
            M = self._rescale_alpha(self._raw_s)
            return M, None, None
        if self.frontend == 'mustd':
            M = self.ndm(Z, ref)
            return M, None, None
        A, H, _ = self.ndm(Z)                                    # B x L x r, B x r x C
        alpha = self._alpha(H, ref, A=A, Z=Z)                    # B x r
        M = torch.clamp((A * alpha.unsqueeze(1)).sum(dim=2), 0.0, 1.0)
        return M, alpha, A

    @staticmethod
    def _pool(Z, M, eps, floor=0.05):
        """
        Mask-weighted pooling with a floored denominator.

        The floor is a numerical guard only: it stops sum_l M_l from reaching
        zero and producing a NaN. It does NOT damp the gradient, because the
        downstream metric is cosine and therefore scale-invariant -- shrinking
        the pooled vector by a constant is exactly cancelled by the
        normalisation, as measured. What actually grows as the mask sparsifies
        is gradient CONCENTRATION: the same signal is carried by fewer
        positions, so each surviving position takes a larger step. Use
        --clip_grad and keep mask_mean above ~0.2 to control that.
        """
        L = M.size(1)
        den_fg = M.sum(1, keepdim=True).clamp_min(floor * L)
        den_bg = (1 - M).sum(1, keepdim=True).clamp_min(floor * L)
        fg = (M.unsqueeze(2) * Z).sum(1) / den_fg
        bg = ((1 - M).unsqueeze(2) * Z).sum(1) / den_bg
        return fg, bg

    @staticmethod
    def _cos_dist(a, b):
        """a: N x C, b: M x C -> N x M cosine distance in [0, 2]"""
        return 1.0 - torch.mm(F.normalize(a, dim=1), F.normalize(b, dim=1).t())

    # ---------------- main ---------------------------------------------------
    def forward(self, x_query, x_support, target=None):
        eps, way, shot = self.eps, self.way_num, self.shot_num

        Zq = make_nonneg(self._flatten(x_query), self.nonneg_mode, eps)      # NQ x L x C
        Zs = make_nonneg(self._flatten(x_support), self.nonneg_mode, eps)    # (way*shot) x L x C
        NQ, L, C = Zq.shape

        # ---- reference directions ------------------------------------------
        # query : its own descriptor mean (no label available)
        ref_q = Zq.mean(dim=1)                                               # NQ x C
        # support: class-level descriptor mean, shared by the K images of a class
        Zs_cls = Zs.view(way, shot, L, C)
        ref_s_cls = Zs_cls.mean(dim=(1, 2))                                  # way x C
        ref_s = ref_s_cls.unsqueeze(1).expand(way, shot, C).reshape(way * shot, C)

        # ---- soft masks -----------------------------------------------------
        # Support first: under alpha_norm='supminmax' the query is rescaled with
        # the support set's range, so each query stays independent of the others.
        self._alpha_range = None
        Ms, alpha_s, _ = self._mask(Zs, ref_s)                               # (way*shot) x L
        if self.alpha_norm == 'supminmax' and getattr(self, '_raw_s', None) is not None:
            self._alpha_range = (self._raw_s.min().detach(), self._raw_s.max().detach())
        Mq, alpha_q, _ = self._mask(Zq, ref_q)                               # NQ x L
        self._alpha_range = None

        # ---- decoupling -----------------------------------------------------
        q_fg, q_bg = self._pool(Zq, Mq, eps, self.pool_floor)                                 # NQ x C

        # class prototypes: mass-weighted pooling jointly over the K support images
        Ms_c = Ms.view(way, shot * L)
        Zs_c = Zs.view(way, shot * L, C)
        _M = Ms_c.size(1)
        p_fg = ((Ms_c.unsqueeze(2) * Zs_c).sum(1)
                / Ms_c.sum(1, keepdim=True).clamp_min(self.pool_floor * _M))
        p_bg = (((1 - Ms_c).unsqueeze(2) * Zs_c).sum(1)
                / (1 - Ms_c).sum(1, keepdim=True).clamp_min(self.pool_floor * _M))

        # ---- three-way logit ------------------------------------------------
        d_fg = self._cos_dist(q_fg, p_fg)                                    # NQ x way
        logit = -self.gamma_fg * d_fg
        d_sep = d_qbg = None
        if self.use_bg_sep:
            d_sep = self._cos_dist(q_fg, p_bg)
            logit = logit + self.gamma_bg * d_sep
        if self.use_qbg:
            d_qbg = self._cos_dist(q_bg, p_bg)
            logit = logit - self.beta_bg * d_qbg

        if getattr(self, 'record_diagnostics', True):
            # diagnostics (cheap, always recorded)
            # kept so that the diagnostics report the mask that was actually
            # applied, rather than one recomputed after _alpha_range is cleared
            self._last_Ms = Ms.detach()
            self._last_Mq, self._last_alpha_q = Mq.detach(), (
                alpha_q.detach() if alpha_q is not None else None)

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

        aux = self._aux_losses(q_fg, q_bg, p_fg, p_bg, Mq, target)
        return logit, aux

    @staticmethod
    def _offdiag_cos(P):
        Pn = F.normalize(P, dim=1)
        G = torch.mm(Pn, Pn.t())
        n = G.size(0)
        return (G.sum() - G.diag().sum()) / (n * (n - 1) + 1e-8)

    # ---------------- auxiliary objectives -----------------------------------
    def _aux_losses(self, q_fg, q_bg, p_fg, p_bg, Mq, target):
        eps, D = self.eps, self.margin
        NQ = q_fg.size(0)
        idx = torch.arange(NQ, device=q_fg.device)

        d_qp_fg = self._cos_dist(q_fg, p_fg)                    # NQ x way
        d_pos = d_qp_fg[idx, target]                            # NQ

        out = {}

        # --- (1) inter-class foreground negative (hardest) --------------------
        if self.use_triplet:
            masked = d_qp_fg.clone()
            masked[idx, target] = float('inf')
            d_neg = masked.min(dim=1).values
            out['triplet'] = F.relu(D + d_pos - d_neg).mean()
        else:
            out['triplet'] = q_fg.new_zeros(())

        # --- (2) intra-class background negative ------------------------------
        if self.use_bgloss:
            d_qfg_pbg = self._cos_dist(q_fg, p_bg)[idx, target]
            d_qbg_pbg = self._cos_dist(q_bg, p_bg)[idx, target]
            out['bg'] = (F.relu(D + d_pos - d_qfg_pbg) + 0.5 * d_qbg_pbg).mean()
        else:
            out['bg'] = q_fg.new_zeros(())

        # --- (3) prototype diversity ------------------------------------------
        if self.use_div:
            Pn = F.normalize(p_fg, dim=1)
            G = torch.mm(Pn, Pn.t())
            n = G.size(0)
            off = ~torch.eye(n, dtype=torch.bool, device=G.device)
            if self.div_margin > 0:
                # non-negative prototypes always have cos >= 0, so a target of 0
                # is unreachable and the plain penalty fights the geometry.
                out['div'] = (F.relu(G[off] - self.div_margin) ** 2).mean()
            else:
                I = torch.eye(n, device=G.device, dtype=G.dtype)
                out['div'] = ((G - I) ** 2).mean()
        else:
            out['div'] = q_fg.new_zeros(())

        # --- (4) mask sparsity -------------------------------------------------
        # The plain form penalises the mask mean linearly and therefore keeps
        # pushing it toward zero with nothing to stop it; both 5-shot runs
        # sparsified until the prototypes collapsed. The budget form stops
        # pushing once the mask is already at or below the target occupancy.
        if not self.use_mask or self.uniform_mask:
            out['mask'] = q_fg.new_zeros(())
        elif self.mask_budget > 0:
            out['mask'] = F.relu(Mq.mean() - self.mask_budget)
        else:
            out['mask'] = Mq.mean()

        out['total'] = (out['triplet']
                        + self.lambda_bg * out['bg']
                        + self.lambda_div * out['div']
                        + self.lambda_mask * out['mask'])
        return out

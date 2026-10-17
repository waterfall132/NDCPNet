"""Global cosine baseline and candidate local evidence branches.

No labels, images from other queries, or mutable episode cache enter this head.
The caller supplies the project's existing NDM; its parameters are not copied.
"""
import torch
import torch.nn.functional as F


BRANCHES = ('D1', 'D1q', 'D2', 'D3')


def cosine_matrix(q, p):
    return F.normalize(q, dim=-1) @ F.normalize(p, dim=-1).T


def gap_scores(fq, fs, way=5, shot=1):
    p = fs.mean((2, 3)).reshape(way, shot, -1).mean(1)
    return cosine_matrix(fq.mean((2, 3)), p)


def discrimination_scores(z, prototypes):
    """z: Q,L,C; prototypes: N,C. Returns Q,L,N; no target labels."""
    n = prototypes.shape[0]
    if n < 2:
        raise ValueError('Need at least two candidate classes')
    sim = F.normalize(z, dim=-1) @ F.normalize(prototypes, dim=-1).T
    return sim - (sim.sum(-1, keepdim=True) - sim) / (n - 1)


def smooth_evidence(d, assignment, eps=1e-8):
    """NDM component average followed by spatial propagation, not max matching."""
    den = assignment.sum(1).unsqueeze(-1) + eps
    component = assignment.transpose(1, 2) @ d / den
    return assignment @ component


def attention_scores(z, prototypes, evidence, temperature=0.2):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    weights = (evidence / temperature).softmax(dim=1)
    vectors = weights.transpose(1, 2) @ z  # Q,N,C
    scores = (F.normalize(vectors, dim=-1) *
              F.normalize(prototypes, dim=-1).unsqueeze(0)).sum(-1)
    return scores, weights


def mix_scores(gap, local, weight):
    if not 0 <= weight <= 1:
        raise ValueError('weight must be in [0,1]')
    # Exact recovery, including floating point and zero-vector edge cases.
    if weight == 0:
        return gap
    return (1 - weight) * gap + weight * local


class GAPLocalEvidence(torch.nn.Module):
    """Differentiable head; frozen evaluation is enforced by the evaluator.

    D1: original energy mask on BOTH support and query, used as a local branch.
    D1q: same query energy mask, but GAP support prototypes (controlled bridge).
    D2: query class-contrast attention, GAP support prototypes, no NDM.
    D3: D2 with NDM smoothing of class-contrast evidence.
    All scores are unscaled cosine scores; a training wrapper may multiply the
    final mixed scores by ONE positive temperature before cross entropy.
    """
    def __init__(self, ndm, way=5, shot=1, temperature=0.2,
                 eps=1e-8, pool_floor=0.05):
        super().__init__()
        self.ndm = ndm
        self.way, self.shot = way, shot
        self.temperature, self.eps, self.pool_floor = temperature, eps, pool_floor

    def _energy(self, z, a):
        return (a * z.norm(dim=-1).unsqueeze(-1)).sum(1) / (a.sum(1) + self.eps)

    def _rescale(self, raw, lo, hi):
        rng = hi - lo
        return torch.where(rng < 1e-6, torch.full_like(raw, 0.5),
                           ((raw-lo)/(rng+self.eps)).clamp(0, 1))

    def _pool(self, z, mask):
        den = mask.sum(1, keepdim=True).clamp_min(self.pool_floor * mask.shape[1])
        return (z * mask.unsqueeze(-1)).sum(1) / den

    def branches(self, fq, fs, names=BRANCHES):
        names = tuple(names)
        if set(names) - set(BRANCHES):
            raise ValueError('Unknown branch')
        if fs.shape[0] != self.way * self.shot:
            raise ValueError('Support features must be class-major, way*shot images')
        if fq.ndim != 4 or fs.ndim != 4 or fq.shape[1] != fs.shape[1]:
            raise ValueError('Expected Q,C,H,W and (way*shot),C,H,W')
        if not torch.isfinite(fq).all() or not torch.isfinite(fs).all():
            raise ValueError('Nonfinite features')
        if (fq < 0).any() or (fs < 0).any():
            raise ValueError('This version requires nonnegative ReLU feature maps')
        zq = fq.flatten(2).transpose(1, 2).contiguous()
        zs = fs.flatten(2).transpose(1, 2).contiguous()
        p = fs.mean((2, 3)).reshape(self.way, self.shot, -1).mean(1)
        result = {'D0': gap_scores(fq, fs, self.way, self.shot)}
        diagnostics = {}
        need_energy = bool(set(names) & {'D1', 'D1q'})
        aq = None
        if need_energy or 'D3' in names:
            aq, _, _ = self.ndm(zq)
        if need_energy:
            ass, _, _ = self.ndm(zs)
            raw_s = self._energy(zs, ass)
            lo, hi = raw_s.min(), raw_s.max()
            ms = (ass * self._rescale(raw_s, lo, hi).unsqueeze(1)).sum(-1).clamp(0, 1)
            # Match the original support-derived calibration and detached range.
            mq = (aq * self._rescale(self._energy(zq, aq), lo.detach(), hi.detach())
                  .unsqueeze(1)).sum(-1).clamp(0, 1)
            qlocal = self._pool(zq, mq)
            if 'D1q' in names:
                result['D1q'] = cosine_matrix(qlocal, p)
            if 'D1' in names:
                pcs = self._pool(zs.reshape(self.way, -1, zs.shape[-1]),
                                 ms.reshape(self.way, -1))
                result['D1'] = cosine_matrix(qlocal, pcs)
            diagnostics['energy_query_mask_mean'] = mq.mean()
            diagnostics['energy_zero_query_masks'] = (mq.sum(1) == 0).sum()
        if set(names) & {'D2', 'D3'}:
            d = discrimination_scores(zq, p)
            for name in ('D2', 'D3'):
                if name not in names:
                    continue
                evidence = d if name == 'D2' else smooth_evidence(d, aq, self.eps)
                result[name], w = attention_scores(zq, p, evidence, self.temperature)
                diagnostics[name + '_effective_positions'] = (1 / w.square().sum(1)).mean()
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite branch scores')
        return result, diagnostics

    def forward(self, fq, fs, branch='D2', weight=0.1):
        if weight == 0 or branch == 'D0':
            return gap_scores(fq, fs, self.way, self.shot)
        scores, _ = self.branches(fq, fs, (branch,))
        return mix_scores(scores['D0'], scores[branch], weight)

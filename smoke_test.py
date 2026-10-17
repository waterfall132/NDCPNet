#!/usr/bin/env python3
"""Sanity checks for NDCN. Run:  python smoke_test.py"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
from models.ndcn import NDM, NDCN_Metric, make_nonneg

torch.manual_seed(0)
ok = lambda name, cond: print(('  PASS  ' if cond else '  FAIL  ') + name)

print('=== 1. NDM: non-negativity, partition, monotone descent ===')
for (L, C) in [(25, 640), (441, 64)]:
    Z = F.relu(torch.randn(8, L, C))
    ndm = NDM(rank=5, n_iter=3)
    A, H, err = ndm(Z, return_recon=True)
    ok('A >= 0                       (L=%d)' % L, bool((A >= -1e-6).all()))
    ok('H >= 0                       (L=%d)' % L, bool((H >= -1e-6).all()))
    ok('rows of A sum to 1           (L=%d)' % L, torch.allclose(A.sum(-1), torch.ones(8, L), atol=1e-4))
    traj = ndm.recon_trajectory(Z)
    ok('recon error non-increasing   (L=%d)' % L, all(traj[i] >= traj[i+1] - 1e-6 for i in range(len(traj)-1)))
    print('        trajectory: ' + ' -> '.join('%.4f' % t for t in traj))

print('\n=== 1b. NDM on genuinely rank-5 data (error should collapse) ===')
Wt = F.relu(torch.randn(4, 25, 5)); Ht = F.relu(torch.randn(4, 5, 640))
Zlr = torch.bmm(Wt, Ht)
traj = NDM(rank=5, n_iter=8).recon_trajectory(Zlr)
print('        trajectory: ' + ' -> '.join('%.4f' % t for t in traj))
# multiplicative updates are first-order and converge slowly; T=3 (the paper's
# setting) only partially converges, which is expected and harmless.
ok('error reduced by >25% in 8 iters', traj[-1] < 0.75 * traj[0])

print('\n=== 2. NDM determinism (no random seed inside) ===')
Z = F.relu(torch.randn(4, 25, 640))
A1, _, _ = NDM(5, 3)(Z)
A2, _, _ = NDM(5, 3)(Z)
ok('two calls give identical A', torch.allclose(A1, A2, atol=1e-6))

print('\n=== 3. NDCN head: forward shapes, backward, param count ===')
way, shot, q, C, H, W = 5, 1, 15, 640, 5, 5
head = NDCN_Metric(way_num=way, shot_num=shot, rank=5, n_iter=3)
lq = torch.randn(way * q, C, H, W, requires_grad=True)      # leaf
ls = torch.randn(way * shot, C, H, W, requires_grad=True)   # leaf
xq, xs = F.relu(lq), F.relu(ls)
tgt = torch.arange(way).repeat_interleave(q)

head.train()
logit, aux = head(xq, xs, tgt)
ok('logit shape == (NQ, way)', tuple(logit.shape) == (way * q, way))
ok('aux keys complete', set(aux) >= {'triplet', 'bg', 'div', 'mask', 'total'})
loss = nn.CrossEntropyLoss()(logit, tgt) + 0.1 * aux['total']
loss.backward()
ok('gradient reaches query features', lq.grad is not None and float(lq.grad.abs().sum()) > 0)
ok('gradient reaches support features', ls.grad is not None and float(ls.grad.abs().sum()) > 0)
# gamma_fg / gamma_bg are softplus properties now; the leaves are _raw_*
ok('gradient reaches gamma_fg', head._raw_gamma_fg.grad is not None and float(head._raw_gamma_fg.grad.abs()) > 0)
ok('gradient reaches gamma_bg', head._raw_gamma_bg.grad is not None and float(head._raw_gamma_bg.grad.abs()) > 0)
ok('gradient reaches beta_bg', head.beta_bg.grad is not None and float(head.beta_bg.grad.abs()) > 0)
ok('temperatures are strictly positive', float(head.gamma_fg) > 0 and float(head.gamma_bg) > 0)
xq, xs = xq.detach(), xs.detach()
n_par = sum(p.numel() for p in head.parameters())
ok('head has exactly 3 parameters (got %d)' % n_par, n_par == 3)

head.eval()
out = head(xq.detach(), xs.detach())
ok('eval mode returns logits only', torch.is_tensor(out))

print('\n=== 4. 5-shot path ===')
shot = 5
head5 = NDCN_Metric(way_num=way, shot_num=shot); head5.train()
xs5 = F.relu(torch.randn(way * shot, C, H, W))
l5, a5 = head5(F.relu(torch.randn(way * q, C, H, W)), xs5, tgt)
ok('5-shot logit shape', tuple(l5.shape) == (way * q, way))

print('\n=== 5. Ablation switches ===')
h_noqbg = NDCN_Metric(way_num=way, shot_num=1, use_qbg=False); h_noqbg.train()
l_a, _ = h_noqbg(xq.detach(), xs.detach(), tgt)
h_full = NDCN_Metric(way_num=way, shot_num=1, use_qbg=True); h_full.train()
l_b, _ = h_full(xq.detach(), xs.detach(), tgt)
ok('--no_qbg changes the logit', not torch.allclose(l_a, l_b))
h_free = NDCN_Metric(way_num=way, shot_num=1, score_mode='free'); h_free.train()
l_c, _ = h_free(xq.detach(), xs.detach(), tgt)
ok('score_mode=free runs', tuple(l_c.shape) == (way * q, way))
h_ms = NDCN_Metric(way_num=way, shot_num=1, frontend='mustd'); h_ms.train()
l_d, _ = h_ms(xq.detach(), xs.detach(), tgt)
ok('frontend=mustd runs', tuple(l_d.shape) == (way * q, way))

print('\n=== 6. Negative-input handling (LeakyReLU backbones) ===')
Zneg = torch.randn(4, 25, 64)
ok('input really has negatives', bool((Zneg < 0).any()))
for mode in ['relu', 'shift']:
    Zp = make_nonneg(Zneg, mode)
    ok('nonneg_mode=%-6s -> Z >= 0' % mode, bool((Zp >= -1e-6).all()))

print('\n=== 7. Timing (CPU, ResNet12 shape) ===')
import time
head_t = NDCN_Metric(way_num=5, shot_num=1); head_t.eval()
xq_t = F.relu(torch.randn(75, 640, 5, 5)); xs_t = F.relu(torch.randn(5, 640, 5, 5))
with torch.no_grad():
    head_t(xq_t, xs_t)
    t0 = time.time()
    for _ in range(10):
        head_t(xq_t, xs_t)
    print('        %.1f ms / episode (CPU; GPU will be far faster)' % ((time.time() - t0) * 100))
print('\nAll checks done.')

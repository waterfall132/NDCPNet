#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired analysis of the 2x2 mask test.

    python paired_stats.py --mask_ckpt DIR_A --unif_ckpt DIR_B

Test_NDCN.py now writes the accuracy of every individual episode to
episode_acc*.npy. Because all four cells were evaluated with --seed 0,
episode i is the same task in all four files, so the differences can be
analysed as paired samples.

Why this is needed: the "h" printed by Test_NDCN.py is the mean of five
marginal confidence intervals, one per round. It is not a pooled interval
over all 10,000 episodes, and it says nothing about the uncertainty of a
DIFFERENCE between two systems. A gap of 0.08 cannot be judged against it.

The script reports, for each contrast, the paired mean difference with a
bootstrap confidence interval and the fraction of episodes on which each
system wins. It deliberately does not print a verdict of "equivalent":
failing to detect a difference is not evidence of equivalence, which would
require an equivalence bound fixed in advance.
"""

import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--mask_ckpt', required=True, help='checkpoint dir trained WITH the mask')
ap.add_argument('--unif_ckpt', required=True, help='checkpoint dir trained with a uniform mask')
ap.add_argument('--n_boot', type=int, default=10000)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

CELLS = {
    'A  train mask / test mask':   (args.mask_ckpt, 'episode_acc_on.npy'),
    'B  train mask / test unif':   (args.mask_ckpt, 'episode_acc_off.npy'),
    'C  train unif / test mask':   (args.unif_ckpt, 'episode_acc_on.npy'),
    'D  train unif / test unif':   (args.unif_ckpt, 'episode_acc_off.npy'),
}

acc = {}
for name, (d, f) in CELLS.items():
    p = os.path.join(d, f)
    if not os.path.isfile(p):
        print('missing: %s' % p)
        print('  rerun that cell with the updated Test_NDCN.py so the per-episode'
              ' accuracies are written.')
        raise SystemExit(1)
    acc[name[0]] = np.load(p).astype(np.float64)

n = min(len(v) for v in acc.values())
if len({len(v) for v in acc.values()}) > 1:
    print('cells have different episode counts %s; truncating to %d'
          % ({k: len(v) for k, v in acc.items()}, n))
acc = {k: v[:n] for k, v in acc.items()}

rng = np.random.default_rng(args.seed)
idx = rng.integers(0, n, size=(args.n_boot, n))


def paired(x, y):
    d = x - y
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi, float((d > 0).mean()), float((d == 0).mean())


print('\n%d paired episodes, %d bootstrap resamples\n' % (n, args.n_boot))
print('%-28s %8s   %8s' % ('cell', 'mean', 'pooled 95% CI'))
for k in 'ABCD':
    v = acc[k]
    h = 1.96 * v.std(ddof=1) / np.sqrt(n)
    lbl = [name for name in CELLS if name[0] == k][0]
    print('%-28s %8.2f   +/- %.2f' % (lbl, v.mean(), h))

print('\n%-34s %9s %-22s %s' % ('paired contrast', 'diff', '95% CI of the diff', 'win rate'))
for lbl, x, y in [('A - B   mask on vs off, mask-trained', 'A', 'B'),
                  ('D - C   unif on vs off, unif-trained', 'D', 'C'),
                  ('A - D   full system vs uniform system', 'A', 'D')]:
    m, lo, hi, win, tie = paired(acc[x], acc[y])
    sig = 'excludes 0' if lo > 0 or hi < 0 else 'includes 0'
    print('%-34s %+9.2f [%+.2f, %+.2f] %-9s %.1f%% wins, %.1f%% ties'
          % (lbl, m, lo, hi, sig, 100 * win, 100 * tie))

print("""
Reading the last row: a CI that includes zero means no difference was
detected in this comparison. That is not the same as the two systems being
equivalent; establishing equivalence needs a margin chosen in advance and a
test designed for it. Report it as "no system-level improvement detected".
""")

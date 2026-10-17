#!/usr/bin/env python3
"""Report the four-cell pooling table and paired confidence intervals."""

import argparse
import numpy as np


def load(path):
    x = np.load(path)['accuracy'].astype(np.float64)
    if x.ndim != 2:
        raise ValueError('%s must contain a [round, episode] accuracy array' % path)
    return x


def mean_ci(x):
    flat = x.reshape(-1)
    h = 1.96 * flat.std(ddof=1) / np.sqrt(flat.size)
    return flat.mean(), h


def paired_delta(a, b):
    if a.shape != b.shape:
        raise ValueError('paired arrays have different shapes: %s versus %s'
                         % (a.shape, b.shape))
    return mean_ci(a - b)


parser = argparse.ArgumentParser()
parser.add_argument('--trainmask_testmask', required=True)
parser.add_argument('--trainmask_testuniform', required=True)
parser.add_argument('--trainuniform_testmask', required=True)
parser.add_argument('--trainuniform_testuniform', required=True)
args = parser.parse_args()

a = load(args.trainmask_testmask)
b = load(args.trainmask_testuniform)
c = load(args.trainuniform_testmask)
d = load(args.trainuniform_testuniform)

for name, x in [('train mask / test mask', a),
                ('train mask / test uniform', b),
                ('train uniform / test mask', c),
                ('train uniform / test uniform', d)]:
    m, h = mean_ci(x)
    print('%-31s %7.3f +/- %.3f' % (name, m, h))

print('\nPaired differences (first minus second)')
for name, x, y in [('mask effect after mask training', a, b),
                   ('mask effect after uniform training', c, d),
                   ('end-to-end mask system', a, d),
                   ('training effect under mask test', a, c)]:
    m, h = paired_delta(x, y)
    print('%-31s %+7.3f +/- %.3f' % (name, m, h))

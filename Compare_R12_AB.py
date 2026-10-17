#!/usr/bin/env python3
"""Verify paired input provenance and compare fixed-step validation results."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def interval(values):
    x = np.asarray(values, dtype=float)
    h = stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    return {'mean': float(x.mean()), 'ci95_low': float(x.mean()-h), 'ci95_high': float(x.mean()+h)}


def compare(root):
    pa, pb = read(root/'A/provenance.json'), read(root/'B/provenance.json')
    for key in ['pretrained_sha256', 'source_sha256', 'initial_model_sha256',
                'val_manifest_sha256', 'val_transform', 'torch_version']:
        if pa[key] != pb[key]:
            raise ValueError('Unpaired provenance: ' + key)
    allowed = {'arm', 'outf', 'uniform_mask', 'no_triplet', 'no_div', 'no_mask'}
    if set(pa['config']) != set(pb['config']):
        raise ValueError('Different configuration keys')
    for k in pa['config']:
        if k not in allowed and pa['config'][k] != pb['config'][k]:
            raise ValueError('Unexpected config difference: ' + k)
    ta, tb = read(root/'A/train_trace.json'), read(root/'B/train_trace.json')
    if ta != tb:
        raise ValueError('Training inputs/order/LR differ. Do not call this an A/B paired comparison.')
    if not ta:
        raise ValueError('No training updates found')
    ra, rb = read(root/'A/summary.json'), read(root/'B/summary.json')
    if ra['last_step'] != rb['last_step'] or len(ta) != ra['last_step']:
        raise ValueError('Unequal or incomplete update budgets')
    if pa['arm'] != 'A' or pb['arm'] != 'B':
        raise ValueError('Expected A=mask/full and B=uniform/CE')
    for k in ['uniform_mask', 'no_triplet', 'no_div', 'no_mask']:
        if pa['config'][k] or not pb['config'][k]:
            raise ValueError('Wrong A/B ablation settings: ' + k)
    files_a = {p.name for p in (root/'A').glob('eval_step_*.npz')}
    files_b = {p.name for p in (root/'B').glob('eval_step_*.npz')}
    if files_a != files_b or 'eval_step_000000.npz' not in files_a:
        raise ValueError('Different evaluation steps')
    with np.load(root/'A/eval_step_000000.npz') as z:
        p0 = z['gap'].copy()
    comparisons = []
    expected_n = pa['config']['episode_val_num']
    print('Input hashes verified: same initial weights, tasks, augmented inputs and LR schedule.')
    for name in sorted(files_a):
        with np.load(root/'A'/name) as za, np.load(root/'B'/name) as zb:
            data = {'A_gap': za['gap'].copy(), 'A_mask': za['mask'].copy(),
                    'B_gap': zb['gap'].copy(), 'B_mask': zb['mask'].copy()}
        if any(x.shape != (expected_n,) or not np.isfinite(x).all() for x in data.values()):
            raise ValueError('Invalid episode score array')
        step = int(name.split('_')[-1].split('.')[0])
        row = {'step': step, 'accuracy': {k: interval(v) for k,v in data.items()},
               'paired_deltas': {
                   'A_mask_minus_P0_gap': interval(data['A_mask']-p0),
                   'B_gap_minus_P0_gap': interval(data['B_gap']-p0),
                   'A_mask_minus_B_gap': interval(data['A_mask']-data['B_gap']),
                   'A_gap_minus_B_gap': interval(data['A_gap']-data['B_gap']),
                   'A_mask_minus_A_gap': interval(data['A_mask']-data['A_gap'])}}
        comparisons.append(row)
        delta = row['paired_deltas']['A_mask_minus_B_gap']
        print('step=%d | A mask %.2f, A GAP %.2f | B GAP %.2f, B mask %.2f | A mask - B GAP %+.2f [%+.2f,%+.2f]' %
              (step, data['A_mask'].mean(), data['A_gap'].mean(), data['B_gap'].mean(),
               data['B_mask'].mean(), delta['mean'], delta['ci95_low'], delta['ci95_high']))
    result = {'pairing_verified': True, 'updates_per_arm': len(ta), 'split': 'val',
              'comparisons': comparisons,
              'note': 'Exploratory fixed-checkpoint task CIs; not training-seed uncertainty or final test results.'}
    (root/'paired_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', root/'paired_summary.json')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./results/R12_ABcheck0906')
    compare(Path(p.parse_args().root))

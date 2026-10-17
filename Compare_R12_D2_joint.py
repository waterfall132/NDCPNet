#!/usr/bin/env python3
"""Paired validation comparison: independent GAP+CE vs jointly trained D2+CE."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def interval(x):
    x = np.asarray(x, dtype=float)
    h = float(stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
    return dict(mean=float(x.mean()), ci95_low=float(x.mean()-h), ci95_high=float(x.mean()+h))


def compare(root, expected_steps=4000):
    arms = ('GAP', 'D2')
    p = {a: read(root/a/'provenance.json') for a in arms}
    s = {a: read(root/a/'summary.json') for a in arms}
    for a in arms:
        if p[a]['arm'] != a or s[a]['arm'] != a or s[a]['split'] != 'val' or p[a]['test_split_used']:
            raise ValueError('Wrong arm/split')
        cfg = p[a]['config']
        if cfg['experiment'] != 'D2_joint_v1' or cfg['d2_weight'] != .25 or cfg['d2_temperature'] != .2:
            raise ValueError('Wrong experiment or changed D2 constants')
        if not all(cfg[k] for k in ('uniform_mask','no_triplet','no_div','no_mask','no_bgloss','no_bg_sep','no_qbg')):
            raise ValueError('Both arms must disable old mask and all auxiliary losses')
        if cfg['epochs'] != 30 or cfg['episode_train_num'] != 4000 or cfg.get('freeze_bn_stats', False):
            raise ValueError('Changed schedule/BN protocol')
        if s[a]['selection_metric'] != ('gap' if a == 'GAP' else 'd2'):
            raise ValueError('Wrong selection metric')
    for k in ('pretrained_sha256','source_sha256','initial_model_sha256','val_manifest_sha256','val_transform','torch_version','csv_sha256'):
        if p['GAP'][k] != p['D2'][k]:
            raise ValueError('Unpaired provenance: ' + k)
    if set(p['GAP']['config']) != set(p['D2']['config']):
        raise ValueError('Config keys differ')
    for k in p['GAP']['config']:
        if k not in ('arm','outf') and p['GAP']['config'][k] != p['D2']['config'][k]:
            raise ValueError('Config difference: ' + k)
    trace = {a: read(root/a/'train_trace.json') for a in arms}
    if trace['GAP'] != trace['D2']:
        raise ValueError('Training input/order/LR hashes differ')
    for a in arms:
        if len(trace[a]) != expected_steps or s[a]['last_step'] != expected_steps:
            raise ValueError('Incomplete budget; expected %d steps' % expected_steps)
        if [x['step'] for x in trace[a]] != list(range(1, expected_steps+1)):
            raise ValueError('Noncontiguous training trace')
    files = {a: {f.name for f in (root/a).glob('eval_step_*.npz')} for a in arms}
    if files['GAP'] != files['D2'] or 'eval_step_000000.npz' not in files['GAP']:
        raise ValueError('Evaluation steps differ')
    n = p['GAP']['config']['episode_val_num']
    cache = {}
    ref_inputs = ref_labels = None
    for a in arms:
        for name in sorted(files[a]):
            with np.load(root/a/name, allow_pickle=False) as z:
                d = {k: z[k].copy() for k in z.files}
            for k in ('gap','d2'):
                if d[k].shape != (n,) or not np.isfinite(d[k]).all():
                    raise ValueError('Invalid per-episode accuracies')
                pred = d['pred_'+k]
                if pred.shape != (n,75) or d['labels'].shape != (n,75) or ((pred < 0) | (pred >= 5)).any():
                    raise ValueError('Invalid query predictions')
                computed = 100 * (pred == d['labels']).mean(axis=1)
                if not np.allclose(computed, d[k], atol=1e-5, rtol=0):
                    raise ValueError('Saved accuracy disagrees with query predictions')
            if d['input_sha256'].shape != (n,):
                raise ValueError('Missing validation input hashes')
            if ref_inputs is None:
                ref_inputs, ref_labels = d['input_sha256'], d['labels']
            elif not np.array_equal(ref_inputs, d['input_sha256']) or not np.array_equal(ref_labels, d['labels']):
                raise ValueError('Actual validation inputs/labels changed across checkpoints or arms')
            cache[a, int(name[10:-4])] = d
    for k in ('gap','d2','pred_gap','pred_d2'):
        if not np.array_equal(cache['GAP',0][k], cache['D2',0][k]):
            raise ValueError('Step-zero results differ: ' + k)
    p0 = cache['GAP',0]['gap']

    def row(g, d):
        return {'accuracy': {'GAP_gap': interval(g['gap']), 'GAP_d2': interval(g['d2']),
                             'D2_gap': interval(d['gap']), 'D2_d2': interval(d['d2'])},
                'paired_deltas': {'D2_d2_minus_GAP_gap': interval(d['d2']-g['gap']),
                                  'D2_d2_minus_P0_gap': interval(d['d2']-p0),
                                  'GAP_gap_minus_P0_gap': interval(g['gap']-p0),
                                  'D2_d2_minus_D2_gap': interval(d['d2']-d['gap']),
                                  'D2_gap_minus_GAP_gap': interval(d['gap']-g['gap'])}}

    print('PAIRING PASS: same P0, source, initial weights, training inputs/LR and actual validation inputs.')
    rows = []
    for step in sorted(k[1] for k in cache if k[0] == 'GAP'):
        r = row(cache['GAP',step], cache['D2',step]); r['step'] = step; rows.append(r)
        delta = r['paired_deltas']['D2_d2_minus_GAP_gap']
        print('step=%d | GAP %.4f | D2 %.4f | D2 using GAP %.4f | D2-GAP %+.4f [%+.4f,%+.4f]' %
              (step, r['accuracy']['GAP_gap']['mean'], r['accuracy']['D2_d2']['mean'],
               r['accuracy']['D2_gap']['mean'],delta['mean'],delta['ci95_low'],delta['ci95_high']))
    for a in arms:
        metric = s[a]['selection_metric']
        expected_best = max(s[a]['history'], key=lambda x: x[metric]['mean'])['step']
        if expected_best != s[a]['best_step']:
            raise ValueError('Selected checkpoint inconsistent with history')
    selected_steps = {a: s[a]['best_step'] for a in arms}
    selected = row(cache['GAP',selected_steps['GAP']], cache['D2',selected_steps['D2']])
    print('VALIDATION-SELECTED steps:', selected_steps)
    for k,v in selected['paired_deltas'].items():
        print('%s: %+.4f [%+.4f,%+.4f]' % (k,v['mean'],v['ci95_low'],v['ci95_high']))
    result = {'split':'val', 'pairing_verified': True, 'steps':expected_steps,
              'selected_steps':selected_steps, 'fixed_steps':rows, 'selected':selected,
              'note':'Exploratory task-level 95% t intervals; selected on these same tasks; no selection/multiplicity correction or training-seed uncertainty. Not test evidence.'}
    (root/'paired_D2_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', root/'paired_D2_summary.json')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--expected_steps', type=int, default=4000)
    a = p.parse_args()
    compare(Path(a.root), a.expected_steps)

#!/usr/bin/env python3
"""Paired validation comparison: original NDPNet+full loss vs independent GAP+CE, both SGD."""
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
    arms = ('B', 'A')
    required = [root/a/f for a in arms for f in ('provenance.json','summary.json','train_trace.json')]
    if any(not f.is_file() for f in required):
        print('PENDING: one arm has not yet saved its initial evaluation/trace. Let the pair runner continue.')
        return {'status':'pending'}
    p = {a: read(root/a/'provenance.json') for a in arms}
    s = {a: read(root/a/'summary.json') for a in arms}
    for a in arms:
        if p[a]['arm'] != a or s[a]['arm'] != a or s[a]['split'] != 'val' or p[a]['test_split_used']:
            raise ValueError('Wrong arm/split')
        cfg = p[a]['config']
        if cfg['experiment'] != 'R12_SGD_AB_v1' or cfg['optimizer'] != 'sgd' or cfg['adam'] or cfg['nesterov']:
            raise ValueError('Not the SGD experiment')
        for k in ('uniform_mask','no_triplet','no_div','no_mask'):
            if cfg[k] != (a == 'B'):
                raise ValueError('Wrong original NDPNet / GAP ablation flags')
        if not all(cfg[k] for k in ('no_bgloss','no_bg_sep','no_qbg')):
            raise ValueError('Keep background terms disabled')
        if cfg['epochs'] != 30 or cfg['episode_train_num'] != 4000 or cfg.get('freeze_bn_stats', False):
            raise ValueError('Changed schedule/BN protocol')
        if s[a]['selection_metric'] != ('gap' if a == 'B' else 'mask'):
            raise ValueError('Wrong selection metric')
    for k in ('pretrained_sha256','source_sha256','initial_model_sha256','val_manifest_sha256','val_transform','torch_version','csv_sha256'):
        if p['B'][k] != p['A'][k]:
            raise ValueError('Unpaired provenance: ' + k)
    if set(p['B']['config']) != set(p['A']['config']):
        raise ValueError('Config keys differ')
    for k in p['B']['config']:
        if k not in ('arm','outf','uniform_mask','no_triplet','no_div','no_mask') and p['B']['config'][k] != p['A']['config'][k]:
            raise ValueError('Config difference: ' + k)
    trace = {a: read(root/a/'train_trace.json') for a in arms}
    lengths = {a: len(trace[a]) for a in arms}
    common = min(lengths.values())
    # Compare the shared prefix before deciding that different progress is a mismatch.
    if trace['B'][:common] != trace['A'][:common]:
        raise ValueError('Training input/order/LR hashes differ WITHIN the shared completed prefix')
    for a in arms:
        if [x['step'] for x in trace[a]] != list(range(1, lengths[a]+1)):
            raise ValueError('Noncontiguous training trace: ' + a)
        if lengths[a] > expected_steps or s[a]['last_step'] > expected_steps:
            raise ValueError('Results exceed expected budget; use the correct expected_steps')
    if any(lengths[a] != expected_steps or s[a]['last_step'] != expected_steps for a in arms):
        print('PENDING: B trace=%d, summary=%d; A trace=%d, summary=%d; target=%d per arm.' %
              (lengths['B'], s['B']['last_step'], lengths['A'], s['A']['last_step'], expected_steps))
        print('Shared prefix verified: %d training inputs/LR entries match.' % common)
        print('No final comparison written. Wait for both runs to finish, then rerun this command.')
        return {'status':'pending', 'common_steps_verified':common, 'trace_lengths':lengths}
    files = {a: {f.name for f in (root/a).glob('eval_step_*.npz')} for a in arms}
    if files['B'] != files['A'] or 'eval_step_000000.npz' not in files['B']:
        raise ValueError('Evaluation steps differ')
    n = p['B']['config']['episode_val_num']
    cache = {}
    ref_inputs = ref_labels = None
    for a in arms:
        for name in sorted(files[a]):
            with np.load(root/a/name, allow_pickle=False) as z:
                d = {k: z[k].copy() for k in z.files}
            for k in ('gap','mask'):
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
    for k in ('gap','mask','pred_gap','pred_mask'):
        if not np.array_equal(cache['B',0][k], cache['A',0][k]):
            raise ValueError('Step-zero results differ: ' + k)
    p0 = cache['B',0]['gap']

    def row(g, d):
        return {'accuracy': {'B_gap': interval(g['gap']), 'B_mask': interval(g['mask']),
                             'A_gap': interval(d['gap']), 'A_mask': interval(d['mask'])},
                'paired_deltas': {'A_mask_minus_B_gap': interval(d['mask']-g['gap']),
                                  'A_mask_minus_P0_gap': interval(d['mask']-p0),
                                  'B_gap_minus_P0_gap': interval(g['gap']-p0),
                                  'A_mask_minus_A_gap': interval(d['mask']-d['gap']),
                                  'A_gap_minus_B_gap': interval(d['gap']-g['gap'])}}

    print('PAIRING PASS: same P0, source, initial weights, training inputs/LR and actual validation inputs.')
    rows = []
    for step in sorted(k[1] for k in cache if k[0] == 'B'):
        r = row(cache['B',step], cache['A',step]); r['step'] = step; rows.append(r)
        delta = r['paired_deltas']['A_mask_minus_B_gap']
        print('step=%d | B GAP %.4f | A MASK %.4f | A GAP %.4f | A-B %+.4f [%+.4f,%+.4f]' %
              (step, r['accuracy']['B_gap']['mean'], r['accuracy']['A_mask']['mean'],
               r['accuracy']['A_gap']['mean'],delta['mean'],delta['ci95_low'],delta['ci95_high']))
    for a in arms:
        metric = s[a]['selection_metric']
        expected_best = max(s[a]['history'], key=lambda x: x[metric]['mean'])['step']
        if expected_best != s[a]['best_step']:
            raise ValueError('Selected checkpoint inconsistent with history')
    selected_steps = {a: s[a]['best_step'] for a in arms}
    selected = row(cache['B',selected_steps['B']], cache['A',selected_steps['A']])
    print('VALIDATION-SELECTED steps:', selected_steps)
    for k,v in selected['paired_deltas'].items():
        print('%s: %+.4f [%+.4f,%+.4f]' % (k,v['mean'],v['ci95_low'],v['ci95_high']))
    result = {'split':'val', 'pairing_verified': True, 'steps':expected_steps,
              'selected_steps':selected_steps, 'fixed_steps':rows, 'selected':selected,
              'note':'Exploratory task-level 95% t intervals; selected on these same tasks; no selection/multiplicity correction or training-seed uncertainty. Not test evidence.'}
    (root/'paired_SGD_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', root/'paired_SGD_summary.json')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--expected_steps', type=int, default=4000)
    a = p.parse_args()
    compare(Path(a.root), a.expected_steps)

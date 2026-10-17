#!/usr/bin/env python3
"""Compare ordinary-BN, lr=1e-4 A/B/C on recorded identical validation tasks.

Only the two reviewed trainer hashes are accepted. All other source hashes,
initial weights, data transforms, tasks, inputs and LR traces must match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats

OLD_TRAINER_HASH = "1e876bb6bdf640b106376c81c8b6231d2c64d26c88c96df01db7a6691439e347"
ABC_TRAINER_HASH = "2af0a7706e9d954cecc0448f1b17d95dc39d791cc25289ab1bce9fa14d6cee2c"

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def interval(values):
    x = np.asarray(values, dtype=float)
    half = stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=float(x.mean()), ci95_low=float(x.mean()-half), ci95_high=float(x.mean()+half))

def compare(dirs):
    p = {a: read(d/'provenance.json') for a, d in dirs.items()}
    summaries = {a: read(d/'summary.json') for a, d in dirs.items()}
    traces = {a: read(d/'train_trace.json') for a, d in dirs.items()}
    expected = {
        'A': dict(uniform_mask=False, no_triplet=False, no_div=False, no_mask=False),
        'B': dict(uniform_mask=True, no_triplet=True, no_div=True, no_mask=True),
        'C': dict(uniform_mask=False, no_triplet=True, no_div=True, no_mask=True),
    }
    allowed = {'arm', 'outf', 'uniform_mask', 'no_triplet', 'no_div', 'no_mask'}
    for arm, d in dirs.items():
        v, cfg = p[arm], p[arm]['config']
        require(read(d/'config.json') == cfg, arm + ': config/provenance mismatch')
        require(v['arm'] == arm and cfg['arm'] == arm and summaries[arm]['arm'] == arm, 'Wrong arm directory: ' + arm)
        require(not cfg.get('freeze_bn_stats', False), arm + ': frozen-BN results are not allowed')
        require(cfg['lr'] == .0001 and cfg['weight_decay'] == .0005 and cfg['wd_scalars'] == -1, 'Unexpected optimizer configuration')
        require(cfg['adam'] and cfg['beta1'] == .5 and cfg['epochs'] == 30 and cfg['episode_train_num'] == 4000, 'Unexpected optimizer/schedule')
        require(cfg['data_name'] == 'miniImageNet' and cfg['way_num'] == 5 and cfg['shot_num'] == 1, 'Wrong experiment')
        require(all(cfg[k] == val for k, val in expected[arm].items()), 'Wrong ablation flags: ' + arm)
        require(all(cfg[k] for k in ['no_bg_sep', 'no_qbg', 'no_bgloss']), 'Background branches must be disabled')
        require(summaries[arm]['split'] == 'val' and v['test_split_used'] is False, 'Expected validation-only results')
        require(summaries[arm]['selection_metric'] == ('gap' if arm == 'B' else 'mask'), 'Wrong model selection metric')
        require(v['source_sha256']['shortcheck_trainer'] in {OLD_TRAINER_HASH, ABC_TRAINER_HASH}, 'Unreviewed trainer hash: ' + arm)
        if arm == 'C':
            require(v['source_sha256']['shortcheck_trainer'] == ABC_TRAINER_HASH, 'C must use the supplied ABC trainer')
        require(hashlib.sha256((d/'val_episodes.json').read_bytes()).hexdigest() == v['val_manifest_sha256'], 'Manifest hash mismatch: ' + arm)
        for key in ['pretrained_sha256', 'initial_model_sha256', 'val_manifest_sha256', 'val_transform', 'torch_version']:
            require(v[key] == p['A'][key], 'Unpaired ' + key + ': ' + arm)
        source = {k: val for k, val in v['source_sha256'].items() if k != 'shortcheck_trainer'}
        source_a = {k: val for k, val in p['A']['source_sha256'].items() if k != 'shortcheck_trainer'}
        require(source == source_a, 'Model/data source mismatch: ' + arm)
        require(set(cfg) == set(p['A']['config']), 'Different config keys: ' + arm)
        require(all(cfg[k] == p['A']['config'][k] for k in cfg if k not in allowed), 'Unexpected config differences: ' + arm)
        require(traces[arm] == traces['A'], 'Unpaired augmented inputs or LR schedule: ' + arm)
        require(summaries[arm]['last_step'] == 1000 and len(traces[arm]) == 1000, 'Expected exactly 1000 updates')
        require([r['step'] for r in traces[arm]] == list(range(1,1001)), 'Noncontiguous updates')
        require(all(r['lr'] == .0001 for r in traces[arm]), 'Unexpected LR trace')
    print('PAIRING PASS: same P0, model/data code, tasks, augmented inputs, LR and 1000 updates.')
    print('Trainer hash difference is limited to the two explicitly reviewed versions.')
    data, rows = {}, []
    n = p['A']['config']['episode_val_num']
    require(n == 600, 'Expected 600 validation tasks')
    for step in [0,100,500,1000]:
        data[step] = {}
        for arm,d in dirs.items():
            with np.load(d / ('eval_step_%06d.npz' % step), allow_pickle=False) as z:
                for mode in ['gap','mask']:
                    x=z[mode].copy()
                    require(x.shape==(n,) and np.isfinite(x).all() and ((x>=0)&(x<=100)).all(), 'Invalid score array')
                    data[step][arm+'_'+mode]=x
        v=data[step]
        deltas={
            'C_mask_minus_B_gap': v['C_mask']-v['B_gap'],
            'A_mask_minus_C_mask': v['A_mask']-v['C_mask'],
            'A_mask_minus_B_gap': v['A_mask']-v['B_gap'],
            'C_gap_minus_B_gap': v['C_gap']-v['B_gap'],
            'C_mask_minus_C_gap': v['C_mask']-v['C_gap'],
            'C_mask_minus_C_P0_gap': v['C_mask']-data[0]['C_gap'],
            'C_mask_minus_C_P0_mask': v['C_mask']-data[0]['C_mask'],
        }
        row=dict(step=step, accuracy={k:interval(x) for k,x in v.items()}, paired_deltas={k:interval(x) for k,x in deltas.items()})
        rows.append(row)
        print('step=%d | A mask %.2f | B GAP %.2f | C mask %.2f | C GAP %.2f' % (step,v['A_mask'].mean(),v['B_gap'].mean(),v['C_mask'].mean(),v['C_gap'].mean()))
        for name in ['C_mask_minus_B_gap','A_mask_minus_C_mask','C_mask_minus_C_P0_gap']:
            d=row['paired_deltas'][name]
            print('  %s: %+.3f [%+.3f, %+.3f]' % (name,d['mean'],d['ci95_low'],d['ci95_high']))
    result=dict(pairing_verified=True, comparisons=rows,
                note='Exploratory paired task t intervals, conditional on these runs; no training-seed uncertainty or multiple-comparison adjustment. Validation only.')
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-dir', default='./results/R12_lr1e4_short0906')
    parser.add_argument('--b-dir', default='./results/R12_lr1e4_B_short0906')
    parser.add_argument('--c-dir', default='./results/R12_lr1e4_C_ce0908')
    args=parser.parse_args()
    dirs={a:Path(getattr(args,a.lower()+'_dir')) for a in ['A','B','C']}
    try:
        result=compare(dirs)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit('COMPARISON STOPPED: '+str(exc))
    out=dirs['C']/'paired_ABC_summary.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('Saved:',out)

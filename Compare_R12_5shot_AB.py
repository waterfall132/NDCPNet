#!/usr/bin/env python3
"""Check paired 4000-step 5-shot runs and summarize VALIDATION only."""
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
import Evaluate_R12_5shot_AB as ev


def main(a):
    ev.np=np;ev.stats=stats
    dirs={'A':Path(a.a_dir),'B':Path(a.b_dir)}
    prov,sums=ev.audit_folders(dirs)
    paths={arm:dirs[arm]/f'eval_step_{sums[arm]["best_step"]:06d}.npz' for arm in dirs}
    aa=np.load(paths['A']);bb=np.load(paths['B'])
    before=np.load(dirs['B']/'eval_step_000000.npz')['gap']
    chosen={'A_mask':aa['mask'],'A_gap':aa['gap'],'B_gap':bb['gap'],'B_mask':bb['mask'],'P0_gap':before}
    result={'split':'val','selected_steps':{arm:sums[arm]['best_step'] for arm in dirs},
            'accuracy':{k:ev.interval(v) for k,v in chosen.items()},
            'A_mask_minus_B_gap':ev.interval(chosen['A_mask']-chosen['B_gap']),
            'A_mask_minus_P0_gap':ev.interval(chosen['A_mask']-before),
            'B_gap_minus_P0_gap':ev.interval(chosen['B_gap']-before),
            'note':'Validation-selected checkpoints evaluated on selection tasks: exploratory, not test confirmation.'}
    print('PAIRING PASS: same P0, model/data code, training inputs, LR and validation tasks.')
    for k,v in result['accuracy'].items():
        print(f'{k:10} {v["mean"]:.4f} [{v["ci95_low"]:.4f},{v["ci95_high"]:.4f}]')
    for k in ['A_mask_minus_B_gap','A_mask_minus_P0_gap','B_gap_minus_P0_gap']:
        v=result[k];print(k,f'{v["mean"]:+.4f} [{v["ci95_low"]:+.4f},{v["ci95_high"]:+.4f}]')
    ev.write(dirs['A'].parent/'R12_mini5_paired_val0909.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a-dir',default='./results/R12_mini5_A0909')
    p.add_argument('--b-dir',default='./results/R12_mini5_B0909')
    main(p.parse_args())

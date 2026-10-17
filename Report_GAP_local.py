#!/usr/bin/env python3
"""Print validation results without importing PyTorch."""
import argparse
import json
from pathlib import Path


def main(root):
    for seed in (0,1):
        p=Path(root)/f'seed{seed}'/'summary.json'
        if not p.is_file():
            print(f'seed={seed}: not complete'); continue
        s=json.loads(p.read_text())
        print(f'\nseed={seed}, validation tasks={s["tasks"]}')
        for k,v in s['results'].items():
            d=v['minus_D0']
            print(f'{k:12} acc={v["accuracy"]["mean"]:.4f} delta={d["mean"]:+.4f} [{d["ci95_low"]:+.4f},{d["ci95_high"]:+.4f}]')
        print('Selected/locked:',s.get('selected',s.get('locked_selected')))
        for k,v in s.get('controlled_contrasts',{}).items():
            print(k, f'{v["mean"]:+.4f} [{v["ci95_low"]:+.4f},{v["ci95_high"]:+.4f}]')
    print('\nValidation-only exploratory evidence. Do not interpret as independent test improvement.')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='./results/GAP_local0909')
    main(p.parse_args().root)

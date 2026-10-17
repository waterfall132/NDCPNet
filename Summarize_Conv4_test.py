#!/usr/bin/env python3
"""Compute a pooled task CI; do not average the legacy tester's round CI widths."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import t
p=argparse.ArgumentParser()
p.add_argument('--directory',required=True)
p.add_argument('--mode',choices=['mask','uniform'],required=True)
a=p.parse_args()
f=Path(a.directory)/f'episode_accuracy_seed0_{a.mode}.npz'
with np.load(f,allow_pickle=False) as z:
 x=z['accuracy'].astype(float)
 if int(z['seed'])!=0 or str(z['eval_mask'])!=a.mode: raise ValueError('Wrong seed/mode in exported file')
if x.shape!=(5,2000) or not np.isfinite(x).all() or (x<0).any() or (x>100).any():
 raise ValueError(f'Expected 5 x 2000 percentages, got shape {x.shape}')
y=x.reshape(-1); h=float(t.ppf(.975,len(y)-1)*y.std(ddof=1)/np.sqrt(len(y)))
r={'split':'test','mode':a.mode,'rounds':5,'episodes_per_round':2000,'total_episodes':len(y),'mean':float(y.mean()),'ci95_low':float(y.mean()-h),'ci95_high':float(y.mean()+h),'ci95_half_width':h,'round_means':x.mean(1).tolist(),'note':'Task-level t interval conditional on one checkpoint and this test split. Five episode seeds are not five training seeds. Historical tester seeds tasks consistently but does not export per-input hashes; this file does not certify cross-run input pairing.'}
out=Path(a.directory)/f'conv4_test_summary_{a.mode}.json'
out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'TEST {a.mode}: {r["mean"]:.4f} +/- {h:.4f}; n={len(y)}; saved {out}')

#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torchvision
from paired_common import (Episodes,loader,seed_all,make_model,evaluate,ci,save_json,
                           atomic,STAT_NAMES,source_hash,audit_data,state_hash)


def select_mode(model,pooling,T):
    model.eval()
    model.classifier.uniform_mask=pooling=='uniform'
    model.classifier.ndm.T=T


@torch.no_grad()
def timings(models,settings,ds,device,workers,repeats,warmups):
    result={name:[] for name in settings}
    raw=[]
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
    for q,s,_,idx in loader(ds,workers):
        q,s=q.to(device),s.to(device)
        if int(idx)==0:
            for name,(arm,pooling,T) in settings.items():
                m=models[arm];select_mode(m,pooling,T);m.classifier.record_diagnostics=False
                for _ in range(warmups):m(q,s)
            sync()
        local={name:[] for name in settings}
        names=list(settings)
        for rep in range(repeats):
            offset=(int(idx)+rep)%len(names)
            order=names[offset:]+names[:offset]
            for name in order:
                arm,pooling,T=settings[name];m=models[arm]
                select_mode(m,pooling,T);m.classifier.record_diagnostics=False
                sync();start=time.perf_counter();m(q,s);sync()
                elapsed=(time.perf_counter()-start)*1000
                local[name].append(elapsed);raw.append([int(idx),rep,name,elapsed])
        for name in settings:result[name].append(float(np.median(local[name])))
    return result,raw


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--root',required=True)
    p.add_argument('--pair',action='store_true',help='Also evaluate independently trained uniform arm (same losses)')
    p.add_argument('--episodes',type=int,default=2000);p.add_argument('--test-seed',type=int,default=924001)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--timing-tasks',type=int,default=100)
    p.add_argument('--timing-repeats',type=int,default=3);p.add_argument('--warmups',type=int,default=10)
    p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    if args.episodes<2 or min(args.timing_tasks,args.timing_repeats,args.warmups)<1:p.error('Invalid task counts')
    device=torch.device('cpu' if args.cpu else 'cuda');root=Path(args.root)
    out=root/f'eval_test{args.test_seed}_n{args.episodes}{"_pair" if args.pair else ""}'
    if out.exists():raise FileExistsError(f'{out} already exists; original outputs protected')
    arms=['ndm','uniform'] if args.pair else ['ndm']
    models={};reports={};selected={}
    for arm in arms:
        report=json.loads((root/arm/'training_summary.json').read_text());reports[arm]=report
        if not report['complete']:raise ValueError(f'{arm}: full declared budget not finished')
        if report['metadata']['source']!=source_hash():raise ValueError('Source changed since training')
        cfg=report['metadata']['config']
        if audit_data(args.data,cfg['shot'],True)!=report['metadata']['dataset']:raise ValueError('Dataset mismatch')
        ck=torch.load(root/arm/'best.pth.tar',map_location='cpu',weights_only=False)
        expected=max(report['history'],key=lambda x:x['accuracy']['mean'])
        if ck['metadata']!=report['metadata'] or ck['arm']!=arm or ck['step']!=expected['step']:
            raise ValueError(f'{arm}: checkpoint selection mismatch')
        seed_all(cfg['seed']);m=make_model(cfg,arm,device);m.load_state_dict(ck['model'],strict=True)
        models[arm]=m;selected[arm]=dict(epoch=expected['epoch'],val_accuracy=expected['accuracy'],
                 checkpoint_sha256=hashlib.sha256((root/arm/'best.pth.tar').read_bytes()).hexdigest())
        del ck
    if args.pair:
        a,b=reports['ndm'],reports['uniform']
        if a['metadata']!=b['metadata'] or a['epoch_hashes']!=b['epoch_hashes']:
            raise ValueError('Training pairing failed: config/P0/tasks/augmented inputs/LR differ')
        if [(v['val_inputs'],v['val_tasks']) for v in a['history']] != [(v['val_inputs'],v['val_tasks']) for v in b['history']]:
            raise ValueError('Validation pairing failed')
        print('PAIRING PASS: initialization, training tensors/LR and validation tensors',flush=True)
    settings={f'ndm_T{T}':('ndm','mask',T) for T in (1,2,3,5)}
    settings['ndm_uniform']=('ndm','uniform',3)
    if args.pair:
        settings.update(uniform_mask=('uniform','mask',3),uniform_uniform=('uniform','uniform',3))
    out.mkdir(parents=True)
    ds=Episodes(args.data,'test',cfg['shot'],args.episodes,args.test_seed)
    before={arm:state_hash(m.state_dict()) for arm,m in models.items()}
    results={}
    for name,(arm,pooling,T) in settings.items():
        seed_all(924);m=models[arm];select_mode(m,pooling,T)
        print(f'[EVAL] {name} {args.episodes} tasks; same selected checkpoint',flush=True)
        r=evaluate(m,ds,device,args.workers,diagnostics=True);results[name]=r
        atomic(out/(name+'.npz'),lambda f:np.savez_compressed(f,**r))
        print('[RESULT]',name,ci(r['accuracy']),flush=True)
    if len({r['input_hash'] for r in results.values()})!=1:raise ValueError('Test input mismatch')
    timing_ds=Episodes(args.data,'test',cfg['shot'],args.timing_tasks,args.test_seed+1)
    print('[TIMING] synchronized inference only; exclude I/O/transfers; use an idle GPU',flush=True)
    time_values,time_raw=timings(models,settings,timing_ds,device,args.workers,args.timing_repeats,args.warmups)
    assert all(state_hash(m.state_dict())==before[arm] for arm,m in models.items()), 'Weights/BN buffers changed during evaluation'
    counts={arm:dict(total_parameters=sum(v.numel() for v in m.parameters()),
              backbone_parameters=sum(v.numel() for v in m.features.parameters()),
              registered_head_parameters=sum(v.numel() for v in m.classifier.parameters()),
              active_foreground_temperature_parameters=1,
              note='Three registered head scalars; background branches disabled, only foreground temperature affects logits.')
            for arm,m in models.items()}
    summary=dict(dataset='StanfordCars',split='test',shot=cfg['shot'],training_seed=cfg['seed'],test_seed=args.test_seed,
              protocol=reports['ndm']['metadata'],selected_checkpoints=selected,
              actual_test_input_hash=results['ndm_T3']['input_hash'],task_hash=ds.task_hash,
              training_time={arm:r['timing'] for arm,r in reports.items()},parameters=counts,
              hardware=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
              evaluation_environment=dict(torch=torch.__version__,torchvision=torchvision.__version__,
                                          cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version()),
              timing_protocol=dict(tasks=args.timing_tasks,repeats=args.timing_repeats,warmups=args.warmups,
                definition='Synchronized wall ms per complete episode: backbone query+support, NDM including NNDSVD, TCSM, pooling, classifier. Excludes loading/H2D/diagnostics. Per-task median over repeats. Rotation of configuration order. FP32.'),
              ci_note='95% Student-t intervals across episode accuracies, conditional on these training runs. Not binomial intervals over pooled queries; not across-seed intervals.',
              settings={},paired_differences={})
    for name,r in results.items():
        spatial={}
        for group in ('q','s'):
            spatial[group]={}
            for j,k in enumerate(STAT_NAMES):
                a=r['stats_'+group][:,j];v=a[np.isfinite(a)]
                spatial[group][k]=dict(mean=float(v.mean()) if len(v) else None,valid_tasks=len(v))
        tt=np.asarray(time_values[name])
        summary['settings'][name]=dict(accuracy=ci(r['accuracy']),mask=spatial,
               timing_ms=dict(mean=float(tt.mean()),median=float(np.median(tt)),p95=float(np.percentile(tt,95))))
    pairs={f'T{T}_minus_T3':(f'ndm_T{T}','ndm_T3') for T in (1,2,5)}
    pairs['A_mask_minus_A_uniform']=('ndm_T3','ndm_uniform')
    if args.pair:
        pairs.update(A_mask_minus_B_uniform=('ndm_T3','uniform_uniform'),
                     B_mask_minus_B_uniform=('uniform_mask','uniform_uniform'))
    for name,(left,right) in pairs.items():summary['paired_differences'][name]=ci(results[left]['accuracy']-results[right]['accuracy'])
    save_json(out/'summary.json',summary);save_json(out/'test_tasks.json',ds.tasks)
    with open(out/'episode_accuracies.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['episode']+list(settings))
        for i in range(args.episodes):w.writerow([i]+[results[n]['accuracy'][i] for n in settings])
    with open(out/'timing_raw.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['episode','repeat','configuration','ms']);w.writerows(time_raw)
    lines=['% Cars/ResNet-12 ONLY. Do not label as mini-ImageNet/Conv-4 or CUB.',
           r'\begin{tabular}{ccc}',r'\toprule',r'$T$ & Accuracy (\%) & Time (ms/task) \\',r'\midrule']
    for T in (1,2,3,5):
        rr=summary['settings'][f'ndm_T{T}'];a=rr['accuracy'];ms=rr['timing_ms']['mean']
        lines.append(f"{T} & ${a['mean']:.2f} \\pm {a['ci95_half_width']:.2f}$ & {ms:.2f} \\\\")
    lines += [r'\bottomrule',r'\end{tabular}']
    (out/'iterations_Cars.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['paired_differences'],indent=2));print('SAVED',out,flush=True)


if __name__=='__main__':main()

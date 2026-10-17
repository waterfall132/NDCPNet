#!/usr/bin/env python3
"""Evaluate fixed validation-selected A/B checkpoints and P0 on one test manifest.
No optimizer, parameter updates, checkpoint selection or model modifications.
Requires the existing Train_NDCN_abcheck_abc.py and unchanged model/data modules.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

REVIEWED_TRAINER_SHA = '2af0a7706e9d954cecc0448f1b17d95dc39d791cc25289ab1bce9fa14d6cee2c'


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a-dir', default='./results/R12_lr3e5_A_short')
    p.add_argument('--b-dir', default='./results/R12_lr3e5_B_short')
    p.add_argument('--pretrained', default='./pretrain/mini_r12/backbone_best.pth.tar')
    p.add_argument('--dataset-dir', default='/root/miniImageNet/For_FewShot')
    p.add_argument('--outdir', default='./results/R12_selected_AB_test')
    p.add_argument('--episodes', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    return p.parse_args()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def audit_folders(dirs):
    prov={a:read(d/'provenance.json') for a,d in dirs.items()}
    sums={a:read(d/'summary.json') for a,d in dirs.items()}
    traces={a:read(d/'train_trace.json') for a,d in dirs.items()}
    allowed={'arm','outf','uniform_mask','no_triplet','no_div','no_mask'}
    for arm,d in dirs.items():
        p,s=prov[arm],sums[arm];cfg=p['config']
        require(read(d/'config.json')==cfg, 'config/provenance mismatch: '+arm)
        require(p['arm']==arm and s['arm']==arm and cfg['arm']==arm, 'Wrong arm directory')
        require(s['last_step']==4000 and len(traces[arm])==4000, 'Expected completed 4000-step runs')
        require([r['step'] for r in traces[arm]]==list(range(1,4001)), 'Noncontiguous training trace')
        require(all(r['lr']==3e-5 for r in traces[arm]), 'Unexpected training LR trace')
        require(s['split']=='val' and not p['test_split_used'], 'Selection must use validation')
        metric='mask' if arm=='A' else 'gap'
        require(s['selection_metric']==metric, 'Wrong selection metric')
        best=max(s['history'],key=lambda r:r[metric]['mean'])
        require(s['best_step']==best['step'] and abs(s['best_val']-best[metric]['mean'])<1e-8, 'Best selection inconsistent with history')
        require(cfg['data_name']=='miniImageNet' and cfg['encoder_model']=='ResNet12_ReLU' and cfg['way_num']==5 and cfg['shot_num']==1 and cfg['query_num']==15, 'Wrong experiment')
        require(cfg['lr']==3e-5 and not cfg.get('freeze_bn_stats',False), 'Expected ordinary BN and lr=3e-5')
        require(all(cfg[k] for k in ['no_bg_sep','no_qbg','no_bgloss']), 'Background branches must be disabled')
        require(all(cfg[k]==(arm=='B') for k in ['uniform_mask','no_triplet','no_div','no_mask']), 'Wrong A/B ablation flags')
        require(p['source_sha256']['shortcheck_trainer']==REVIEWED_TRAINER_SHA, 'Unreviewed trainer source')
        require(sha(d/'val_episodes.json')==p['val_manifest_sha256'], 'Validation manifest hash mismatch')
        require(set(cfg)==set(prov['A']['config']), 'Different config keys')
        require(all(cfg[k]==prov['A']['config'][k] for k in cfg if k not in allowed), 'Unexpected A/B configuration differences')
        for key in ['pretrained_sha256','initial_model_sha256','source_sha256','val_manifest_sha256','val_transform','torch_version']:
            require(p[key]==prov['A'][key], 'A/B mismatch: '+key)
        require(traces[arm]==traces['A'], 'Different training inputs or LR')
    return prov,sums


def interval(x):
    x=np.asarray(x,dtype=float)
    require(x.ndim==1 and len(x)>1 and np.isfinite(x).all(), 'Invalid statistical sample')
    mean=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(mean),ci95_low=float(mean-h),ci95_high=float(mean+h))


def summarize(scores, predictions, targets):
    contrasts={
        'A_mask_minus_B_gap':scores['A_mask']-scores['B_gap'],
        'A_mask_minus_P0_gap':scores['A_mask']-scores['P0_gap'],
        'B_gap_minus_P0_gap':scores['B_gap']-scores['P0_gap'],
        'A_mask_minus_A_gap':scores['A_mask']-scores['A_gap'],
        'A_gap_minus_B_gap':scores['A_gap']-scores['B_gap'],
        'B_mask_minus_B_gap':scores['B_mask']-scores['B_gap'],
    }
    effects={}
    for arm in ['A','B']:
        g=predictions[arm+'_gap']==targets;m=predictions[arm+'_mask']==targets
        corrected=((~g)&m).mean(1)*100;broken=(g&(~m)).mean(1)*100
        effects[arm]=dict(corrected_percentage_points=interval(corrected),
                          broken_percentage_points=interval(broken),
                          net_percentage_points=interval(corrected-broken))
    return dict(accuracy={k:interval(v) for k,v in scores.items()},
                paired_deltas={k:interval(v) for k,v in contrasts.items()},
                mask_prediction_changes=effects)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def main(a):
    require(a.episodes>=2, 'Need at least two episodes')
    require(a.device!='cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    dirs={'A':Path(a.a_dir),'B':Path(a.b_dir)}
    prov,sums=audit_folders(dirs)
    require(sha(trainer.__file__)==REVIEWED_TRAINER_SHA, 'ABC trainer file changed')
    require(str(torch.__version__)==prov['A']['torch_version'], 'Use the same PyTorch environment as training')
    require(sha(a.pretrained)==prov['A']['pretrained_sha256'], 'Wrong pretrained checkpoint')
    require((Path(a.dataset_dir)/'test.csv').is_file(), 'Missing test.csv')
    device=torch.device(a.device)
    torch.backends.cudnn.benchmark=True
    seed_all(a.seed)
    models={};selection={}
    for arm,d in dirs.items():
        path=d/'model_best.pth.tar'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        require(ck.get('shortcheck_format')==2, 'Unsupported checkpoint format')
        require(ck['config']==prov[arm]['config'] and ck['source_sha256']==prov[arm]['source_sha256'], 'Checkpoint configuration/source mismatch')
        require(ck['pretrained_sha256']==prov[arm]['pretrained_sha256'] and ck['initial_model_sha256']==prov[arm]['initial_model_sha256'], 'Checkpoint initialization mismatch')
        require(ck['global_step']==sums[arm]['best_step']==ck['best_step'], 'model_best does not match selected validation step')
        require(abs(ck['best_prec1']-sums[arm]['best_val'])<1e-8, 'Selected validation score mismatch')
        require(ck['val_manifest_sha256']==prov[arm]['val_manifest_sha256'], 'Checkpoint validation manifest mismatch')
        cfg=trainer.build_ndcn_cfg(argparse.Namespace(**ck['config']))
        require(cfg==ck['ndcn_cfg'], 'Saved head configuration inconsistent')
        model=FewShotNet.define_model(encoder_model='ResNet12_ReLU',classifier_model='NDCN',norm='batch',way_num=5,shot_num=1,init_type='normal',use_gpu=(a.device=='cuda'),ndcn_cfg=cfg)
        model.load_state_dict(ck['model'],strict=True);model.to(device).eval()
        models[arm]=model
        selection[arm]=dict(path=str(path.resolve()),sha256=sha(path),step=ck['global_step'],validation_accuracy=ck['best_prec1'])
        print('[selected] %s step=%d val=%.4f' % (arm,ck['global_step'],ck['best_prec1']),flush=True)
        del ck
    for name,digest in prov['A']['source_sha256'].items():
        if name=='shortcheck_trainer':continue
        mod=sys.modules.get(name)
        require(mod is not None and sha(mod.__file__)==digest, 'Model/data source mismatch: '+name)
    p0=copy.deepcopy(models['B'])
    pre=torch.load(a.pretrained,map_location='cpu',weights_only=False)
    require(pre.get('encoder_model')=='ResNet12_ReLU' and pre.get('data_name')=='miniImageNet', 'Pretraining metadata mismatch')
    p0.features.load_state_dict(pre['backbone'],strict=True);p0.eval();del pre
    opt=argparse.Namespace(**prov['A']['config'])
    opt.dataset_dir=a.dataset_dir;opt.episode_test_num=a.episodes;opt.testepisodeSize=1;opt.test_aug=False;opt.workers=0
    seed_all(a.seed)
    ds=FewShotDataloader.get_Fewshot_dataloader(opt,['test'])[0].dataset
    require(len(ds)==a.episodes, 'Unexpected test dataset length')
    require(str(ds.transform)==prov['A']['val_transform'], 'Test preprocessing differs from audited validation preprocessing')
    loader=torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=False)
    manifest=json.dumps(ds.data_list,ensure_ascii=False,sort_keys=True)
    (out/'test_episodes.json').write_text(manifest,encoding='utf-8')
    write(out/'provenance.json',dict(arguments=vars(a),selection=selection,
        pretrained_sha256=sha(a.pretrained),source_sha256=prov['A']['source_sha256'],
        evaluator_sha256=sha(__file__),test_csv_sha256=sha(Path(a.dataset_dir)/'test.csv'),
        episode_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
        transform=str(ds.transform),split='test',selection_split='val',training_updates=4000))
    keys=['A_gap','A_mask','B_gap','B_mask','P0_gap']
    predictions={k:[] for k in keys};scores={k:[] for k in keys};targets=[]
    def gap_logits(fq,fs):
        return F.normalize(fq.mean((2,3)),dim=1) @ F.normalize(fs.mean((2,3)),dim=1).T
    with torch.no_grad():
        for i,batch in enumerate(loader):
            q,y,s,_=batch
            q=torch.cat(q,0).to(device);s=torch.cat(s,0).reshape(-1,3,84,84).to(device);y=torch.cat(y,0).long().to(device)
            require(q.shape[0]==75 and s.shape[0]==5, 'Expected 75 queries and 5 supports')
            logits={}
            for arm,model in models.items():
                fq,fs=model.features(q),model.features(s)
                logits[arm+'_gap']=gap_logits(fq,fs)
                model.classifier.uniform_mask=False;model.classifier._raw_s=None
                logits[arm+'_mask']=model.classifier(fq,fs)
            logits['P0_gap']=gap_logits(p0.features(q),p0.features(s))
            for key,v in logits.items():
                require(bool(torch.isfinite(v).all()), 'Nonfinite logits: '+key)
                pred=v.argmax(1);predictions[key].append(pred.cpu().numpy())
                scores[key].append(100*(pred==y).float().mean().item())
            targets.append(y.cpu().numpy())
            if (i+1)%100==0:print('[test] %d/%d' % (i+1,len(loader)),flush=True)
    targets=np.stack(targets);predictions={k:np.stack(v) for k,v in predictions.items()};scores={k:np.asarray(v) for k,v in scores.items()}
    np.savez_compressed(out/'episode_scores.npz',**scores)
    np.savez_compressed(out/'query_predictions.npz',targets=targets,**predictions)
    result=summarize(scores,predictions,targets)
    result.update(split='test',episodes=a.episodes,selection=selection,
        note='Paired task t intervals conditional on these trained and validation-selected checkpoints. Not training-seed uncertainty. No multiple-comparison adjustment. Do not tune on these test results.')
    write(out/'summary.json',result)
    for k,v in result['accuracy'].items():print('%s %.4f [%.4f, %.4f]' % (k,v['mean'],v['ci95_low'],v['ci95_high']))
    for k,v in result['paired_deltas'].items():print('%s %+.4f [%+.4f, %+.4f]' % (k,v['mean'],v['ci95_low'],v['ci95_high']))
    print('[DONE] evaluation only; saved to',out.resolve(),flush=True)


if __name__=='__main__':
    args=arguments()
    import numpy as np
    from scipy import stats
    import torch
    import torch.nn.functional as F
    import Train_NDCN_abcheck_abc as trainer
    import models.network as FewShotNet
    import dataset.general_dataloader as FewShotDataloader
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES=True
    main(args)

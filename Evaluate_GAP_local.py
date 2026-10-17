#!/usr/bin/env python3
"""Validation-only frozen-backbone screen. No optimizer or test split option."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import time

TRAINER_SHA = '2af0a7706e9d954cecc0448f1b17d95dc39d791cc25289ab1bce9fa14d6cee2c'
WEIGHTS = (0.1, 0.25)
TEMPERATURE = 0.2


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', default='.')
    p.add_argument('--b-dir', required=True)
    p.add_argument('--dataset-dir', default='/root/miniImageNet/For_FewShot')
    p.add_argument('--outdir', required=True)
    p.add_argument('--stage', choices=['smoke','discover','confirm'], required=True)
    p.add_argument('--selection', help='selection.json from seed=0 discover, required for confirm')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    return p.parse_args()


def main(a):
    sys.path.insert(0, str(Path(a.project_root).resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import ImageFile
    from ndp_gap_local import GAPLocalEvidence, BRANCHES, mix_scores
    import ndp_gap_local
    import evidence_stats
    from evidence_stats import summarize, choose
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    require(a.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    random.seed(20260906); np.random.seed(20260906); torch.manual_seed(20260906)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260906)
    torch.backends.cudnn.benchmark = True  # matches the audited evaluator
    trainer = importlib.import_module('Train_NDCN_abcheck_abc')
    require(sha(trainer.__file__) == TRAINER_SHA, 'ABC trainer changed; do not bypass this check')
    network = importlib.import_module('models.network')
    dataloader = importlib.import_module('dataset.general_dataloader')
    folder = Path(a.b_dir)
    prov, history = read(folder/'provenance.json'), read(folder/'summary.json')
    cfg = read(folder/'config.json')
    require(cfg == prov['config'], 'config/provenance mismatch')
    require(cfg['arm'] == prov['arm'] == history['arm'] == 'B', 'Only independent B checkpoints accepted')
    require(history['split'] == 'val' and history['selection_metric'] == 'gap'
            and not prov['test_split_used'], 'Checkpoint must be selected on validation GAP')
    require(history['last_step'] == 4000, 'Expected completed 4000-update B run')
    best = max(history['history'], key=lambda x: x['gap']['mean'])
    require(history['best_step'] == best['step'] and abs(history['best_val']-best['gap']['mean']) < 1e-8,
            'Selection history inconsistent')
    require(cfg['data_name']=='miniImageNet' and cfg['encoder_model']=='ResNet12_ReLU'
            and cfg['way_num']==5 and cfg['shot_num']==1 and cfg['query_num']==15,
            'This runner is audited only for miniImageNet / ResNet12_ReLU / 5-way 1-shot')
    require(cfg['score_mode']=='free' and cfg['alpha_norm']=='supminmax'
            and cfg['frontend']=='ndm' and cfg['nonneg_mode']=='relu', 'Unexpected old head settings')
    require(all(cfg[k] for k in ('uniform_mask','no_triplet','no_div','no_mask',
                                 'no_bg_sep','no_qbg','no_bgloss')), 'Not the audited B configuration')
    require(cfg['lr']==3e-5 and not cfg.get('freeze_bn_stats',False), 'Expected lr=3e-5 ordinary-BN training')
    require(str(torch.__version__)==prov['torch_version'], 'Use the original training PyTorch environment')
    for name, digest in prov['source_sha256'].items():
        mod = trainer if name == 'shortcheck_trainer' else importlib.import_module(name)
        require(sha(mod.__file__)==digest, 'Model/data code changed: '+name)
    require(sha(folder/'val_episodes.json')==prov['val_manifest_sha256'], 'Original validation manifest changed')
    seed = int(cfg['seed'])
    require(a.stage=='smoke' or seed==({'discover':0,'confirm':1}[a.stage]), 'discover uses seed=0 B; confirm uses seed=1 B')
    require((a.stage=='confirm') == bool(a.selection), 'Only confirm requires --selection')
    path = folder/'model_best.pth.tar'
    ck = torch.load(path, map_location='cpu', weights_only=False)  # trusted own checkpoint only
    require(ck['shortcheck_format']==2 and ck['config']==cfg and ck['source_sha256']==prov['source_sha256'],
            'Checkpoint metadata mismatch')
    require(ck['global_step']==ck['best_step']==history['best_step'] and
            abs(ck['best_prec1']-history['best_val'])<1e-8, 'Not validation-selected model_best')
    for k in ['pretrained_sha256','initial_model_sha256','val_manifest_sha256']:
        require(ck[k]==prov[k], 'Checkpoint provenance mismatch: '+k)
    head_cfg = trainer.build_ndcn_cfg(argparse.Namespace(**cfg))
    require(head_cfg==ck['ndcn_cfg'], 'Head config mismatch')
    model = network.define_model(encoder_model='ResNet12_ReLU', classifier_model='NDCN',
            norm='batch', way_num=5, shot_num=1, init_type='normal',
            use_gpu=a.device=='cuda', ndcn_cfg=head_cfg)
    model.load_state_dict(ck['model'], strict=True)
    model.to(a.device).eval().requires_grad_(False)
    head = GAPLocalEvidence(model.classifier.ndm, temperature=TEMPERATURE,
                           eps=model.classifier.eps, pool_floor=model.classifier.pool_floor).eval()
    original_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    original_counters = trainer.counters(model)
    manifest = read(folder/'val_episodes.json')
    require(len(manifest)==600, 'Expected the existing 600 validation tasks')
    count = 20 if a.stage=='smoke' else 600
    opt = argparse.Namespace(**cfg)
    opt.dataset_dir=a.dataset_dir; opt.episode_val_num=600; opt.workers=0
    opt.test_aug=False; opt.testepisodeSize=1
    ds = dataloader.get_Fewshot_dataloader(opt, ['val'])[0].dataset
    require(str(ds.transform)==prov['val_transform'], 'Validation transform changed')
    # Use the actual original task list; do not rely on reseeding to reconstruct it.
    ds.data_list = manifest[:count]
    for task in ds.data_list:
        require(len(task)==5, 'Expected 5 classes per task')
        for c, row in enumerate(task):
            require(row['target']==c and len(row['support_set'])==1 and len(row['query_img'])==15,
                    'Invalid class-major task layout')
            for f in row['support_set']+row['query_img']:
                require(Path(f).resolve().is_relative_to((Path(a.dataset_dir)/'images').resolve()) and Path(f).is_file(),
                        'Manifest image missing or dataset path differs: '+str(f))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    protocol = dict(version=1, temperature=TEMPERATURE, weights=list(WEIGHTS), split='val',
        training_config={k:v for k,v in cfg.items() if k not in ('seed','outf')},
        core_sha256=sha(ndp_gap_local.__file__), evaluator_sha256=sha(__file__),
        statistics_sha256=sha(evidence_stats.__file__), source_sha256=prov['source_sha256'],
        pretrained_sha256=prov['pretrained_sha256'], val_manifest_sha256=prov['val_manifest_sha256'],
        val_csv_sha256=sha(Path(a.dataset_dir)/'val.csv'), transform=str(ds.transform))
    locked = None
    if a.stage=='confirm':
        locked = read(a.selection)
        require(locked['protocol']==protocol and locked['stage']=='discover' and locked['training_seed']==0,
                'Selection protocol differs; do not retune during confirm')
        recipes = {k:tuple(v) for k,v in locked['recipes'].items()}
        selected = locked['selected']
        require(selected in recipes, 'Invalid locked selection')
        recipes = {k:v for k,v in recipes.items() if k in ('D0',selected)}
    else:
        recipes = {'D0': ('D0',0.0)}
        for w in WEIGHTS:
            for name in BRANCHES:
                recipes[f'{name}_w{w:g}'] = (name,w)
    names = tuple(dict.fromkeys(v[0] for v in recipes.values() if v[0]!='D0'))
    out=Path(a.outdir); out.mkdir(parents=True, exist_ok=False)
    write(out/'episodes.json',ds.data_list)
    write(out/'provenance.json',dict(arguments=vars(a),protocol=protocol,training_seed=seed,
        selection_sha256=sha(a.selection) if a.selection else None,
        checkpoint=dict(path=str(path.resolve()),sha256=sha(path),step=ck['global_step'],
                        validation_accuracy=ck['best_prec1']), stage=a.stage,
        tasks=count, recipes=recipes, model_updated=False, test_split_used=False))
    expected = np.load(folder/f'eval_step_{ck["global_step"]:06d}.npz')['gap'][:count]
    del ck
    predictions={k:[] for k in recipes}; targets=[]; logits={k:[] for k in recipes}
    diag=[]; baseline_old=[]; start=time.monotonic()
    with torch.no_grad():
        for i,(qi,qt,si,st) in enumerate(loader):
            q=torch.cat(qi,0).to(a.device); s=torch.cat(si,0).reshape(-1,3,84,84).to(a.device)
            y=torch.cat(qt,0).long().to(a.device)
            require(q.shape[0]==75 and s.shape[0]==5 and
                    torch.equal(torch.cat(st,0).long(),torch.arange(5)), 'Invalid batch labels/shapes')
            fq,fs=model.features(q),model.features(s)
            scores, diagnostics=head.branches(fq,fs,names)
            original_gap=F.normalize(fq.mean((2,3)),dim=1) @ F.normalize(fs.mean((2,3)),dim=1).T
            require(torch.allclose(scores['D0'],original_gap,atol=1e-7,rtol=1e-6), 'D0 mismatch with original GAP')
            if i==0 and 'D1' in scores:
                model.classifier.uniform_mask=False; model.classifier._raw_s=None
                old=model.classifier(fq,fs)/model.classifier.gamma_fg+1
                model.classifier.uniform_mask=True; model.classifier._raw_s=None
                require(torch.allclose(old,scores['D1'],atol=2e-5,rtol=2e-5), 'D1 does not reproduce original mask cosine')
            for k,(name,w) in recipes.items():
                value=scores['D0'] if name=='D0' else mix_scores(scores['D0'],scores[name],w)
                predictions[k].append(value.argmax(-1).cpu().numpy())
                logits[k].append(value.cpu().numpy())
            baseline_old.append(100*(original_gap.argmax(-1)==y).float().mean().item())
            targets.append(y.cpu().numpy())
            diag.append({k:float(v.detach().cpu()) for k,v in diagnostics.items()})
            if (i+1)%20==0:
                print(f'[val {a.stage}] {i+1}/{count}; elapsed {time.monotonic()-start:.0f}s',flush=True)
    unchanged=all(torch.equal(original_state[k],v.detach().cpu()) for k,v in model.state_dict().items())
    trainer.restore_counters(model,original_counters)
    require(unchanged,'Unexpected model parameter/BN-buffer mutation')
    predictions={k:np.stack(v) for k,v in predictions.items()}; targets=np.stack(targets)
    np.savez_compressed(out/'query_predictions.npz',targets=targets,**predictions)
    np.savez_compressed(out/'cosine_scores.npz',**{k:np.stack(v) for k,v in logits.items()})
    write(out/'diagnostics.json',diag)
    # Audited trainer stores float32-derived percentages; allow only rounding noise.
    err=float(np.max(np.abs(np.asarray(baseline_old)-expected)))
    write(out/'baseline_check.json',dict(max_task_error_pp=err,passed=err<1e-4,
                                       weights_and_bn_unchanged=unchanged))
    require(err<1e-4, 'GAP task scores do not reproduce saved B validation; stop and inspect baseline_check.json')
    result,scores=summarize(predictions,targets)
    np.savez_compressed(out/'episode_scores.npz',**scores)
    result.update(stage=a.stage,training_seed=seed,split='val',tasks=count,
        note='Exploratory paired task t intervals, unadjusted for candidate selection. Same validation classes/tasks across seeds; not independent class generalization or seed CI.')
    if a.stage=='discover':
        selected=choose(result)
        write(out/'selection.json',dict(stage='discover',training_seed=seed,protocol=protocol,
            recipes=recipes,selected=selected,selection_rule='max mean validation accuracy; D0 wins exact ties',
            selected_result=result['results'][selected],checkpoint_sha256=sha(path)))
        result['selected']=selected
    elif a.stage=='confirm':
        result['locked_selected']=selected
        result['seed0_selected_result']=locked['selected_result']
    write(out/'summary.json',result)
    for k,v in result['results'].items():
        d=v['minus_D0']
        print(f'{k:12} {v["accuracy"]["mean"]:.4f} | vs D0 {d["mean"]:+.4f} [{d["ci95_low"]:+.4f},{d["ci95_high"]:+.4f}]')
    print('[DONE] frozen validation only; results:',out.resolve(),flush=True)


if __name__=='__main__':
    main(arguments())

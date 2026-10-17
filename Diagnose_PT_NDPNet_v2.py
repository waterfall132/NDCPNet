#!/usr/bin/env python3
"""Read-only paired validation of pretraining versus meta-training checkpoints.

Place beside models/, dataset/, utils.py. No optimization or BN updates.
P0=classification-pretrained backbone; P1=meta-trained backbone.
P1 is mask-trained: P1_gap is NOT a separately uniform-trained baseline.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def interval(values):
    a = np.asarray(values, dtype=float)
    h = float(t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {'mean': float(a.mean()), 'ci95_half_width': h,
            'ci95_low': float(a.mean()-h), 'ci95_high': float(a.mean()+h)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--data_name', default='miniImageNet')
    p.add_argument('--encoder_model', required=True,
                   choices=['ResNet12_ReLU', 'Conv64F_Local_ReLU'])
    p.add_argument('--pretrained', required=True)
    p.add_argument('--meta_checkpoint', required=True,
                   help='model_best.pth.tar or its directory')
    p.add_argument('--episodes', type=int, default=600)
    p.add_argument('--seed', type=int, default=20260906)
    p.add_argument('--outdir', required=True)
    p.add_argument('--bn_probe', action='store_true',
                   help='also try P1 with P0 BN running buffers; diagnostic hybrid only')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    a = p.parse_args()
    if a.episodes < 2:
        p.error('--episodes must be >= 2')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu for a smoke check')
    import models.network as net
    import dataset.general_dataloader as data

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dev = torch.device(a.device)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed output exists; choose a new --outdir')
    meta_path = Path(a.meta_checkpoint)
    if meta_path.is_dir():
        meta_path = meta_path / 'model_best.pth.tar'
    pre = torch.load(a.pretrained, map_location='cpu', weights_only=False)
    meta = torch.load(meta_path, map_location='cpu', weights_only=False)
    for name, ck in [('pretrained', pre), ('meta', meta)]:
        for key, expected in [('encoder_model', a.encoder_model),
                              ('data_name', a.data_name)]:
            if ck.get(key) not in (None, expected):
                raise ValueError('%s %s mismatch: %s' % (name, key, ck[key]))
    if meta.get('classifier_model') not in (None, 'NDCN'):
        raise ValueError('This diagnostic supports the NDCN pooling head only')
    for key, expected in [('way_num', 5), ('shot_num', 1)]:
        if meta.get(key) not in (None, expected):
            raise ValueError('This diagnostic is 5-way 1-shot; incompatible '+key)
    cfg = meta.get('ndcn_cfg')
    if not cfg:
        raise ValueError('No ndcn_cfg in checkpoint; cannot silently guess method settings')
    cfg = dict(cfg)
    if cfg.get('uniform_mask', False):
        raise ValueError('Expected a mask-trained meta checkpoint')
    if cfg.get('use_bg_sep', True) or cfg.get('use_qbg', True):
        raise ValueError('Expected no_bg_sep + no_qbg for pure cosine pooling comparison')
    model = net.define_model(encoder_model=a.encoder_model, classifier_model='NDCN',
                             way_num=5, shot_num=1, use_gpu=False, ndcn_cfg=cfg)
    sd = meta['model']
    if all(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    pre_features = copy.deepcopy(model.features)
    pre_features.load_state_dict(pre['backbone'], strict=True)
    pre_features.eval()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not module.track_running_stats:
            raise ValueError('BN without running statistics changes inference assumptions: '+name)

    # Record parameter/buffer drift; this is descriptive, not proof of damage.
    s0, s1 = pre_features.state_dict(), model.features.state_dict()
    drift = {}
    for k, v in s1.items():
        if v.is_floating_point():
            diff = (v.float() - s0[k].float()).norm().item()
            drift[k] = {'l2_change': diff,
                        'relative_l2_change': diff / max(s0[k].float().norm().item(), 1e-12)}
    backbones = {'P0': pre_features, 'P1': model.features}
    if a.bn_probe:
        hybrid = copy.deepcopy(model.features)
        hs = hybrid.state_dict()
        for k in hs:
            if k.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                hs[k] = s0[k].clone()
        hybrid.load_state_dict(hs, strict=True)
        backbones['P1_oldBN'] = hybrid.eval()
    model.to(dev).eval()
    for b in backbones.values():
        b.to(dev).eval()

    # Use the actual server dataloader and its validation transform. Never test.csv.
    opt = argparse.Namespace(dataset_dir=a.dataset_dir, data_name=a.data_name,
        mode='val', imageSize=84, train_aug=True, test_aug=False, workers=0,
        way_num=5, shot_num=1, query_num=15, aug_shot_num=20, episodeSize=1,
        testepisodeSize=1, episode_train_num=1, episode_val_num=a.episodes,
        episode_test_num=1, seed=a.seed, current_epoch=0)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    existing = data.get_Fewshot_dataloader(opt, ['val'])[0]
    ds = existing.dataset
    if not hasattr(ds, 'data_list'):
        raise AttributeError('Dataset lacks data_list; upload current dataloader for adaptation')
    manifest = json.dumps(ds.data_list, ensure_ascii=False, sort_keys=True)
    (out / 'episodes.json').write_text(manifest, encoding='utf-8')
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    module_paths = {name: m.__file__ for name, m in list(__import__('sys').modules.items())
                    if name.startswith(('models.', 'dataset.')) and getattr(m, '__file__', None)}
    provenance = {'arguments': vars(a), 'torch_version': str(torch.__version__),
        'meta_checkpoint_sha256': sha(meta_path), 'pretrained_sha256': sha(a.pretrained),
        'split': 'val', 'transform': str(getattr(ds, 'transform', None)),
        'ndcn_cfg': cfg, 'meta_epoch': meta.get('epoch_index'),
        'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'}
                             for g in meta.get('optimizer', {}).get('param_groups', [])],
        'episode_manifest_sha256': hashlib.sha256(manifest.encode()).hexdigest(),
        'source_sha256': {name: sha(path) for name, path in module_paths.items()},
        'backbone_drift': drift}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    scores = {key+'_'+pool: [] for key in backbones for pool in ('gap','mask')}
    diags = {key: [] for key in backbones}
    # Read-only instrumentation of the exact rescaling calls: the outputs
    # returned by the original function are passed back without modification.
    rescale_trace = []
    original_rescale = model.classifier._rescale_alpha

    def traced_rescale(raw):
        bounds = getattr(model.classifier, '_alpha_range', None)
        info = {'raw_min': raw.min().item(), 'raw_max': raw.max().item()}
        if bounds is not None:
            lo, hi = bounds
            info.update(support_lo=float(lo), support_hi=float(hi),
                        raw_below_support_fraction=(raw < lo).float().mean().item(),
                        raw_above_support_fraction=(raw > hi).float().mean().item(),
                        all_components_below_support_fraction=(raw < lo).all(1).float().mean().item())
        rescale_trace.append(info)
        return original_rescale(raw)

    model.classifier._rescale_alpha = traced_rescale
    print('Transform:', provenance['transform'], flush=True)
    print('Checkpoints loaded strictly. No optimizer steps. Validation only.', flush=True)
    with torch.inference_mode():
        for idx, (qi, qt, si, _) in enumerate(loader):
            q = torch.cat(qi, 0).to(dev)
            s = torch.cat(si, 0).reshape(-1, 3, 84, 84).to(dev)
            y = torch.cat(qt, 0).to(dev).long()
            if len(s) != 5 or len(q) != 75:
                raise ValueError('Expected 5 supports and 75 queries per episode')
            for name, b in backbones.items():
                fq, fs = b(q), b(s)
                if fq.min().item() < -1e-6 or fs.min().item() < -1e-6:
                    raise ValueError('Signed features: raw GAP and nonnegative head are not matched')
                gap = F.normalize(fq.mean((2,3)), dim=1) @ F.normalize(fs.mean((2,3)), dim=1).T
                head = model.classifier
                head.uniform_mask = True
                head._raw_s = None
                unif = head(fq, fs)
                if not torch.allclose(unif-unif[:, :1],
                    head.gamma_fg*(gap-gap[:, :1]), atol=3e-4, rtol=3e-4):
                    raise RuntimeError('uniform NDCN disagrees with independent GAP cosine scores')
                head.uniform_mask = False
                head._raw_s = None
                rescale_trace.clear()
                mask = head(fq, fs)
                if not torch.isfinite(mask).all():
                    raise FloatingPointError('Non-finite mask logits')
                for pool, logits in [('gap', gap), ('mask', mask)]:
                    scores[name+'_'+pool].append(100*(logits.argmax(1)==y).float().mean().item())
                row = {k: float(v) for k,v in getattr(head, '_diag', {}).items()}
                mq = getattr(head, '_last_Mq', None)
                aq = getattr(head, '_last_alpha_q', None)
                if mq is None:
                    raise RuntimeError('Use supplied 0906 ndcn.py with cached forward masks')
                zero = (mq.sum(1) == 0)
                row['zero_query_mask_fraction'] = zero.float().mean().item()
                row['constant_query_mask_fraction'] = ((mq.amax(1)-mq.amin(1)) < 1e-6).float().mean().item()
                row['mask_changes_gap_prediction_fraction'] = (mask.argmax(1) != gap.argmax(1)).float().mean().item()
                if aq is not None:
                    row['alpha_zero_fraction'] = (aq == 0).float().mean().item()
                    row['alpha_one_fraction'] = (aq == 1).float().mean().item()
                if len(rescale_trace) == 2:
                    row.update({'query_'+k: v for k,v in rescale_trace[-1].items()})
                diags[name].append(row)
            if (idx+1) % 50 == 0:
                print('Episodes %d/%d | %s' % (idx+1, len(loader),
                      ' | '.join('%s %.2f' % (k,np.mean(v)) for k,v in scores.items())), flush=True)
    pairs = [('P0_mask','P0_gap'), ('P1_mask','P1_gap'),
             ('P1_gap','P0_gap'), ('P1_mask','P0_mask'), ('P1_mask','P0_gap')]
    if a.bn_probe:
        pairs += [('P1_oldBN_gap','P1_gap'), ('P1_oldBN_mask','P1_mask')]
    result = {'accuracy': {k:interval(v) for k,v in scores.items()},
              'paired_deltas': {x+' minus '+y:interval(np.array(scores[x])-scores[y]) for x,y in pairs},
              'diagnostics_mean': {k: {d:float(np.mean([v[d] for v in rows]))
                                     for d in rows[0]} for k,rows in diags.items()},
              'note':'CIs condition on fixed checkpoints, not training-seed variability. '
                     'P1_oldBN is a diagnostic hybrid, never a reported trained model.'}
    np.savez(out / 'episode_scores.npz', **scores)
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bounded ResNet-12 audit; uses the project's unchanged model and data modules.

Epochs still mean 4000 updates and the cosine horizon remains 30 epochs.
Extra validation restores RNG, buffers, modes and Python DropBlock counters.
Resume only this script's model_last.pth.tar, including a partial epoch.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', default=str(Path(__file__).with_name('r12_abcheck_settings.json')))
    p.add_argument('--dataset_dir', default='/root/miniImageNet/For_FewShot')
    p.add_argument('--pretrained', default='./pretrain/mini_r12/backbone_best.pth.tar')
    p.add_argument('--outdir', required=True)
    p.add_argument('--arm', choices=['A', 'B', 'C'], required=True,
                   help='A: mask+full loss; B: uniform+CE; C: mask+CE (all auxiliary losses off)')
    p.add_argument('--resume', default='', help='only this ABC script\'s model_last.pth.tar')
    p.add_argument('--max_steps', type=int, default=1000, help='total updates, including resumed ones')
    p.add_argument('--eval_steps', default='100,500,1000')
    p.add_argument('--val_episodes', type=int, default=600)
    p.add_argument('--val_seed', type=int, default=20260906)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--adam', action='store_true', default=True)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--weight_decay', type=float, default=0.0005)
    p.add_argument('--wd_scalars', type=float, default=-1)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    return p.parse_args()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if s['cuda']:
        torch.cuda.set_rng_state_all(s['cuda'])


def counters(model):
    return {n: m.num_batches_tracked for n, m in model.named_modules()
            if isinstance(getattr(m, 'num_batches_tracked', None), int)}


def restore_counters(model, saved):
    mods = dict(model.named_modules())
    for n, v in saved.items():
        mods[n].num_batches_tracked = v


@contextmanager
def preserve_state(model=None):
    rng = rng_state()
    modes = [(m, m.training) for m in model.modules()] if model is not None else []
    ctrs = counters(model) if model is not None else {}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()} if model is not None else {}
    try:
        yield
    finally:
        if model is not None:
            with torch.no_grad():
                for n, b in model.named_buffers():
                    b.copy_(buffers[n])
            restore_counters(model, ctrs)
            for m, mode in modes:
                m.training = mode
        restore_rng(rng)


def batch_tensors(batch, device):
    qi, qt, si, _ = batch
    q = torch.cat(qi, 0).to(device)
    s = torch.cat(si, 0).reshape(-1, 3, 84, 84).to(device)
    y = torch.cat(qt, 0).to(device).long()
    if q.shape[0] != 75 or s.shape[0] != 5:
        raise ValueError('Expected exactly 75 queries and 5 supports for 5-way 1-shot')
    return q, s, y


def mean_ci(values):
    arr = np.asarray(values, dtype=float)
    half = float(stats.t.ppf(.975, len(arr)-1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return {'mean': float(arr.mean()), 'ci95_half_width': half}


def evaluate(model, loader, device):
    scores = {'gap': [], 'mask': []}
    head = model.classifier
    previous_uniform = head.uniform_mask
    try:
        with preserve_state(model), torch.no_grad():
            model.eval()
            head.uniform_mask = False
            for i, batch in enumerate(loader):
                q, s, y = batch_tensors(batch, device)
                fq, fs = model.features(q), model.features(s)
                gap = F.normalize(fq.mean((2, 3)), dim=1) @ F.normalize(fs.mean((2, 3)), dim=1).T
                head._raw_s = None
                mask = head(fq, fs)
                if not torch.isfinite(mask).all() or not torch.isfinite(gap).all():
                    raise FloatingPointError('Non-finite validation scores')
                for key, logits in [('gap', gap), ('mask', mask)]:
                    scores[key].append(100 * (logits.argmax(1) == y).float().mean().item())
                if (i + 1) % 100 == 0:
                    print('  [eval] %d/%d' % (i + 1, len(loader)), flush=True)
    finally:
        head.uniform_mask = previous_uniform
        head._raw_s = None
    result = {k: mean_ci(v) for k, v in scores.items()}
    result['mask_minus_gap'] = mean_ci(np.array(scores['mask']) - np.array(scores['gap']))
    return scores, result


def seed_stream(seed, index, phase):
    value = (seed + 1000003 * index + 1000033 * phase) % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tensor_tree_hash(tree):
    """Hash CPU input tensors or a model state dict, including shapes and keys."""
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            arr = x.detach().cpu().contiguous().numpy()
            h.update(str((arr.shape, str(arr.dtype))).encode())
            h.update(arr.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())
    visit(tree)
    return h.hexdigest()


def arm_settings(arm):
    if arm not in ('A', 'B', 'C'):
        raise ValueError('Unknown arm: ' + arm)
    return dict(uniform_mask=(arm == 'B'), no_triplet=(arm != 'A'),
                no_div=(arm != 'A'), no_mask=(arm != 'A'),
                no_bgloss=True, no_bg_sep=True, no_qbg=True)


def main(a):
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate the same environment used for your training.')
    if not 1 <= a.max_steps <= 120000 or a.val_episodes < 2:
        raise ValueError('max_steps must be 1..120000; val_episodes must be >=2')
    targets = {int(x) for x in a.eval_steps.split(',') if x.strip()} | {a.max_steps}
    settings = json.loads(Path(a.settings).read_text(encoding='utf-8'))
    settings.update(dataset_dir=a.dataset_dir, pretrained=a.pretrained, workers=0, arm=a.arm,
                    outf=str(Path(a.outdir).resolve()), crash_tol=0.0,
                    episode_val_num=a.val_episodes, seed=a.seed,
                    lr=a.lr, weight_decay=a.weight_decay, wd_scalars=a.wd_scalars)
    if settings.get('freeze_bn_stats', False):
        raise ValueError('This experiment requires original BN training behavior, not frozen BN')
    settings.update(arm_settings(a.arm))
    opt = argparse.Namespace(**settings)
    if opt.epochs != 30 or opt.episode_train_num != 4000:
        raise ValueError('Keep epochs=30 and episode_train_num=4000; use max_steps to stop early')
    for required in [Path(a.pretrained), Path(a.dataset_dir) / 'train.csv', Path(a.dataset_dir) / 'val.csv']:
        if not required.is_file():
            raise FileNotFoundError('Required input not found: ' + str(required))
    cfg = build_ndcn_cfg(opt)
    out = Path(a.outdir).resolve()
    if not a.resume:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError('Resume requires the original output directory')
    run_lock = open(out / '.run.lock', 'a+')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another shortcheck process is using this output directory')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(a.device)
    model = FewShotNet.define_model(encoder_model='ResNet12_ReLU', classifier_model='NDCN',
        norm='batch', way_num=5, shot_num=1, init_type='normal', use_gpu=(a.device == 'cuda'), ndcn_cfg=cfg)
    if a.wd_scalars < 0:
        groups = [{'params': model.parameters(), 'weight_decay': a.weight_decay}]
    else:
        scal, rest = [], []
        for n, p in model.named_parameters():
            (scal if n.endswith(('_raw_gamma_fg', '_raw_gamma_bg', 'beta_bg')) else rest).append(p)
        groups = [{'params': rest, 'weight_decay': a.weight_decay},
                  {'params': scal, 'weight_decay': a.wd_scalars}]
    optimizer = torch.optim.Adam(groups, lr=a.lr, betas=(.5, .9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=a.lr * .001)
    checkpoint = None
    history, step, best, best_step, plan = [], 0, -1., -1, None
    trace = []
    pre_hash = sha(a.pretrained)
    source_hashes = {n: sha(m.__file__) for n, m in list(sys.modules.items())
                     if n.startswith(('models.', 'dataset.')) and getattr(m, '__file__', None)}
    source_hashes['shortcheck_trainer'] = sha(__file__)
    if a.resume:
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=False)
        if checkpoint.get('shortcheck_format') != 2:
            raise ValueError('Resume only this script\'s model_last.pth.tar, never the previous 54% checkpoint')
        if checkpoint['config'] != settings or checkpoint['source_sha256'] != source_hashes:
            raise ValueError('Settings/source changed; refusing a mixed-protocol resume')
        if checkpoint['pretrained_sha256'] != pre_hash or checkpoint['val_seed'] != a.val_seed:
            raise ValueError('Pretrained file or validation seed changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        restore_counters(model, checkpoint['python_counters'])
        history, step = checkpoint['history'], checkpoint['global_step']
        trace = checkpoint['train_trace']
        initial_sha = checkpoint['initial_model_sha256']
        best, best_step, plan = checkpoint['best_prec1'], checkpoint['best_step'], checkpoint['train_plan']
        if a.max_steps <= step:
            raise ValueError('max_steps is the total update target and must exceed the saved global_step')
    else:
        pre = torch.load(a.pretrained, map_location='cpu', weights_only=False)
        for key, expected in [('encoder_model', 'ResNet12_ReLU'), ('data_name', 'miniImageNet')]:
            if pre.get(key) != expected:
                raise ValueError('Checkpoint metadata mismatch: ' + key)
        model.features.load_state_dict(pre['backbone'], strict=True)
        print('[init] PRETRAINED BACKBONE ONLY; strict key match passed', flush=True)
        print('[init] epoch=%s historical_val=%s sha256=%s' %
              (pre.get('epoch'), pre.get('episodic_val'), pre_hash), flush=True)
        del pre
        initial_sha = tensor_tree_hash(model.state_dict())
    # Fixed validation tasks are built outside the training RNG stream.
    with preserve_state():
        random.seed(a.val_seed)
        np.random.seed(a.val_seed)
        torch.manual_seed(a.val_seed)
        ds = FewShotDataloader.get_Fewshot_dataloader(opt, ['val'])[0].dataset
        val_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    manifest = json.dumps(ds.data_list, ensure_ascii=False, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    if checkpoint and checkpoint['val_manifest_sha256'] != manifest_hash:
        raise ValueError('Validation task list changed')
    (out / 'val_episodes.json').write_text(manifest, encoding='utf-8')
    write_json(out / 'config.json', settings)
    write_json(out / 'provenance.json', {'config': settings, 'arguments': vars(a),
        'pretrained_sha256': pre_hash, 'source_sha256': source_hashes,
        'val_manifest_sha256': manifest_hash, 'val_transform': str(ds.transform),
        'torch_version': str(torch.__version__), 'test_split_used': False, 'arm': a.arm, 'initial_model_sha256': initial_sha,
        'notes': 'Paired A/B uses update-index seeds and verifies input hashes; not a replay of historic runs.'})
    if checkpoint:
        restore_rng(checkpoint['rng'])
        del checkpoint
    print('[protocol] arm=%s; pooling=%s; loss=%s; BN=original/train; all backbone parameters trainable' %
          (a.arm, 'uniform' if a.arm == 'B' else 'mask',
           'CE+auxiliary' if a.arm == 'A' else 'CE only'), flush=True)
    print('[optim] Adam lr=%g weight_decay=%g wd_scalars=%g betas=(0.5,0.9)' %
          (a.lr, a.weight_decay, a.wd_scalars), flush=True)
    print('[schedule] 30 epochs x 4000 updates; stop at total step %d; current step %d' %
          (a.max_steps, step), flush=True)

    def save_check(is_best):
        state = {'shortcheck_format': 2, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'python_counters': counters(model),
            'train_trace': trace, 'initial_model_sha256': initial_sha,
            'global_step': step, 'epoch_index': (step - 1) // 4000, 'train_plan': plan,
            'best_prec1': best, 'best_step': best_step, 'history': history, 'config': settings,
            'source_sha256': source_hashes, 'pretrained_sha256': pre_hash,
            'val_seed': a.val_seed, 'val_manifest_sha256': manifest_hash,
            'encoder_model': 'ResNet12_ReLU', 'classifier_model': 'NDCN', 'data_name': 'miniImageNet',
            'way_num': 5, 'shot_num': 1, 'query_num': 15, 'ndcn_cfg': cfg}
        names = ['model_last.pth.tar'] + (['model_best.pth.tar'] if is_best else [])
        for name in names:
            temp = out / (name + '.tmp')
            torch.save(state, temp)
            temp.replace(out / name)

    def check():
        nonlocal best, best_step
        scores, result = evaluate(model, val_loader, device)
        np.savez_compressed(out / ('eval_step_%06d.npz' % step), **scores)
        result.update(step=step, lr=optimizer.param_groups[0]['lr'])
        history.append(result)
        write_json(out / 'train_trace.json', trace)
        selection_metric = 'gap' if a.arm == 'B' else 'mask'
        improved = result[selection_metric]['mean'] > best
        if improved:
            best, best_step = result[selection_metric]['mean'], step
        write_json(out / 'summary.json', {'history': history, 'best_val': best,
            'selection_metric': selection_metric, 'arm': a.arm,
            'best_step': best_step, 'last_step': step, 'split': 'val',
            'ci_note': '95% t intervals across fixed validation tasks, conditional on this training run'})
        print(('[CHECK arm=' + a.arm + ' step=%d] GAP %.2f +/- %.2f | MASK %.2f +/- %.2f | best ' + selection_metric + ' %.2f at step %d') %
              (step, result['gap']['mean'], result['gap']['ci95_half_width'],
               result['mask']['mean'], result['mask']['ci95_half_width'], best, best_step), flush=True)
        save_check(improved)

    if step == 0:
        check()
    elif step not in {r['step'] for r in history}:
        check()
    total_loss, total_acc, window_n, start = 0., 0., 0, time.time()
    while step < a.max_steps:
        epoch = step // 4000
        if plan is None:
            # Use the same project training transform and 4000-task epoch size.
            with preserve_state():
                seed_stream(a.seed, epoch, phase=7)
                loaders = FewShotDataloader.get_Fewshot_dataloader(opt, ['train', 'val'])
                train_ds = loaders[0].dataset
                plan = {'epoch': epoch, 'data_list': train_ds.data_list,
                        'order': torch.randperm(4000).tolist(), 'position': 0}
                del loaders
        else:
            # Dataset reconstruction must not advance the saved training RNG.
            with preserve_state():
                train_ds = FewShotDataloader.get_Fewshot_dataloader(opt, ['train'])[0].dataset
            train_ds.data_list = plan['data_list']
            if plan['epoch'] != epoch or plan['position'] != step % 4000:
                raise ValueError('Inconsistent partial-epoch checkpoint')
        local_generator = torch.Generator().manual_seed(a.seed + epoch)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1,
            sampler=plan['order'][plan['position']:], num_workers=0, pin_memory=(a.device == 'cuda'),
            generator=local_generator)
        model.train()
        iterator = iter(train_loader)
        while plan is not None and plan['position'] < 4000:
            seed_stream(a.seed, step, phase=11)
            batch = next(iterator)
            batch_sha = tensor_tree_hash(batch)
            trace.append({'step': step + 1, 'input_sha256': batch_sha,
                          'lr': optimizer.param_groups[0]['lr']})
            seed_stream(a.seed, step, phase=13)
            q, s, y = batch_tensors(batch, device)
            if a.arm == 'C':
                if model.classifier.uniform_mask:
                    raise RuntimeError('C must retain the mask path')
                if any(not m.training for m in model.modules()
                       if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
                    raise RuntimeError('C requires ordinary training-mode BN')
            logits, aux = model(q, s, y)
            ce = F.cross_entropy(logits, y)
            if a.arm in ('B', 'C'):
                if any(not torch.isfinite(v).all() or abs(float(v.detach())) > 1e-12
                       for v in aux.values()):
                    raise RuntimeError('B/C must have finite zero auxiliary losses')
                loss = ce
            else:
                loss = ce + opt.eta * aux['total']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss; previous diagnostic checkpoint preserved')
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('Non-finite gradients; optimizer step cancelled')
            optimizer.step()
            step += 1
            plan['position'] += 1
            total_loss += float(loss.detach())
            total_acc += 100 * (logits.argmax(1) == y).float().mean().item()
            window_n += 1
            if step % 100 == 0:
                print('[TRAIN step=%d epoch=%d] lr=%.8g loss=%.4f acc=%.2f grad_norm=%.3f elapsed=%.0fs' %
                      (step, epoch, optimizer.param_groups[0]['lr'], total_loss/window_n,
                       total_acc/window_n, float(grad_norm), time.time()-start), flush=True)
                total_loss, total_acc, window_n = 0., 0., 0
            at_epoch_end = plan['position'] == 4000
            if at_epoch_end:
                scheduler.step()  # Never step merely because the short check stops.
                plan = None
            if step in targets or at_epoch_end:
                check()
            if step >= a.max_steps or at_epoch_end:
                break
    print('[DONE] stopped automatically at step %d. Results: %s' % (step, out), flush=True)
    print('[DONE] Continue ONLY with this script and model_last.pth.tar; max_steps is a total target.', flush=True)


# Model configuration copied unchanged from the reviewed trainer.
def build_ndcn_cfg(opt):
    return dict(rank=opt.rank, n_iter=opt.n_iter, beta=opt.beta, margin=opt.margin,
        lambda_bg=opt.lambda_bg, lambda_div=opt.lambda_div, lambda_mask=opt.lambda_mask,
        score_mode=opt.score_mode, frontend=opt.frontend, nonneg_mode=opt.nonneg_mode,
        alpha_norm=opt.alpha_norm, alpha_tau=opt.alpha_tau, div_margin=opt.div_margin,
        pool_floor=opt.pool_floor, mask_budget=opt.mask_budget, uniform_mask=opt.uniform_mask,
        sinkhorn_eps=opt.sinkhorn_eps, sinkhorn_iters=opt.sinkhorn_iters,
        ot_chunk=opt.ot_chunk, ot_uniform=opt.ot_uniform, match_mode=opt.match_mode,
        match_tau=opt.match_tau, comp_blend=opt.comp_blend, recon_lam=opt.recon_lam,
        recon_chunk=opt.recon_chunk, recon_mask_power=opt.recon_mask_power,
        use_bg_sep=not opt.no_bg_sep, use_qbg=not opt.no_qbg, use_triplet=not opt.no_triplet,
        use_bgloss=not opt.no_bgloss, use_div=not opt.no_div, use_mask=not opt.no_mask,
        grad_last_only=not opt.grad_all_iters, neighbor_k=opt.neighbor_k, chunk_size=opt.chunk_size)


if __name__ == '__main__':
    args = arguments()  # --help works without importing the training dependencies.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import stats
    from PIL import ImageFile
    import dataset.general_dataloader as FewShotDataloader
    import models.network as FewShotNet
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    main(args)

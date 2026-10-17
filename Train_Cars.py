#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import math
from pathlib import Path
import platform
import time

import torch
import torchvision
from torch.nn import functional as F
from paired_common import (seed_all, audit_data, source_hash, make_model, state_hash,
    Episodes, loader, hash_batch, evaluate, ci, save_json, save_checkpoint, rng_state,
    restore_rng, digest)


def main():
    p = argparse.ArgumentParser(description='Cars 130/17/49: NDPNet or independently trained uniform pooling')
    p.add_argument('--data', required=True, help='Contains train.csv,val.csv,test.csv and images/')
    p.add_argument('--out', required=True, help='Shared experiment root; arm subdirectory created automatically')
    p.add_argument('--arm', choices=['ndm', 'uniform'], required=True)
    p.add_argument('--shot', type=int, choices=[1, 5], default=5)
    p.add_argument('--encoder', choices=['ResNet12_ReLU', 'Conv64F_Local_ReLU'], default='ResNet12_ReLU')
    p.add_argument('--optimizer', choices=['adam', 'sgd'], required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--weight-decay', type=float, default=.0005)
    p.add_argument('--sgd-dampening', type=float, default=.9)
    p.add_argument('--epochs', type=int, default=30, help='Full cosine horizon; do not shorten for a pilot')
    p.add_argument('--episodes', type=int, default=4000)
    p.add_argument('--val-episodes', type=int, default=600)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--val-seed', type=int, default=240920)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-steps', type=int, help='Optional early stop; cosine horizon unchanged')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--backup-dir', help='A genuinely persistent mounted directory; copies both best and last')
    p.add_argument('--cpu', action='store_true', help='For small smoke tests only')
    args = p.parse_args()
    if min(args.epochs, args.episodes, args.val_episodes, args.save_every) < 1 or args.lr <= 0:
        p.error('Invalid training budget or learning rate')
    if args.val_episodes < 2:
        p.error('val-episodes must be at least 2 for a confidence interval')
    if args.max_steps is not None and args.max_steps < 1:
        p.error('max-steps must be positive')
    device = torch.device('cpu' if args.cpu else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install matching torch/torchvision first')
    out = Path(args.out).resolve() / args.arm
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pth.tar').exists() and not args.resume:
        raise FileExistsError('Existing run: use --resume or choose a new --out. Nothing overwritten.')
    cfg = {k: v for k, v in vars(args).items() if k not in
           ('data','out','arm','workers','max_steps','save_every','resume','backup_dir','cpu')}
    cfg.update(dataset_name='StanfordCars', way=5, query=15, initialization='random/from_scratch',
               rank=5, n_iter=3, score='free', normalization='supminmax',
               eta=.5, lambda_div=.2, lambda_mask=.01, margin=.05, div_margin=.3,
               mask_budget=.25, pool_floor=.05, beta=10., clip_grad=5.,
               bn='train', crash_guard=False, precision='float32',
               augmentation='RRC84+jitter(.4,.4,.4)+flip; eval Resize92/CenterCrop84; norm .5',
               adam_betas=[.5,.9], sgd_momentum=.9, nesterov=False,
               wd_scalars='same_as_backbone', eta_min_ratio=.001)
    metadata = dict(config=cfg, dataset=audit_data(args.data, args.shot, full=True),
                    source=source_hash(),
                    environment=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                     python=platform.python_version()))
    seed_all(args.seed)
    model = make_model(cfg, args.arm, device)
    metadata['initial_state_sha256'] = state_hash(model.state_dict())
    metadata['protocol_hash'] = digest(metadata)
    opt = (torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.5,.9), weight_decay=args.weight_decay)
           if args.optimizer == 'adam' else
           torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9,
                           dampening=args.sgd_dampening, weight_decay=args.weight_decay))
    step, best, best_epoch, history, epoch_hashes, current_hashes = 0, -1., -1, [], [], []
    timing = dict(training_loop_wall_seconds=0., update_seconds=0., validation_seconds=0.)
    if args.resume:
        ck = torch.load(out/'last.pth.tar', map_location='cpu', weights_only=False)
        if ck['metadata'] != metadata or ck['arm'] != args.arm:
            raise ValueError('Resume rejected: data/code/config/environment/initialization changed')
        model.load_state_dict(ck['model'], strict=True)
        opt.load_state_dict(ck['optimizer'])
        step, best, best_epoch = ck['step'], ck['best_val'], ck['best_epoch']
        history, epoch_hashes, current_hashes = ck['history'], ck['epoch_hashes'], ck['current_hashes']
        restore_rng(ck['rng'])
        timing = ck['timing']
    save_json(out/'protocol.json', dict(arm=args.arm, **metadata))
    val = Episodes(args.data, 'val', args.shot, args.val_episodes, args.val_seed)
    full_steps = args.epochs * args.episodes
    target = min(args.max_steps or full_steps, full_steps)
    print(f'[PROTOCOL] {args.arm}: {cfg}', flush=True)
    print(f'[INIT] {metadata["initial_state_sha256"]}; step={step}; target={target}/{full_steps}', flush=True)
    start = time.perf_counter()
    past_wall = timing['training_loop_wall_seconds']

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def timing_now():
        return dict(timing, training_loop_wall_seconds=past_wall + time.perf_counter()-start)

    def snapshot(filename):
        payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                       metadata=metadata, arm=args.arm, step=step, best_val=best,
                       best_epoch=best_epoch, history=history, epoch_hashes=epoch_hashes,
                       current_hashes=current_hashes, timing=timing_now())
        save_checkpoint(out/filename, payload, args.backup_dir,
                        f'{Path(args.out).resolve().name}/{args.arm}/{filename}')
        if args.backup_dir:
            save_json(Path(args.backup_dir)/Path(args.out).resolve().name/args.arm/'protocol.json',
                      dict(arm=args.arm, **metadata))

    if step == 0 and not args.resume:
        snapshot('last.pth.tar')
    while step < target:
        epoch = step // args.episodes
        offset = step % args.episodes
        lr = args.lr * (.001 + .999 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
        for group in opt.param_groups:
            group['lr'] = lr
        train = Episodes(args.data, 'train', args.shot, args.episodes, args.seed, epoch)
        model.train()
        for q, s, y, idx in loader(train, args.workers):
            if int(idx) < offset:
                continue
            h = hashlib.sha256(); hash_batch(h, q, s, y, idx, lr)
            sync(); update_start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits, aux = model(q.to(device, non_blocking=True), s.to(device, non_blocking=True),
                                y.to(device, non_blocking=True))
            ce = F.cross_entropy(logits, y.to(device, non_blocking=True))
            loss = ce + .5 * aux['total']
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss before step {step+1}; last checkpoint retained')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            sync(); timing['update_seconds'] += time.perf_counter()-update_start
            current_hashes.append(h.hexdigest()); step += 1
            if step % 100 == 0:
                print(f'[TRAIN {args.arm}] step={step} epoch={epoch+1} lr={lr:.8g} '
                      f'loss={float(loss.detach()):.4f} ce={float(ce.detach()):.4f} '
                      f'grad_norm={float(norm):.3f} elapsed={time.perf_counter()-start:.0f}s', flush=True)
            if step % args.episodes == 0:
                assert len(current_hashes) == args.episodes
                epoch_hashes.append(dict(epoch=epoch+1, tasks=train.task_hash,
                                         inputs_and_lr=digest(current_hashes), lr=lr))
                current_hashes = []
                states = rng_state()
                sync(); val_start = time.perf_counter()
                ev = evaluate(model, val, device, args.workers)
                sync(); timing['validation_seconds'] += time.perf_counter()-val_start
                restore_rng(states)
                result = ci(ev['accuracy'])
                record = dict(epoch=epoch+1, step=step, accuracy=result,
                              val_inputs=ev['input_hash'], val_tasks=ev['task_hash'], timing=timing_now())
                history.append(record)
                improved = result['mean'] > best
                if improved:
                    best, best_epoch = result['mean'], epoch+1
                print(f'[VAL {args.arm}] epoch={epoch+1} {result["mean"]:.4f} +/- '
                      f'{result["ci95_half_width"]:.4f}; best={best:.4f}@{best_epoch}', flush=True)
                if improved:
                    snapshot('best.pth.tar')
                save_json(out/'history.json', history)
                model.train()
            if step % args.save_every == 0 or step % args.episodes == 0 or step == target:
                snapshot('last.pth.tar')
            if step >= target:
                break
    save_json(out/'training_summary.json', dict(arm=args.arm, metadata=metadata,
              step=step, full_steps=full_steps, complete=step==full_steps,
              best_val=best, best_epoch=best_epoch, history=history, epoch_hashes=epoch_hashes, timing=timing_now()))
    save_json(out/'training_time.json', dict(**timing_now(),
              note='Elapsed executed training-loop time, including validation/loading/checkpoints; excludes offline downtime and initialization. Update time includes H2D and backward/optimizer, excludes image loading. Resume may replay unsaved steps.'))
    print('[TIME]', timing_now(), flush=True)
    print(f'[DONE] {step}/{full_steps}; {out}; select by VAL only; test not evaluated.', flush=True)


if __name__ == '__main__':
    main()

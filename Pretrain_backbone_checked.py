#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# rev 2026-09-04
"""
Base-class pre-training for NDPNet.

    python Pretrain_backbone.py \
        --dataset_dir /root/miniImageNet/For_FewShot --data_name miniImageNet \
        --encoder_model ResNet12_ReLU --epochs 90 \
        --outf ./pretrain/mini_r12

Trains the backbone as an ordinary classifier over the base classes of the
same dataset, then saves the backbone weights so that Train_NDCN.py can
start episodic training from them (--pretrained).

This is domain-internal: only the training split of the dataset in question
is used, with no external weights and no data from validation or test
classes. The script implements the standard linear-classifier variant of
base-class pre-training (global average pooling + FC + cross-entropy). It
shares the two-stage idea of Meta-Baseline and KLSANet, but it does not
reproduce the FRN/EGAL proxy reconstruction head or the two-branch TST_MFL
objective. State the exact protocol next to every reported result.

Model selection follows Meta-Baseline rather than classification accuracy:
every --val_freq epochs the backbone is evaluated with 5-way 1-shot cosine
prototype episodes on the validation classes, and the checkpoint with the
best episodic accuracy is kept. Base-class training accuracy may keep rising
after the features stop transferring, so it is not used as the checkpoint-
selection criterion.
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.dont_write_bytecode = True
import models.network as FewShotNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', required=True)
parser.add_argument('--data_name', default='miniImageNet')
parser.add_argument('--encoder_model', default='ResNet12_ReLU',
                    choices=['Conv64F_Local_ReLU', 'ResNet12_ReLU'])
parser.add_argument('--outf', required=True)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--milestones', default='60,80',
                    help='comma-separated epochs at which the lr is divided by 10')
parser.add_argument('--label_smooth', type=float, default=0.0)
parser.add_argument('--val_freq', type=int, default=5)
parser.add_argument('--val_episodes', type=int, default=400,
                    help='5-way 1-shot episodes used to pick the checkpoint')
parser.add_argument('--eval_resize', type=int, default=92,
                    help='resize before the 84x84 center crop; 92 matches the episodic dataloader')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resume', default='',
                    help='optional backbone_last.pth.tar produced by this script')
parser.add_argument('--deterministic', action='store_true',
                    help='use deterministic CUDA kernels where available')
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no_cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
opt = parser.parse_args()

os.makedirs(opt.outf, exist_ok=True)
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
cudnn.benchmark = not opt.deterministic
cudnn.deterministic = opt.deterministic
device = torch.device('cuda' if (opt.cuda and torch.cuda.is_available()) else 'cpu')

LOG = open(os.path.join(opt.outf, 'pretrain_log.txt'), 'a+')


def log(msg):
    print(msg)
    print(msg, file=LOG)
    LOG.flush()


log('\n' + '=' * 60)
log(str(opt))

# ---------------------------------------------------------------- data
NORM = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(opt.imageSize),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), NORM])
eval_tf = transforms.Compose([
    transforms.Resize((opt.eval_resize, opt.eval_resize)),
    transforms.CenterCrop(opt.imageSize),
    transforms.ToTensor(), NORM])


def read_split(csv_path):
    per_class = {}
    with open(csv_path) as f:
        rd = csv.reader(f)
        next(rd, None)
        for line in rd:
            if len(line) < 2:
                continue
            per_class.setdefault(line[1], []).append(line[0])
    names = sorted(per_class)
    return per_class, names


class BaseClassDataset(Dataset):
    """Every image of the base classes, with an ordinary integer label."""

    def __init__(self, root, per_class, names, transform):
        self.root = os.path.join(root, 'images')
        self.transform = transform
        self.items = []
        idx = {c: i for i, c in enumerate(names)}
        for c in names:
            for fn in per_class[c]:
                self.items.append((fn, idx[c]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, y = self.items[i]
        img = Image.open(os.path.join(self.root, fn)).convert('RGB')
        return self.transform(img), y


class EpisodeSet(Dataset):
    """5-way 1-shot episodes over the validation classes, for model selection."""

    def __init__(self, root, per_class, names, transform, n_episodes, seed=0):
        self.root = os.path.join(root, 'images')
        self.transform = transform
        rng = random.Random(seed)          # fixed, so every check sees the same tasks
        self.episodes = []
        usable = [c for c in names if len(per_class[c]) >= 16]
        if len(usable) < 5:
            raise ValueError('validation split has fewer than five usable classes')
        for _ in range(n_episodes):
            cls = rng.sample(usable, 5)
            sup, qry = [], []
            for c in cls:
                picks = rng.sample(per_class[c], 16)
                sup.append(picks[0])
                qry.extend(picks[1:16])
            self.episodes.append((sup, qry))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        sup, qry = self.episodes[i]
        load = lambda fn: self.transform(
            Image.open(os.path.join(self.root, fn)).convert('RGB'))
        s = torch.stack([load(f) for f in sup])
        q = torch.stack([load(f) for f in qry])
        y = torch.arange(5).repeat_interleave(15)
        return s, q, y


train_pc, train_names = read_split(os.path.join(opt.dataset_dir, 'train.csv'))
val_pc, val_names = read_split(os.path.join(opt.dataset_dir, 'val.csv'))
overlap = set(train_names) & set(val_names)
if overlap:
    raise ValueError('train/val class leakage: %s' % sorted(overlap)[:5])
n_base = len(train_names)
if n_base < 5:
    raise ValueError('training split has fewer than five classes')
log('base classes %d | base images %d | val classes %d'
    % (n_base, sum(len(v) for v in train_pc.values()), len(val_names)))

train_loader = DataLoader(
    BaseClassDataset(opt.dataset_dir, train_pc, train_names, train_tf),
    batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
    pin_memory=True, drop_last=True)
val_loader = DataLoader(
    EpisodeSet(opt.dataset_dir, val_pc, val_names, eval_tf, opt.val_episodes),
    batch_size=1, shuffle=False, num_workers=max(1, opt.workers // 2))

# ---------------------------------------------------------------- model
# The backbone is taken from the same factory the episodic script uses, so
# the state dict keys match and no renaming is needed on load.
shell = FewShotNet.define_model(
    encoder_model=opt.encoder_model, classifier_model='NDCN',
    way_num=5, shot_num=1, init_type='normal', use_gpu=False,
    ndcn_cfg=dict(rank=5, score_mode='free', alpha_norm='supminmax'))
backbone = shell.features
del shell

with torch.no_grad():
    feat_dim = backbone(torch.randn(2, 3, opt.imageSize, opt.imageSize)).shape[1]
head = nn.Linear(feat_dim, n_base)
backbone, head = backbone.to(device), head.to(device)
log('feature dim %d | backbone params %d'
    % (feat_dim, sum(p.numel() for p in backbone.parameters())))

criterion = nn.CrossEntropyLoss(label_smoothing=opt.label_smooth)
optimizer = optim.SGD(list(backbone.parameters()) + list(head.parameters()),
                      lr=opt.lr, momentum=opt.momentum, nesterov=True,
                      weight_decay=opt.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(x) for x in opt.milestones.split(',')], gamma=0.1)


@torch.no_grad()
def episodic_val():
    """5-way 1-shot cosine prototype accuracy, with plain average pooling."""
    backbone.eval()
    accs = []
    for s, q, y in val_loader:
        s, q, y = s[0].to(device), q[0].to(device), y[0].to(device)
        fs = backbone(s).mean(dim=(2, 3))
        fq = backbone(q).mean(dim=(2, 3))
        logit = F.normalize(fq, dim=1) @ F.normalize(fs, dim=1).t()
        accs.append((logit.argmax(1) == y).float().mean().item())
    # Sample standard deviation (ddof=1) gives the usual 95% confidence
    # interval reported in few-shot papers.
    sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    return 100.0 * float(np.mean(accs)), 100.0 * 1.96 * sd / np.sqrt(len(accs))


best, best_epoch, start_epoch = 0.0, -1, 0
if opt.resume:
    resume = torch.load(opt.resume, map_location='cpu', weights_only=False)
    if resume.get('encoder_model') not in (None, opt.encoder_model):
        raise ValueError('resume checkpoint encoder is %s, requested %s'
                         % (resume.get('encoder_model'), opt.encoder_model))
    if resume.get('data_name') not in (None, opt.data_name):
        raise ValueError('resume checkpoint dataset is %s, requested %s'
                         % (resume.get('data_name'), opt.data_name))
    if int(resume.get('n_base_classes', n_base)) != n_base:
        raise ValueError('resume checkpoint has %s base classes, current split has %d'
                         % (resume.get('n_base_classes'), n_base))
    backbone.load_state_dict(resume['backbone'], strict=True)
    head.load_state_dict(resume['head'], strict=True)
    optimizer.load_state_dict(resume['optimizer'])
    scheduler.load_state_dict(resume['scheduler'])
    start_epoch = int(resume['epoch']) + 1
    best = float(resume.get('best_episodic_val', 0.0))
    best_epoch = int(resume.get('best_epoch', -1))
    log('resumed %s at epoch %d | best %.2f at epoch %d'
        % (opt.resume, start_epoch, best, best_epoch))


def checkpoint_state(epoch):
    return {
        'backbone': backbone.state_dict(),
        'head': head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'encoder_model': opt.encoder_model,
        'data_name': opt.data_name,
        'epoch': epoch,
        'best_episodic_val': best,
        'best_epoch': best_epoch,
        'n_base_classes': n_base,
        'protocol': 'linear-fc base-class classification (GAP+FC+CE)',
        'image_size': opt.imageSize,
        'eval_resize': opt.eval_resize,
        'seed': opt.seed,
    }


for epoch in range(start_epoch, opt.epochs):
    backbone.train(); head.train()
    tot = corr = 0
    run_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = head(backbone(x).mean(dim=(2, 3)))
        loss = criterion(out, y)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(head.parameters()), 5.0)
        optimizer.step()
        run_loss += loss.item() * y.size(0)
        corr += (out.argmax(1) == y).sum().item(); tot += y.size(0)
        if i % 100 == 0:
            log('  epoch %3d [%4d/%4d] loss %.3f  base acc %.2f%%  lr %.4f'
                % (epoch, i, len(train_loader), run_loss / max(tot, 1),
                   100.0 * corr / max(tot, 1), optimizer.param_groups[0]['lr']))
    scheduler.step()

    if (epoch + 1) % opt.val_freq == 0 or epoch == opt.epochs - 1:
        acc, h = episodic_val()
        flag = ''
        if acc > best:
            best, best_epoch = acc, epoch
            state = checkpoint_state(epoch)
            # Keep this compatibility field because Train_NDCN.py prints it.
            state['episodic_val'] = acc
            torch.save(state, os.path.join(opt.outf, 'backbone_best.pth.tar'))
            flag = '   <- saved'
        log('epoch %3d | base acc %.2f%% | 5-way 1-shot val %.2f +/- %.2f%s'
            % (epoch, 100.0 * corr / max(tot, 1), acc, h, flag))

    # This checkpoint is for interruption recovery. The best checkpoint above
    # remains the one used to initialise episodic training.
    torch.save(checkpoint_state(epoch),
               os.path.join(opt.outf, 'backbone_last.pth.tar'))

log('\nbest episodic val %.2f%% at epoch %d' % (best, best_epoch))
log('saved to %s' % os.path.join(opt.outf, 'backbone_best.pth.tar'))
log('\nNext: pass it to episodic training, e.g.')
log('  python Train_NDCN.py ... --pretrained %s/backbone_best.pth.tar'
    % opt.outf)
log('Report the pre-training protocol next to any number obtained this way.')

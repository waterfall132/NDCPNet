#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test NDCN and dump the mechanism diagnostics needed for the paper's
verification tables.

    python Test_NDCN.py --resume ./results/NDCN_.../ \
        --encoder_model ResNet12_ReLU --classifier_model NDCN --dump_diag
"""

from __future__ import print_function
import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
from PIL import ImageFile
import sys
sys.dont_write_bytecode = True

import dataset.general_dataloader as FewShotDataloader
import models.network as FewShotNet
from models.ndcn import make_nonneg
import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data/Datasets/CUB_birds/For_FewShot')
parser.add_argument('--data_name', default='CubBird')
parser.add_argument('--mode', default='test')
parser.add_argument('--outf', default='./results/')
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--encoder_model', default='ResNet12_ReLU')
parser.add_argument('--classifier_model', default='NDCN')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--train_aug', action='store_true', default=True)
parser.add_argument('--test_aug', action='store_true', default=False)
parser.add_argument('--episodeSize', type=int, default=1)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--episode_train_num', type=int, default=4000)
parser.add_argument('--episode_val_num', type=int, default=1000)
parser.add_argument('--episode_test_num', type=int, default=2000)
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--aug_shot_num', type=int, default=20)
parser.add_argument('--neighbor_k', type=int, default=3)
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no_cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--print_freq', '-p', default=200, type=int)
parser.add_argument('--repeat_num', type=int, default=5)
parser.add_argument('--seed', type=int, default=0,
					help='round r uses seed+r; reuse this value for paired checkpoints')
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--dump_diag', action='store_true', default=False,
					help='save per-query diagnostics to diagnostics.npz for the binning analysis')
parser.add_argument('--eval_mask', default='checkpoint',
					choices=['checkpoint', 'mask', 'uniform'],
					help='override the checkpoint at test time; use the same seeds to form '
						 'the four-cell train-mask/test-mask comparison')
opt = parser.parse_args()
cudnn.benchmark = not opt.deterministic
cudnn.deterministic = opt.deterministic
device = torch.device('cuda' if (opt.cuda and torch.cuda.is_available()) else 'cpu')


def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def collect_diag(model, input_q, input_s, target, logits):
	"""
    Per-query quantities used by the mechanism-verification and clutter-binning
    analyses. R_before is computed WITHOUT segmentation labels, as the share of
    descriptor energy carried by the top-half most-activated positions; it is a
    proxy for foreground prevalence, so state it as such in the paper.
    """
	m = model.module if hasattr(model, 'module') else model
	c = m.classifier
	feat_q = m.features(input_q)
	Zq = make_nonneg(c._flatten(feat_q), c.nonneg_mode, c.eps)      # NQ x L x C

	nrm = Zq.norm(dim=2)                                            # NQ x L
	L = nrm.size(1)
	k = max(L // 2, 1)
	topk = nrm.topk(k, dim=1).values
	r_before = topk.sum(1) / (nrm.sum(1) + c.eps)                   # NQ

	# forward() clears the support normalization range. Recomputing _mask
	# here would silently use the query range under supminmax. Read the
	# actual mask cached by the supplied 0906 models/ndcn.py instead.
	if getattr(c, '_last_Mq', None) is None:
		raise RuntimeError('No cached forward mask; use the supplied 0906 models/ndcn.py')
	Mq, alpha_q = c._last_Mq, c._last_alpha_q
	mask_mean = Mq.mean(dim=1)
	# concentration of mask mass on the high-energy half
	order = nrm.argsort(dim=1, descending=True)
	mask_sorted = torch.gather(Mq, 1, order)
	r_after = mask_sorted[:, :k].sum(1) / (Mq.sum(1) + c.eps)

	correct = (logits.argmax(dim=1) == target).float()
	n_hi = (alpha_q > 0.7).float().sum(1) if alpha_q is not None else torch.zeros_like(mask_mean)

	return dict(r_before=r_before.cpu().numpy(),
				r_after=r_after.cpu().numpy(),
				mask_mean=mask_mean.cpu().numpy(),
				n_alpha_hi=n_hi.cpu().numpy(),
				correct=correct.cpu().numpy())


def test(test_loader, model, criterion, F_txt, dump=False):
	batch_time, losses, top1 = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
	model.eval()
	accuracies = []
	diag_all = {k: [] for k in ['r_before', 'r_after', 'mask_mean', 'n_alpha_hi', 'correct']}

	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):
		input_var1 = torch.cat(query_images, 0).to(device)
		input_var2 = torch.cat(support_images, 0).squeeze(0).to(device)
		input_var2 = input_var2.contiguous().view(-1, input_var2.size(2),
												 input_var2.size(3), input_var2.size(4))
		target = torch.cat(query_targets, 0).to(device)

		output = model(input_var1, input_var2)
		loss = criterion(output, target)

		if dump:
			d = collect_diag(model, input_var1, input_var2, target, output)
			for k in diag_all:
				diag_all[k].append(d[k])

		prec1, _ = utils.accuracy(output, target, topk=(1, 3))
		losses.update(loss.item(), target.size(0))
		top1.update(prec1[0], target.size(0))
		accuracies.append(prec1)

		batch_time.update(time.time() - end)
		end = time.time()

		if episode_index % opt.print_freq == 0 and episode_index != 0:
			msg = ('Test: [{0}/{1}]\tTime {bt.val:.2f} ({bt.avg:.2f})\t'
				   'Loss {loss.val:.2f} ({loss.avg:.2f})\tPrec@1 {top1.val:.2f} ({top1.avg:.2f})').format(
					   episode_index, len(test_loader), bt=batch_time, loss=losses, top1=top1)
			print(msg); print(msg, file=F_txt)

	print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
	print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1), file=F_txt)
	diag = {k: np.concatenate(v) for k, v in diag_all.items()} if dump else None
	return top1.avg, losses.avg, accuracies, diag


def report_binned(diag, F_txt, n_bins=4):
	"""Clutter-severity binning: accuracy as a function of foreground prevalence."""
	rb, ok = diag['r_before'], diag['correct']
	edges = np.quantile(rb, np.linspace(0, 1, n_bins + 1))
	print('\n--- Accuracy binned by R_before (proxy for foreground prevalence) ---')
	print('--- Accuracy binned by R_before (proxy for foreground prevalence) ---', file=F_txt)
	hdr = '{:>6} {:>16} {:>9} {:>9} {:>9}'.format('bin', 'R_before range', '#samples', 'acc(%)', 'mask_mu')
	print(hdr); print(hdr, file=F_txt)
	for b in range(n_bins):
		lo, hi = edges[b], edges[b + 1]
		sel = (rb >= lo) & (rb <= hi if b == n_bins - 1 else rb < hi)
		if sel.sum() == 0:
			continue
		row = '{:>6} {:>7.3f}-{:.3f} {:>9d} {:>9.2f} {:>9.3f}'.format(
			b + 1, lo, hi, int(sel.sum()), 100.0 * ok[sel].mean(), diag['mask_mean'][sel].mean())
		print(row); print(row, file=F_txt)


if __name__ == '__main__':
	if opt.cuda and not torch.cuda.is_available():
		raise RuntimeError('--cuda was requested but CUDA is not available')
	seed_everything(opt.seed)
	best_model_path = os.path.join(opt.resume, 'model_best.pth.tar')
	ckpt = torch.load(best_model_path, map_location='cpu', weights_only=False)
	if ckpt.get('encoder_model') not in (None, opt.encoder_model):
		raise ValueError('checkpoint encoder is %s, requested %s'
						 % (ckpt.get('encoder_model'), opt.encoder_model))
	if ckpt.get('classifier_model') not in (None, opt.classifier_model):
		raise ValueError('checkpoint classifier is %s, requested %s'
						 % (ckpt.get('classifier_model'), opt.classifier_model))
	if ckpt.get('data_name') not in (None, opt.data_name):
		raise ValueError('checkpoint dataset is %s, requested %s'
						 % (ckpt.get('data_name'), opt.data_name))
	if ckpt.get('shot_num') not in (None, opt.shot_num):
		raise ValueError('checkpoint is %s-shot, requested %s-shot'
						 % (ckpt.get('shot_num'), opt.shot_num))
	ndcn_cfg = ckpt.get('ndcn_cfg', None)

	model = FewShotNet.define_model(
		encoder_model=opt.encoder_model, classifier_model=opt.classifier_model,
		norm='batch', way_num=opt.way_num, shot_num=opt.shot_num,
		init_type='normal', use_gpu=opt.cuda, ndcn_cfg=ndcn_cfg)

	criterion = nn.CrossEntropyLoss().to(device)
	opt.outf, F_txt = utils.set_save_test_path2(opt)
	model.load_state_dict(ckpt['model'])
	if opt.classifier_model.startswith('NDCN') and opt.eval_mask != 'checkpoint':
		classifier = model.module.classifier if hasattr(model, 'module') else model.classifier
		classifier.uniform_mask = (opt.eval_mask == 'uniform')
	print('=> loaded checkpoint (epoch %s, best_prec1 %.3f)' % (ckpt['epoch_index'], ckpt['best_prec1']))
	print('=> initialization: %s' % ckpt.get('initialization', 'not recorded'))
	print('=> evaluation mask mode: %s' % opt.eval_mask)
	print(opt); print(opt, file=F_txt)

	total_accuracy, total_h = 0.0, np.zeros(opt.repeat_num)
	merged = None
	all_episode_acc = []
	for r in range(opt.repeat_num):
		# Construct exactly the same episodes for every checkpoint tested with
		# the same --seed. This makes mask-vs-uniform comparisons paired.
		seed_everything(opt.seed + r)
		print('==================== Round %d ====================' % r)
		print('==================== Round %d ====================' % r, file=F_txt)
		_, _, test_loader = FewShotDataloader.get_Fewshot_dataloader(opt, ['train', 'val', 'test'])
		with torch.no_grad():
			prec1, _, accuracies, diag = test(test_loader, model, criterion, F_txt,
											  dump=opt.dump_diag and r == 0)
		acc, h = utils.mean_confidence_interval(accuracies)
		# utils.accuracy() keeps a singleton dim, so scipy returns shape-(1,)
		# arrays here. numpy < 2.0 unwrapped them silently; numpy >= 2.0 raises.
		acc = float(np.asarray(acc).reshape(-1)[0])
		h = float(np.asarray(h).reshape(-1)[0])
		total_accuracy += acc
		total_h[r] = h
		all_episode_acc.append(np.asarray([
			float(np.asarray(a.detach().cpu()).reshape(-1)[0]) for a in accuracies
		], dtype=np.float64))
		if diag is not None:
			merged = diag
		print('Test accuracy: %f h: %f\n' % (acc, h))
		print('Test accuracy: %f h: %f\n' % (acc, h), file=F_txt)

	print('Mean_accuracy: %f h: %f' % (total_accuracy / opt.repeat_num, total_h.mean()))
	print('Mean_accuracy: %f h: %f' % (total_accuracy / opt.repeat_num, total_h.mean()), file=F_txt)
	paired_path = os.path.join(
		opt.outf, 'episode_accuracy_seed%d_%s.npz' % (opt.seed, opt.eval_mask))
	np.savez(paired_path, accuracy=np.stack(all_episode_acc, axis=0),
			 seed=opt.seed, eval_mask=opt.eval_mask)
	print('Saved paired episode accuracies to %s' % paired_path)

	m = model.module if hasattr(model, 'module') else model
	if m.classifier_model.startswith('NDCN'):
		c = m.classifier
		line = ('[logit scalars] gamma_fg = {:.3f}  gamma_bg = {:.3f}  beta_bg = {:+.3f}'
				.format(float(c.gamma_fg), float(c.gamma_bg), float(c.beta_bg)))
		print(line); print(line, file=F_txt)

	if merged is not None:
		np.savez(os.path.join(opt.outf, 'diagnostics.npz'), **merged)
		report_binned(merged, F_txt)
		print('\nSaved per-query diagnostics to %s' % os.path.join(opt.outf, 'diagnostics.npz'))

	F_txt.close()

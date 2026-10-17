#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train NDCN (Non-negative Decoupling and Contrast Network).
Built on the DN4 (2023) codebase by Wenbin Li.

Quick start
    python Train_NDCN.py --dataset_dir /path/to/CUB --data_name CubBird \
        --encoder_model ResNet12_ReLU --classifier_model NDCN \
        --way_num 5 --shot_num 1
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
import torch.optim as optim
import time
from PIL import ImageFile
import sys
sys.dont_write_bytecode = True

import dataset.general_dataloader as FewShotDataloader
import models.network as FewShotNet
import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


parser = argparse.ArgumentParser()
# ------------------------------ data / io ---------------------------------
parser.add_argument('--dataset_dir', default='/data/Datasets/CUB_birds/For_FewShot')
parser.add_argument('--data_name', default='CubBird', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
parser.add_argument('--mode', default='train')
parser.add_argument('--outf', default='./results/NDCN_')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--pretrained', default='', type=str,
					help='backbone_best.pth.tar from Pretrain_backbone.py. Loads the '
						 'backbone only and leaves the schedule untouched. Numbers '
						 'obtained this way use linear-FC base-class pre-training; '
						 'state this exact protocol in the paper.')
parser.add_argument('--warm_from', default='', type=str,
					help='path to a model_best.pth.tar to WARM RESTART from. Loads the '
						 'weights only and runs a fresh cosine schedule over --epochs at '
						 '--lr, so the learning rate is not left at the annealed value of '
						 'the finished run. Unlike --resume this does not restore the '
						 'optimizer state or the epoch counter.')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--train_aug', action='store_true', default=True)
parser.add_argument('--test_aug', action='store_true', default=False)
# ------------------------------ model -------------------------------------
parser.add_argument('--encoder_model', default='ResNet12_ReLU',
					help='Conv64F_Local_ReLU|ResNet12_ReLU|Conv64F_Local|ResNet12')
parser.add_argument('--classifier_model', default='NDCN', help='NDCN|NDCN_OT|NDCN_Comp|NDCN_Recon|NDCN_Local|DN4|ProtoNet')
parser.add_argument('--neighbor_k', type=int, default=3)
# ------------------------------ few-shot ----------------------------------
parser.add_argument('--episodeSize', type=int, default=1)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--episode_train_num', type=int, default=4000)
parser.add_argument('--episode_val_num', type=int, default=1000)
parser.add_argument('--episode_test_num', type=int, default=1000)
parser.add_argument('--monitor_test', action='store_true', default=False,
					help='legacy debugging only: evaluate test classes every epoch. Leave '
						 'off for all reportable experiments; use Test_NDCN after training.')
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--aug_shot_num', type=int, default=20)
# ------------------------------ NDCN --------------------------------------
ndcn = parser.add_argument_group('NDCN')
ndcn.add_argument('--chunk_size', type=int, default=0,
				  help='NDCN_Local only: split the query batch to cap the cosine matrix (0 = off)')
ndcn.add_argument('--rank', type=int, default=5, help='r: number of NMF components')
ndcn.add_argument('--n_iter', type=int, default=3, help='T: multiplicative updates')
ndcn.add_argument('--beta', type=float, default=10.0, help='init of gamma_fg (gamma_bg = beta/2)')
ndcn.add_argument('--margin', type=float, default=0.3, help='Delta in the triplet/bg losses')
ndcn.add_argument('--eta', type=float, default=0.1, help='weight of the whole auxiliary loss')
ndcn.add_argument('--lambda_bg', type=float, default=0.05)
ndcn.add_argument('--lambda_div', type=float, default=0.01)
ndcn.add_argument('--lambda_mask', type=float, default=0.001)
ndcn.add_argument('--nonneg_mode', default='relu', choices=['relu', 'shift', 'abs', 'none'],
				  help='how to enforce Z >= 0 when the backbone is not ReLU-terminated')
ndcn.add_argument('--sinkhorn_eps', type=float, default=0.05,
				  help='NDCN_OT: entropic regularisation. Smaller approaches exact transport '
					   'but needs more iterations and is less stable.')
ndcn.add_argument('--sinkhorn_iters', type=int, default=20,
				  help='NDCN_OT: Sinkhorn iterations.')
ndcn.add_argument('--ot_chunk', type=int, default=0,
				  help='NDCN_OT: split the query batch to cap memory (0 = auto).')
ndcn.add_argument('--ot_uniform', action='store_true', default=False,
				  help='NDCN_OT: use uniform transport marginals instead of the mask. '
					   'This is the ablation that isolates what the mask contributes.')
ndcn.add_argument('--match_mode', default='max', choices=['max', 'softmax'],
				  help='NDCN_Comp: how each query component is matched to the class '
					   'components. max is permutation-invariant and parameter-free.')
ndcn.add_argument('--match_tau', type=float, default=10.0,
				  help='NDCN_Comp: temperature for match_mode=softmax.')
ndcn.add_argument('--comp_blend', type=float, default=0.0,
				  help='NDCN_Comp: mix the pooled score in with this weight (0 = pure '
					   'component matching, 1 = the pooled baseline). For ablation.')
ndcn.add_argument('--recon_lam', type=float, default=1.0,
				  help='NDCN_Recon: initial ridge coefficient (learned, kept positive).')
ndcn.add_argument('--recon_chunk', type=int, default=0,
				  help='NDCN_Recon: split the query batch to cap memory (0 = off).')
ndcn.add_argument('--recon_mask_power', type=float, default=1.0,
				  help='NDCN_Recon: exponent on the mask before it weights the subspace.')
ndcn.add_argument('--uniform_mask', action='store_true', default=False,
				  help='force the mask to 1 everywhere, so masked pooling reduces to '
					   'average pooling. This is the baseline row of the ablation table; '
					   'L_mask is switched off automatically.')
ndcn.add_argument('--mask_budget', type=float, default=0.0,
				  help='target mask occupancy rho. >0 switches L_mask to the budget form '
					   '[mean(M) - rho]_+, which stops driving the mask sparser once it '
					   'reaches rho. 0 keeps the plain linear penalty.')
ndcn.add_argument('--pool_floor', type=float, default=0.05,
				  help='NaN guard: lower bound on the mask-pooling denominator, as a fraction '
					   'of the number of positions. Does not damp gradients (cosine is '
					   'scale-invariant) -- use --clip_grad for that.')
ndcn.add_argument('--clip_grad', type=float, default=5.0,
				  help='max global gradient norm; 0 disables clipping.')
ndcn.add_argument('--crash_tol', type=float, default=0.60,
				  help='if validation accuracy falls below this fraction of the best so far, '
					   'reload the best checkpoint and continue. 0 disables the guard.')
ndcn.add_argument('--alpha_norm', default='minmax', choices=['minmax', 'setminmax', 'supminmax', 'zsig', 'none'],
				  help="rescale component scores across [0,1]. 'minmax' normalises each image "
					   "against its own components; 'setminmax' normalises against the whole "
					   "support or query set, making the mask depend on the episode; 'none' "
					   "which cannot go below 0.5 for non-negative features")
ndcn.add_argument('--alpha_tau', type=float, default=8.0, help='temperature for alpha_norm=zsig')
ndcn.add_argument('--div_margin', type=float, default=0.0,
				  help='>0 switches L_div to the margin-hinge form (recommended: 0.3)')
ndcn.add_argument('--score_mode', default='ref', choices=['ref', 'centered', 'free'],
				  help='ref = reference-direction alpha; free = activation-norm alpha')
ndcn.add_argument('--frontend', default='ndm', choices=['ndm', 'mustd'])
ndcn.add_argument('--grad_all_iters', action='store_true', default=False,
				  help='backprop through all T updates instead of only the last')
# --------------------- ablation switches (all default ON) ------------------
ab = parser.add_argument_group('ablation')
ab.add_argument('--no_bg_sep', action='store_true', help='drop d_bg_sep from the logit')
ab.add_argument('--no_qbg', action='store_true', help='drop d_qbg from the logit  <-- the (c+) run')
ab.add_argument('--no_triplet', action='store_true')
ab.add_argument('--no_bgloss', action='store_true')
ab.add_argument('--no_div', action='store_true')
ab.add_argument('--no_mask', action='store_true')
# ------------------------------ optim -------------------------------------
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--cosine', dest='cosine', action='store_true')
parser.add_argument('--no_cosine', dest='cosine', action='store_false')
parser.set_defaults(cosine=True)
parser.add_argument('--lr_decay_epochs', type=list, default=[60, 80])
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.001,
					help='L2 on the backbone. The DN4 code this is built on hard-coded 0.001 '
					     'and ignored the flag; 0.001 is kept as the default so runs stay comparable.')
parser.add_argument('--wd_scalars', type=float, default=-1.0,
					help='weight decay for the three logit scalars; -1 = same as the backbone. '
					     'Set 0 to stop L2 from pulling the temperatures toward zero, which '
					     'otherwise contaminates their reported values.')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no_cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--print_freq', '-p', default=100, type=int)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--deterministic', action='store_true',
					help='use deterministic CUDA kernels where available')
opt = parser.parse_args()
cudnn.benchmark = not opt.deterministic
cudnn.deterministic = opt.deterministic

torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
# episodes are drawn with random.sample() in the dataloader, so Python's own
# RNG has to be seeded as well or every run sees different tasks
random.seed(opt.seed)
device = torch.device('cuda' if (opt.cuda and torch.cuda.is_available()) else 'cpu')


def build_ndcn_cfg(opt):
	return dict(
		rank=opt.rank, n_iter=opt.n_iter, beta=opt.beta, margin=opt.margin,
		lambda_bg=opt.lambda_bg, lambda_div=opt.lambda_div, lambda_mask=opt.lambda_mask,
		score_mode=opt.score_mode, frontend=opt.frontend, nonneg_mode=opt.nonneg_mode,
		alpha_norm=opt.alpha_norm, alpha_tau=opt.alpha_tau, div_margin=opt.div_margin,
		pool_floor=opt.pool_floor, mask_budget=opt.mask_budget,
		uniform_mask=opt.uniform_mask,
		sinkhorn_eps=opt.sinkhorn_eps, sinkhorn_iters=opt.sinkhorn_iters,
		ot_chunk=opt.ot_chunk, ot_uniform=opt.ot_uniform,
		match_mode=opt.match_mode, match_tau=opt.match_tau,
		comp_blend=opt.comp_blend,
		recon_lam=opt.recon_lam, recon_chunk=opt.recon_chunk,
		recon_mask_power=opt.recon_mask_power,
		use_bg_sep=not opt.no_bg_sep, use_qbg=not opt.no_qbg,
		use_triplet=not opt.no_triplet, use_bgloss=not opt.no_bgloss,
		use_div=not opt.no_div, use_mask=not opt.no_mask,
		grad_last_only=not opt.grad_all_iters,
		neighbor_k=opt.neighbor_k, chunk_size=opt.chunk_size,
	)


# ============================== train =====================================
def train(train_loader, model, criterion, optimizer, epoch_index, F_txt):
	batch_time, data_time = utils.AverageMeter(), utils.AverageMeter()
	losses, top1 = utils.AverageMeter(), utils.AverageMeter()
	aux_meters = {k: utils.AverageMeter() for k in ['triplet', 'bg', 'div', 'mask']}

	model.train()
	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader):
		data_time.update(time.time() - end)

		input_var1 = torch.cat(query_images, 0).to(device)
		input_var2 = torch.cat(support_images, 0).squeeze(0).to(device)
		input_var2 = input_var2.contiguous().view(-1, input_var2.size(2),
												 input_var2.size(3), input_var2.size(4))
		target = torch.cat(query_targets, 0).to(device)

		if opt.classifier_model.startswith('NDCN'):
			output, aux = model(input_var1, input_var2, target)
			loss = criterion(output, target) + opt.eta * aux['total']
			for k in aux_meters:
				aux_meters[k].update(aux[k].detach().item(), target.size(0))
		else:
			output = model(input_var1, input_var2)
			loss = criterion(output, target)

		prec1, _ = utils.accuracy(output, target, topk=(1, 3))
		losses.update(loss.item(), target.size(0))
		top1.update(prec1[0], target.size(0))

		optimizer.zero_grad()
		loss.backward()
		if opt.clip_grad > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if episode_index % opt.print_freq == 0 and episode_index != 0:
			msg = ('Epoch-({0}): [{1}/{2}]\t'
				   'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
				   'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
				   'Prec@1 {top1.val:.2f} ({top1.avg:.2f})').format(
					   epoch_index, episode_index, len(train_loader),
					   batch_time=batch_time, loss=losses, top1=top1)
			if opt.classifier_model.startswith('NDCN'):
				msg += '\t[tri {:.3f} bg {:.3f} div {:.3f} mask {:.3f}]'.format(
					aux_meters['triplet'].avg, aux_meters['bg'].avg,
					aux_meters['div'].avg, aux_meters['mask'].avg)
			print(msg)
			print(msg, file=F_txt); F_txt.flush()

	return top1.avg, losses.avg


# ============================== validate ==================================
def validate(val_loader, model, criterion, epoch_index, best_prec1, F_txt, tag='Val'):
	batch_time, losses, top1 = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
	model.eval()
	accuracies = []

	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):
		input_var1 = torch.cat(query_images, 0).to(device)
		input_var2 = torch.cat(support_images, 0).squeeze(0).to(device)
		input_var2 = input_var2.contiguous().view(-1, input_var2.size(2),
												 input_var2.size(3), input_var2.size(4))
		target = torch.cat(query_targets, 0).to(device)

		output = model(input_var1, input_var2)          # eval mode -> logits only
		loss = criterion(output, target)

		prec1, _ = utils.accuracy(output, target, topk=(1, 3))
		losses.update(loss.item(), target.size(0))
		top1.update(prec1[0], target.size(0))
		accuracies.append(prec1)

		batch_time.update(time.time() - end)
		end = time.time()

		if episode_index % opt.print_freq == 0 and episode_index != 0:
			msg = ('{tag}-({0}): [{1}/{2}]\t'
				   'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
				   'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
				   'Prec@1 {top1.val:.2f} ({top1.avg:.2f})').format(
					   epoch_index, episode_index, len(val_loader), tag=tag,
					   batch_time=batch_time, loss=losses, top1=top1)
			print(msg)
			print(msg, file=F_txt)

	print(' * Prec@1 {top1.avg:.2f} Best_prec1 {best_prec1:.2f}'.format(top1=top1, best_prec1=best_prec1))
	print(' * Prec@1 {top1.avg:.2f} Best_prec1 {best_prec1:.2f}'.format(top1=top1, best_prec1=best_prec1), file=F_txt)
	return top1.avg, losses.avg, accuracies


def log_scalars(model, F_txt):
	"""Print the three learnable logit scalars (Table: learned three-way weights)."""
	m = model.module if hasattr(model, 'module') else model
	if not m.classifier_model.startswith('NDCN'):
		return
	c = m.classifier
	line = ('  [logit scalars] gamma_fg = {:.3f}   gamma_bg = {:.3f}   beta_bg = {:+.3f}'
			.format(c.gamma_fg.detach().item(), c.gamma_bg.detach().item(), c.beta_bg.detach().item()))
	diag = getattr(c, '_diag', {})
	if diag:
		line += ('\n  [diagnostics]   logit_spread = {:.3f}  mask_mean(q) = {:.3f}  '
				 'mask_mean(s) = {:.3f}  alpha>0.7 = {:.2f}  proto_offdiag_cos = {:.3f}').format(
			float(diag.get('logit_spread', 0.)), float(diag['mask_mean_q']),
			float(diag['mask_mean_s']), float(diag['alpha_hi']), float(diag['proto_offdiag']))
		# head-specific diagnostics, printed only when the head produces them
		extra = [(k, diag[k]) for k in ('match_conc', 'ot_cost', 'plan_conc', 'recon_lam', 'residual') if k in diag]
		if extra:
			line += '\n  [head]          ' + '  '.join('%s = %.4f' % (k, float(v)) for k, v in extra)
	print(line)
	print(line, file=F_txt)


if __name__ == '__main__':
	init_modes = [bool(opt.pretrained), bool(opt.warm_from), bool(opt.resume)]
	if sum(init_modes) > 1:
		raise ValueError('use only one of --pretrained, --warm_from, and --resume')
	if opt.cuda and not torch.cuda.is_available():
		raise RuntimeError('--cuda was requested but CUDA is not available')

	opt.outf, F_txt = utils.set_save_path(opt)

	global best_prec1
	best_prec1 = 0
	model = FewShotNet.define_model(
		encoder_model=opt.encoder_model, classifier_model=opt.classifier_model,
		norm='batch', way_num=opt.way_num, shot_num=opt.shot_num,
		init_type='normal', use_gpu=opt.cuda,
		ndcn_cfg=build_ndcn_cfg(opt) if opt.classifier_model.startswith('NDCN') else None)
	if opt.ngpu > 1:
		if not opt.cuda:
			raise ValueError('--ngpu > 1 requires CUDA')
		model = nn.DataParallel(model, range(opt.ngpu))

	criterion = nn.CrossEntropyLoss().to(device)
	# Make the weight-decay rule identical for Adam and SGD. The submitted
	# version silently hard-coded Adam's decay to 5e-4 and ignored both flags.
	if opt.wd_scalars >= 0:
		scal, rest = [], []
		for n, prm in model.named_parameters():
			(scal if n.endswith(('_raw_gamma_fg', '_raw_gamma_bg', 'beta_bg')) else rest).append(prm)
		param_groups = [
			{'params': rest, 'weight_decay': opt.weight_decay},
			{'params': scal, 'weight_decay': opt.wd_scalars},
		]
	else:
		scal = []
		param_groups = [{'params': model.parameters(), 'weight_decay': opt.weight_decay}]

	if opt.adam:
		optimizer = optim.Adam(param_groups, lr=opt.lr, betas=(opt.beta1, 0.9))
		print("Using Adam Optimizer (weight_decay=%g, scalar decay=%s)"
			  % (opt.weight_decay, opt.wd_scalars if opt.wd_scalars >= 0 else 'same'))
	else:
		optimizer = optim.SGD(param_groups, lr=opt.lr, momentum=opt.momentum,
							  dampening=opt.momentum)
		print("Using SGD Optimizer (weight_decay=%g, scalar decay=%s)"
			  % (opt.weight_decay, opt.wd_scalars if opt.wd_scalars >= 0 else 'same'))

	if opt.pretrained:
		pre = torch.load(opt.pretrained, map_location='cpu', weights_only=False)
		if pre.get('encoder_model') not in (None, opt.encoder_model):
			raise ValueError('pre-trained backbone is %s but --encoder_model is %s'
							 % (pre['encoder_model'], opt.encoder_model))
		if pre.get('data_name') not in (None, opt.data_name):
			raise ValueError('pre-trained on %s but training on %s; base-class '
							 'pre-training must stay within the same dataset'
							 % (pre['data_name'], opt.data_name))
		if 'backbone' not in pre:
			raise KeyError('%s does not contain a backbone state dict' % opt.pretrained)
		target = model.module.features if hasattr(model, 'module') else model.features
		target.load_state_dict(pre['backbone'], strict=True)
		msg = ('=> backbone initialised from %s (pre-train epoch %s, episodic val '
			   '%.2f); strict key match passed'
			   % (opt.pretrained, pre.get('epoch', '?'),
				  float(pre.get('episodic_val', 0.0))))
		print(msg); print(msg, file=F_txt)

	if opt.warm_from:
		ck = torch.load(opt.warm_from, map_location='cpu', weights_only=False)
		model.load_state_dict(ck['model'])
		msg = ('=> warm restart from %s (was epoch %s, best_val %.2f); running a fresh '
			   '%d-epoch cosine schedule at lr=%g'
			   % (opt.warm_from, ck.get('epoch_index', '?'),
				  float(ck.get('best_prec1', 0)), opt.epochs, opt.lr))
		print(msg); print(msg, file=F_txt)

	resume_checkpoint = None
	if opt.resume:
		checkpoint = utils.get_resume_file(opt.resume, F_txt)
		resume_checkpoint = checkpoint
		opt.start_epoch = int(checkpoint['epoch_index']) + 1
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	print(opt); print(opt, file=F_txt)
	print(model); print(model, file=F_txt)
	with open(os.path.join(opt.outf, 'config.json'), 'w') as f:
		json.dump(vars(opt), f, indent=2, default=str)

	scheduler = None
	if opt.cosine:
		eta_min = opt.lr * (opt.lr_decay_rate ** 3)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)
		if resume_checkpoint is not None:
			if resume_checkpoint.get('scheduler') is not None:
				scheduler.load_state_dict(resume_checkpoint['scheduler'])
			else:
				print('WARNING: old resume checkpoint has no scheduler state; LR phase restarts')

	print('===================================== Training =====================================')
	print('===================================== Training =====================================', file=F_txt)

	Train_losses, Val_losses, Test_losses = [], [], []
	best_epoch, test_at_best = -1, 0.0
	for epoch_item in range(opt.start_epoch, opt.epochs):
		print('==================== Epoch %d ====================' % epoch_item)
		print('==================== Epoch %d ====================' % epoch_item, file=F_txt)

		opt.current_epoch = epoch_item
		modes = ['train', 'val', 'test'] if opt.monitor_test else ['train', 'val']
		loaders = FewShotDataloader.get_Fewshot_dataloader(opt, modes)
		train_loader, val_loader = loaders[:2]
		test_loader = loaders[2] if opt.monitor_test else None

		prec1_train, train_loss = train(train_loader, model, criterion, optimizer, epoch_item, F_txt)
		Train_losses.append(train_loss)
		log_scalars(model, F_txt)

		with torch.no_grad():
			prec1_val, val_loss, _ = validate(val_loader, model, criterion, epoch_item,
											  best_prec1, F_txt, tag='Val')
		Val_losses.append(val_loss)

		if opt.monitor_test:
			with torch.no_grad():
				prec1_test, test_loss, _ = validate(test_loader, model, criterion, epoch_item,
												best_prec1, F_txt, tag='Test')
			Test_losses.append(test_loss)
		else:
			prec1_test = float('nan')

		if opt.cosine:
			scheduler.step()
		else:
			utils.adjust_learning_rate(opt, optimizer, epoch_item, F_txt)

		# Crash guard. A single bad batch can collapse the prototypes onto each
		# other; on CUB 5-shot validation fell 73.8 -> 30.7 in one epoch and had
		# not recovered six epochs later. Roll back rather than burn the run.
		if opt.crash_tol > 0 and best_prec1 > 0 and prec1_val < opt.crash_tol * best_prec1:
			warn = ('  [crash guard]   val {:.2f} < {:.0f}% of best {:.2f} -- reloading '
					'model_best and continuing'.format(float(prec1_val),
					100 * opt.crash_tol, float(best_prec1)))
			print(warn); print(warn, file=F_txt)
			ck = os.path.join(opt.outf, 'model_best.pth.tar')
			if os.path.isfile(ck):
				sd = torch.load(ck, map_location='cpu', weights_only=False)
				model.load_state_dict(sd['model'])
				optimizer.load_state_dict(sd['optimizer'])
				model.to(device)

		is_best = prec1_val > best_prec1
		if is_best:
			best_epoch, test_at_best = epoch_item, prec1_test
			best_prec1 = max(prec1_val, best_prec1)
			if opt.monitor_test:
				sel = ('  [selection]     best_val = {:.2f} at epoch {}   '
					   'test at that epoch = {:.2f}'.format(
						   float(best_prec1), best_epoch, float(test_at_best)))
			else:
				sel = ('  [selection]     best_val = {:.2f} at epoch {}; test split '
					   'not evaluated'.format(float(best_prec1), best_epoch))
		print(sel); print(sel, file=F_txt)

		# Python buffers text writes in 8 KB chunks, so without this the log
		# trails the run by several epochs and everything still in the buffer
		# is lost if the job is killed. Also dump the loss curves every epoch
		# instead of only at the end, for the same reason.
		F_txt.flush(); os.fsync(F_txt.fileno())
		for name, arr in (('train_loss', Train_losses), ('val_loss', Val_losses),
						  ('test_loss', Test_losses)):
			np.savetxt(os.path.join(opt.outf, name + '.txt'), np.array(arr))

		state = {'epoch_index': epoch_item,
				 'encoder_model': opt.encoder_model,
				 'data_name': opt.data_name,
				 'way_num': opt.way_num,
				 'shot_num': opt.shot_num,
				 'query_num': opt.query_num,
				 'classifier_model': opt.classifier_model,
				 'ndcn_cfg': build_ndcn_cfg(opt) if opt.classifier_model in ('NDCN', 'NDCN_Local') else None,
				 'model': model.state_dict(),
				 'best_prec1': best_prec1,
				 'optimizer': optimizer.state_dict(),
				 'scheduler': scheduler.state_dict() if scheduler is not None else None,
				 'initialization': ('pretrained:' + opt.pretrained if opt.pretrained else
									'warm_from:' + opt.warm_from if opt.warm_from else
									'resume:' + opt.resume if opt.resume else 'scratch')}
		if is_best:
			utils.save_checkpoint(state, os.path.join(opt.outf, 'model_best.pth.tar'))
		utils.save_checkpoint(state, os.path.join(opt.outf, 'model_last.pth.tar'))
		if epoch_item % 10 == 0:
			utils.save_checkpoint(state, os.path.join(opt.outf, 'epoch_%d.pth.tar' % epoch_item))

	utils.plot_loss_curve(opt, Train_losses, Val_losses, Test_losses)
	print('======================================== Training is END ========================================\n')
	print('======================================== Training is END ========================================\n', file=F_txt)
	F_txt.close()

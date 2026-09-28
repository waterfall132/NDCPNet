import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
import inspect
import random
import pdb
import copy
import sys
sys.dont_write_bytecode = True


# ============================ Backbone & Classifier ===============================
import models.backbone as backbone
import models.backbone_relu as backbone_relu
import models.classifier as classifier
from models.ndcn import NDCN_Metric
from models.ndcn_local import NDCN_Local_Metric
try:
	from models.ndcn_recon import NDCN_Recon_Metric
except ImportError:              # reconstruction head is optional
	NDCN_Recon_Metric = None
try:
	from models.ndcn_comp import NDCN_Comp_Metric
except ImportError:              # component-matching head is optional
	NDCN_Comp_Metric = None
try:
	from models.ndcn_ot import NDCN_OT_Metric
except ImportError:              # optimal-transport head is optional
	NDCN_OT_Metric = None
# ==================================================================================
'''
	All models consist of two parts: backbone module and classifier module.
'''


###############################################################################
# Functions
###############################################################################

encoder_dict = dict(
			Conv64F             = backbone.Conv64F,
			Conv64F_Local       = backbone.Conv64F_Local,
			Conv64F_Local_ReLU  = backbone_relu.Conv64F_Local_ReLU,   # NDCN
			ResNet10            = backbone.ResNet10,
			ResNet12            = backbone.ResNet12,
			ResNet12_ReLU       = backbone_relu.ResNet12_ReLU,        # NDCN
			SeResNet12          = backbone.SeResNet12,
			SeResNet12_ReLU     = backbone_relu.SeResNet12_ReLU,      # NDCN
			ResNet18            = backbone.ResNet18,
			ResNet34            = backbone.ResNet34,
			ResNet50            = backbone.ResNet50,
			ResNet101           = backbone.ResNet101)


classifier_dict = dict(
			ProtoNet      = classifier.Prototype_Metric,
			DN4           = classifier.ImgtoClass_Metric,
			NDCN          = NDCN_Metric,
			NDCN_Local    = NDCN_Local_Metric,
			NDCN_Recon    = NDCN_Recon_Metric,
			NDCN_Comp     = NDCN_Comp_Metric,
			NDCN_OT       = NDCN_OT_Metric)
classifier_dict = {k: v for k, v in classifier_dict.items() if v is not None}


def _filter_kwargs(cls, cfg):
	"""Drop entries cls.__init__ does not accept (walking the MRO for **kwargs)."""
	accepted, k = set(), cls
	for base in inspect.getmro(cls):
		init = base.__dict__.get('__init__')
		if init is None:
			continue
		accepted |= set(inspect.signature(init).parameters)
	dropped = [k for k in cfg if k not in accepted]
	if dropped:
		print('  [network] %s ignores: %s' % (cls.__name__, ', '.join(sorted(dropped))))
	return {k: v for k, v in cfg.items() if k in accepted}


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def print_network(net):
	num_params = sum(p.numel() for p in net.parameters())
	head_params = sum(p.numel() for n, p in net.named_parameters() if n.startswith('classifier'))
	print('Total number of parameters: %d  (classifier head: %d)' % (num_params, head_params))


def define_model(pretrained=False, model_root=None, encoder_model='Conv64F',
				 classifier_model='DN4', norm='batch', init_type='normal',
				 use_gpu=True, **kwargs):
	model = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if classifier_model in ['ProtoNet', 'DN4', 'NDCN', 'NDCN_Local', 'NDCN_Recon', 'NDCN_Comp', 'NDCN_OT']:
		model = Fewshot_model(encoder_model=encoder_model,
							  classifier_model=classifier_model, **kwargs)
	else:
		raise NotImplementedError('Model name [%s] is not recognized' % classifier_model)

	print_network(model)

	if use_gpu:
		model.cuda()

	if pretrained:
		model.load_state_dict(model_root)

	return model


class Fewshot_model(nn.Module):
	'''
		Define a few-shot learning model, which consists of an embedding module
		and a classifier module.

		For classifier_model == 'NDCN' the forward signature accepts an optional
		`target`; during training it returns (logits, aux_dict).
	'''
	def __init__(self, encoder_model='Conv64F', classifier_model='DN4', class_num=64,
				 way_num=5, shot_num=5, query_num=10, neighbor_k=3, ndcn_cfg=None):
		super(Fewshot_model, self).__init__()
		self.encoder_model = encoder_model
		self.classifier_model = classifier_model
		self.way_num = way_num
		self.shot_num = shot_num
		self.query_num = query_num
		self.neighbor_k = neighbor_k
		self.loss_type = 'softmax'

		if encoder_model in ['Conv64F', 'Conv64F_Local', 'Conv64F_Local_ReLU']:
			self.feature_dim = 64
		elif encoder_model in ['ResNet10', 'ResNet18', 'ResNet34']:
			self.feature_dim = 512
		elif encoder_model in ['ResNet12', 'ResNet12_ReLU', 'SeResNet12', 'SeResNet12_ReLU']:
			self.feature_dim = 640
		elif encoder_model in ['ResNet50', 'ResNet101']:
			self.feature_dim = 2048

		encoder_module    = encoder_dict[self.encoder_model]
		classifier_module = classifier_dict[self.classifier_model]

		self.features = encoder_module()

		if self.classifier_model.startswith('NDCN'):
			# Keep only the keys this particular head actually accepts, so a
			# training script that passes options for every head still works
			# with a head that does not know about them.
			cfg = _filter_kwargs(classifier_module, dict(ndcn_cfg or {}))
			self.classifier = classifier_module(way_num=way_num, shot_num=shot_num, **cfg)
		else:
			self.classifier = classifier_module(way_num=way_num, shot_num=shot_num,
												neighbor_k=neighbor_k)

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)

	def forward(self, input1, input2, target=None, is_feature=False):
		x1 = self.features(input1)      # query
		x2 = self.features(input2)      # support set

		if self.classifier_model.startswith('NDCN'):
			out = self.classifier(x1, x2, target)
		else:
			out = self.classifier(x1, x2)

		if is_feature:
			return x1, x2, out
		return out


class Model_with_reused_Encoder(nn.Module):
	'''
		Construct a new few-shot model by reusing a pre-trained embedding module.
	'''
	def __init__(self, pre_trained_model, new_classifier='DN4', way_num=5, shot_num=5,
				 neighbor_k=3, ndcn_cfg=None):
		super(Model_with_reused_Encoder, self).__init__()
		self.way_num = way_num
		self.shot_num = shot_num
		self.neighbor_k = neighbor_k
		self.model = pre_trained_model
		self.classifier_model = new_classifier

		self.features = nn.Sequential(*list(self.model.features.children()))

		classifier_module = classifier_dict[new_classifier]
		if new_classifier.startswith('NDCN'):
			self.classifier = classifier_module(way_num=way_num, shot_num=shot_num,
												**_filter_kwargs(classifier_module,
																 dict(ndcn_cfg or {})))
		else:
			self.classifier = classifier_module(way_num=way_num, shot_num=shot_num,
												neighbor_k=neighbor_k)

	def forward(self, input1, input2, target=None):
		x1 = self.features(input1)
		x2 = self.features(input2)
		if self.classifier_model == 'NDCN':
			return self.classifier(x1, x2, target)
		return self.classifier(x1, x2)

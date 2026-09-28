#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReLU-terminated variants of the DN4 backbones.

Why this file exists
    NDM factorises Z >= 0. In the original DN4 code:
        Conv64F_Local  -> nn.LeakyReLU(0.2)   (min value  -0.2 * |x|)
        ResNet12       -> nn.LeakyReLU(0.1)   (min value  -0.1 * |x|)
    so roughly 30-45 % of descriptor entries are negative and the
    non-negativity property does not hold. These variants swap the final
    activation for ReLU, which makes the assumption exact.

    Use --nonneg_mode relu with the original backbones for a quick run, and
    these backbones for the numbers you report.
"""

import torch
import torch.nn as nn
from torch.nn import init

import models.backbone as backbone


# ---------------------------------------------------------------- Conv64F ---
class Conv64F_Local_ReLU(nn.Module):
    """Conv64F_Local with ReLU instead of LeakyReLU.  3x84x84 -> 64x21x21"""

    def __init__(self):
        super(Conv64F_Local_ReLU, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                        # 64x42x42
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                        # 64x21x21
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.features(x)


# --------------------------------------------------------------- ResNet12 ---
def _relu_patch(module):
    """Recursively replace every LeakyReLU with ReLU (in place)."""
    for name, child in module.named_children():
        if isinstance(child, nn.LeakyReLU):
            setattr(module, name, nn.ReLU(inplace=child.inplace))
        else:
            _relu_patch(child)
    return module


def ResNet12_ReLU(**kwargs):
    """ResNet12 whose BasicBlock activations are ReLU (output is non-negative)."""
    net = backbone.ResNet12(**kwargs)
    return _relu_patch(net)


def SeResNet12_ReLU(**kwargs):
    net = backbone.SeResNet12(**kwargs)
    return _relu_patch(net)


# ------------------------------------------------------------- diagnostics ---
@torch.no_grad()
def negative_fraction(feat):
    """Fraction of strictly negative entries in a feature tensor."""
    return (feat < 0).float().mean().item()

"""
PSPnet class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from ..encoders.squeeze_extractor import *

class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, pool_factors=(1, 2, 3, 6), batch_norm=True):
        super().__init__()
        self.spatial_blocks = []
        for pf in pool_factors:
            self.spatial_blocks += [self._make_spatial_block(in_channels, pf, batch_norm)]
        self.spatial_blocks = nn.ModuleList(self.spatial_blocks)

        bottleneck = []
        bottleneck += [nn.Conv2d(in_channels * (len(pool_factors) + 1), out_channels, kernel_size=1)]
        if batch_norm:
            bottleneck += [nn.BatchNorm2d(out_channels)]
        bottleneck += [nn.ReLU(inplace=True)]
        self.bottleneck = nn.Sequential(*bottleneck)

    def _make_spatial_block(self, in_channels, pool_factor, batch_norm):
        spatial_block = []
        spatial_block += [nn.AdaptiveAvgPool2d(output_size=(pool_factor, pool_factor))]
        spatial_block += [nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)]
        if batch_norm:
            spatial_block += [nn.BatchNorm2d(in_channels)]
        spatial_block += [nn.ReLU(inplace=True)]

        return nn.Sequential(*spatial_block)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool_outs = [x]
        for block in self.spatial_blocks:
            pooled = block(x)
            pool_outs += [F.upsample(pooled, size=(h, w), mode='bilinear')]
        o = torch.cat(pool_outs, dim=1)
        o = self.bottleneck(o)
        return o

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class PSPUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.ReLU(inplace=True)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(x, size=(h, w), mode='bilinear')
        return self.layer(p)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class PSPnet(torch.nn.Module):

    def __init__(self, n_classes, pretrained_model: SqueezeExtractor, batch_norm=True, psp_out_feature=1024):
        super(PSPnet, self).__init__()
        self.features = pretrained_model.features

        # find out_channels of the top layer and define classifier
        for idx, m in reversed(list(enumerate(self.features.modules()))):
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels
                break

        self.PSP = PSPModule(channels, out_channels=psp_out_feature, batch_norm=batch_norm)
        h_psp_out_feature = int(psp_out_feature / 2)
        q_psp_out_feature = int(psp_out_feature / 4)
        e_psp_out_feature = int(psp_out_feature / 8)
        self.upsampling1 = PSPUpsampling(psp_out_feature, h_psp_out_feature, batch_norm=batch_norm)
        self.upsampling2 = PSPUpsampling(h_psp_out_feature, q_psp_out_feature, batch_norm=batch_norm)
        self.upsampling3 = PSPUpsampling(q_psp_out_feature,  e_psp_out_feature, batch_norm=batch_norm)

        self.classifier = nn.Sequential(nn.Conv2d(e_psp_out_feature, n_classes, kernel_size=1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = x
        for f in self.features:
            o = f(o)
            
        o = self.PSP(o)
        o = self.upsampling1(o)
        o = self.upsampling2(o)
        o = self.upsampling3(o)

        o = F.upsample(o, size=(x.shape[2], x.shape[3]), mode='bilinear')
        o = self.classifier(o)

        return o

from ..encoders.vgg import *
from ..encoders.resnet import *
from ..encoders.mobilenet import *

def pspnet_vgg11(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_11(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)
def pspnet_vgg13(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_13(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)
def pspnet_vgg16(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_16(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)
def pspnet_vgg19(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_19(batch_norm, pretrained, fixed_feature)
    copy_feature_info = vgg.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index - 1
    vgg.features = vgg.features[:squeeze_feature_idx]
    return PSPnet(n_classes, vgg, batch_norm)

def pspnet_resnet18(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet18(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet34(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet34(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet50(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet50(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet101(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet101(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)
def pspnet_resnet152(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet152(pretrained, fixed_feature)
    copy_feature_info = resnet.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    resnet.features = resnet.features[:squeeze_feature_idx]
    return PSPnet(n_classes, resnet, batch_norm)

def pspnet_mobilenet_v2(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    mobile_net = mobilenet(pretrained, fixed_feature)
    copy_feature_info = mobile_net.get_copy_feature_info()
    squeeze_feature_idx = copy_feature_info[3].index
    mobile_net.features = mobile_net.features[:squeeze_feature_idx]
    return PSPnet(n_classes, mobile_net, batch_norm)


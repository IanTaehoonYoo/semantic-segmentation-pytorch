"""
Unet class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

import torch
from ..encoders.squeeze_extractor import *

class UnetWithEncoder(torch.nn.Module):

    def __init__(self, n_classes, pretrained_model: SqueezeExtractor, batch_norm=True):
        super(UnetWithEncoder, self).__init__()
        self.copy_feature_info = pretrained_model.get_copy_feature_info()
        self.features = pretrained_model.features

        self.up_layer0 = self._make_up_layer(-1, batch_norm)
        self.up_layer1 = self._make_up_layer(-2, batch_norm)
        self.up_layer2 = self._make_up_layer(-3, batch_norm)
        self.up_layer3 = self._make_up_layer(-4, batch_norm)

        self.up_sampling0 = self._make_up_sampling(-1)
        self.up_sampling1 = self._make_up_sampling(-2)
        self.up_sampling2 = self._make_up_sampling(-3)
        self.up_sampling3 = self._make_up_sampling(-4)

        #find out_channels of the top layer and define classifier
        for f in reversed(self.up_layer3):
            if isinstance(f, nn.Conv2d):
                channels = f.out_channels
                break

        uplayer4 = []
        uplayer4 += [nn.Conv2d(channels, channels, kernel_size=3, padding=1)]
        if batch_norm:
            uplayer4 += [nn.BatchNorm2d(channels)]
        uplayer4 += [nn.ReLU(inplace=True)]
        self.up_layer4 = nn.Sequential(*uplayer4)

        self.up_sampling4 = nn.ConvTranspose2d(channels, channels, kernel_size=4,
                                              stride=2, bias=False)
        self.classifier = nn.Sequential(nn.Conv2d(channels, n_classes, kernel_size=1), nn.ReLU(inplace=True))
        self._initialize_weights()

    def _get_last_out_channels(self, features):
        for idx, m in reversed(list(enumerate(features.modules()))):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        return 0


    def _make_up_sampling(self, cfi_idx):
        if cfi_idx == -1:
            in_channels = self._get_last_out_channels(self.features)
        else:
            in_channels = self.copy_feature_info[cfi_idx + 1].out_channels

        out_channels = self.copy_feature_info[cfi_idx].out_channels
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                  stride=2, bias=False)

    def _make_up_layer(self, cfi_idx, batch_norm):
        idx = self.copy_feature_info[cfi_idx].index
        for k in reversed(range(0, idx)):
            f = self.features[k]
            channels = self._get_last_out_channels(f)

            if channels == 0:
                continue

            out_channels = self.copy_feature_info[cfi_idx].out_channels
            in_channels = out_channels + channels  # for concatenation.

            layer = []
            layer += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            if batch_norm:
                layer += [nn.BatchNorm2d(out_channels)]
            layer += [nn.ReLU(inplace=True)]

            return nn.Sequential(*layer)

        assert False

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
        copy_out = []
        o = x
        cpi = self.copy_feature_info[-4:]
        copy_idx = 0

        for i in range(len(self.features)):
            o = self.features[i](o)
            if i == cpi[copy_idx].index - 1:
                copy_out.append(o)
                if copy_idx + 1 < len(cpi):
                    copy_idx += 1

        o = self.up_sampling0(o)
        o = o[:, :, 1:1 + copy_out[3].size()[2], 1:1 + copy_out[3].size()[3]]
        o = torch.cat([o, copy_out[3]], dim=1)
        o = self.up_layer0(o)

        o = self.up_sampling1(o)
        o = o[:, :, 1:1 + copy_out[2].size()[2], 1:1 + copy_out[2].size()[3]]
        o = torch.cat([o, copy_out[2]], dim=1)
        o = self.up_layer1(o)

        o = self.up_sampling2(o)
        o = o[:, :, 1:1 + copy_out[1].size()[2], 1:1 + copy_out[1].size()[3]]
        o = torch.cat([o, copy_out[1]], dim=1)
        o = self.up_layer2(o)

        o = self.up_sampling3(o)
        o = o[:, :, 1:1 + copy_out[0].size()[2], 1:1 + copy_out[0].size()[3]]
        o = torch.cat([o, copy_out[0]], dim=1)
        o = self.up_layer3(o)

        o = self.up_sampling4(o)
        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]
        o = self.up_layer4(o)
        o = self.classifier(o)

        return o

cfgs = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 'U', 512, 512,
           'U', 256, 256, 'U', 128, 128, 'U', 64, 64]
}

class Unet(torch.nn.Module):

    def __init__(self, n_classes, cfg, batch_norm=True):
        super(Unet, self).__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.classifier = nn.Sequential(nn.Conv2d(cfg[-1], n_classes, kernel_size=1), nn.Sigmoid())

        self._initialize_weights()


    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'U':
                layers += [nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=4,
                                               stride=2, bias=False)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = x
        size_features = len(self.features)
        copys = []
        for i in range(size_features):
            o = self.features[i](o)
            if isinstance(self.features[i], nn.ConvTranspose2d):
                copy = copys.pop()
                o = o[:, :, 1:1 + copy.size()[2], 1:1 + copy.size()[3]]
                o = torch.cat([o, copy], dim=1)
            if i + 1 >= size_features:
                continue
            if isinstance(self.features[i+1], nn.MaxPool2d):
                copys += [o]

        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]
        o = self.classifier(o)

        return o

from ..encoders.vgg import *
from ..encoders.resnet import *
from ..encoders.mobilenet import *

def unet_vgg11(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_11(batch_norm, pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, vgg, batch_norm)
def unet_vgg13(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_13(batch_norm, pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, vgg, batch_norm)
def unet_vgg16(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_16(batch_norm, pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, vgg, batch_norm)
def unet_vgg19(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_19(batch_norm, pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, vgg, batch_norm)

def unet_resnet18(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet18(pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, resnet, batch_norm)
def unet_resnet34(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet34(pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, resnet, batch_norm)
def unet_resnet50(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet50(pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, resnet, batch_norm)
def unet_resnet101(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet101(pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, resnet, batch_norm)
def unet_resnet152(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    resnet = resnet152(pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, resnet, batch_norm)

def unet_mobilenet_v2(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    mobile_net = mobilenet(pretrained, fixed_feature)
    return UnetWithEncoder(n_classes, mobile_net, batch_norm)

def unet(n_classes, batch_size):
    batch_norm = False if batch_size == 1 else True
    return Unet(n_classes, cfgs['A'], batch_norm)


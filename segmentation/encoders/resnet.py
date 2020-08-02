"""
Resnet model Customized from Torchvision.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division
from .squeeze_extractor import *


class _ResNet(SqueezeExtractor):
    def __init__(self, model, fixed_feature=True):
        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        super(_ResNet, self).__init__(model, features, fixed_feature)

    def get_copy_feature_info(self):
        lst_copy_feature_info = []
        channel = 0
        for i in range(len(self.features)):
            feature = self.features[i]
            if isinstance(feature, nn.MaxPool2d):
                lst_copy_feature_info.append(CopyFeatureInfo(i, channel))
            for idx, m in enumerate(feature.modules()):
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    channel = self._get_last_conv2d_out_channels(feature)
                    lst_copy_feature_info.append(CopyFeatureInfo(i, channel))
                    break

        return lst_copy_feature_info



def resnet18(pretrained=False, fixed_feature=True):
    """ "ResNet-18 model from torchvision's resnet model.

    :param pretrained: if true, return a model pretrained on ImageNet
    :param fixed_feature: if true and pretrained is true, model features are fixed while training.
    """
    from torchvision.models.resnet import resnet18
    model = resnet18(pretrained)

    ff = True if pretrained and fixed_feature else False
    return _ResNet(model, ff)

def resnet34(pretrained=False, fixed_feature=True):
    """ "ResNet-34 model from torchvision's resnet model.

    :param pretrained: if true, return a model pretrained on ImageNet
    :param fixed_feature: if true and pretrained is true, model features are fixed while training.
    """
    from torchvision.models.resnet import resnet34
    model = resnet34(pretrained)

    ff = True if pretrained and fixed_feature else False
    return _ResNet(model, ff)

def resnet50(pretrained=False, fixed_feature=True):
    """ "ResNet-50 model from torchvision's resnet model.

    :param pretrained: if true, return a model pretrained on ImageNet
    :param fixed_feature: if true and pretrained is true, model features are fixed while training.
    """
    from torchvision.models.resnet import resnet50
    model = resnet50(pretrained)

    ff = True if pretrained and fixed_feature else False
    return _ResNet(model, ff)

def resnet101(pretrained=False, fixed_feature=True):
    """ "ResNet-101 model from torchvision's resnet model.

    :param pretrained: if true, return a model pretrained on ImageNet
    :param fixed_feature: if true and pretrained is true, model features are fixed while training.
    """
    from torchvision.models.resnet import resnet101
    model = resnet101(pretrained)

    ff = True if pretrained and fixed_feature else False
    return _ResNet(model, ff)

def resnet152(pretrained=False, fixed_feature=True):
    """ "ResNet-152 model from torchvision's resnet model.

    :param pretrained: if true, return a model pretrained on ImageNet
    :param fixed_feature: if true and pretrained is true, model features are fixed while training.
    """
    from torchvision.models.resnet import resnet152
    model = resnet152(pretrained)

    ff = True if pretrained and fixed_feature else False
    return _ResNet(model, ff)


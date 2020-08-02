"""
Mobilenet model Customized from Torchvision.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division
from .squeeze_extractor import *
from torch import nn

class _Mobilenet(SqueezeExtractor):
	def __init__(self, model, features, fixed_feature=True):
		layer = []
		layers = []
		self.zip_factor = [2, 4, 7, 11, 14, 17, 18]
		layer, layers = self._get_layers(features, layer, layers, 0)
		layers = nn.ModuleList(layers)
		super(_Mobilenet, self).__init__(model, layers, fixed_feature)

	def _get_layers(self, features, layer, layers, zip_cnt):
		from torchvision.models.mobilenet import InvertedResidual, ConvBNReLU

		for feature in features.children():
			if isinstance(feature, nn.Sequential) or\
					isinstance(feature, ConvBNReLU):
				layer, layers = self._get_layers(feature, layer, layers, zip_cnt)
				if zip_cnt == 0 and isinstance(feature, ConvBNReLU):
					layers += [nn.Sequential(*layer)]
					layer.clear()
			if isinstance(feature, InvertedResidual):
				layer, layers = self._get_layers(feature, layer, layers, zip_cnt)
				zip_cnt += 1
				if zip_cnt in self.zip_factor:
					layers += [nn.Sequential(*layer)]
					layer.clear()

			if len(list(feature.children())) == 0:
				layer += [feature]

		return layer, layers
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


def mobilenet(pretrained=False, fixed_feature=True):
	""" Mobile-net V2 model from torchvision's resnet model.

	:param pretrained: if true, return a model pretrained on ImageNet
	:param fixed_feature: if true and pretrained is true, model features are fixed while training.
	"""
	from torchvision.models.mobilenet import mobilenet_v2
	model = mobilenet_v2(pretrained)
	features = model.features[:-1]

	ff = True if pretrained and fixed_feature else False
	return _Mobilenet(model, features, ff)

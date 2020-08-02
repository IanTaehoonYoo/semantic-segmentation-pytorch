from __future__ import absolute_import, division, print_function

from .fcn8 import *
from .fcn16 import *
from .fcn32 import *
from .unet import *
from .pspnet import *

model_from_name = {}

model_from_name["fcn8_vgg11"] = fcn8_vgg11
model_from_name["fcn8_vgg13"] = fcn8_vgg13
model_from_name["fcn8_vgg16"] = fcn8_vgg16
model_from_name["fcn8_vgg19"] = fcn8_vgg19
model_from_name["fcn16_vgg11"] = fcn16_vgg11
model_from_name["fcn16_vgg13"] = fcn16_vgg13
model_from_name["fcn16_vgg16"] = fcn16_vgg16
model_from_name["fcn16_vgg19"] = fcn16_vgg19
model_from_name["fcn32_vgg11"] = fcn32_vgg11
model_from_name["fcn32_vgg13"] = fcn32_vgg13
model_from_name["fcn32_vgg16"] = fcn32_vgg16
model_from_name["fcn32_vgg19"] = fcn32_vgg19
model_from_name["fcn8_resnet18"] = fcn8_resnet18
model_from_name["fcn8_resnet34"] = fcn8_resnet34
model_from_name["fcn8_resnet50"] = fcn8_resnet50
model_from_name["fcn8_resnet101"] = fcn8_resnet101
model_from_name["fcn8_resnet152"] = fcn8_resnet152
model_from_name["fcn16_resnet18"] = fcn16_resnet18
model_from_name["fcn16_resnet34"] = fcn16_resnet34
model_from_name["fcn16_resnet50"] = fcn16_resnet50
model_from_name["fcn16_resnet101"] = fcn16_resnet101
model_from_name["fcn16_resnet152"] = fcn16_resnet152
model_from_name["fcn32_resnet18"] = fcn32_resnet18
model_from_name["fcn32_resnet34"] = fcn32_resnet34
model_from_name["fcn32_resnet50"] = fcn32_resnet50
model_from_name["fcn32_resnet101"] = fcn32_resnet101
model_from_name["fcn32_resnet152"] = fcn32_resnet152
model_from_name["fcn8_mobilenet_v2"] = fcn8_mobilenet_v2
model_from_name["fcn16_mobilenet_v2"] = fcn16_mobilenet_v2
model_from_name["fcn32_mobilenet_v2"] = fcn32_mobilenet_v2

model_from_name["unet"] = unet
model_from_name["unet_vgg11"] = unet_vgg11
model_from_name["unet_vgg13"] = unet_vgg13
model_from_name["unet_vgg16"] = unet_vgg16
model_from_name["unet_vgg19"] = unet_vgg19
model_from_name["unet_resnet18"] = unet_resnet18
model_from_name["unet_resnet34"] = unet_resnet34
model_from_name["unet_resnet50"] = unet_resnet50
model_from_name["unet_resnet101"] = unet_resnet101
model_from_name["unet_resnet152"] = unet_resnet152
model_from_name["unet_mobilenet_v2"] = unet_mobilenet_v2

model_from_name["pspnet_vgg11"] = pspnet_vgg11
model_from_name["pspnet_vgg13"] = pspnet_vgg13
model_from_name["pspnet_vgg16"] = pspnet_vgg16
model_from_name["pspnet_vgg19"] = pspnet_vgg19
model_from_name["pspnet_resnet18"] = pspnet_resnet18
model_from_name["pspnet_resnet34"] = pspnet_resnet34
model_from_name["pspnet_resnet50"] = pspnet_resnet50
model_from_name["pspnet_resnet101"] = pspnet_resnet101
model_from_name["pspnet_resnet152"] = pspnet_resnet152
model_from_name["pspnet_mobilenet_v2"] = pspnet_mobilenet_v2





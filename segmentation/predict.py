"""
The predict functions.
The main function is to write output on the color from the gray labeled image.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1, OpenCV-Python 4.1.1.26
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

import random
import cv2
import torch
import numpy as np
import os
import pathlib
import six

def parent(path):
	path = pathlib.Path(path)
	return str(path.parent)

def exist(path):
	return os.path.exists(str(path))

def mkdir(path):
	pathlib.Path(path).mkdir(parents=True, exist_ok=True)

random.seed(0)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

def convert_seg_gray_to_color(input, n_classes, output_path=None, colors=class_colors):
	"""
	Convert the segmented image on gray to color.

	:param input: it is available to get two type(ndarray, string), string type is a file path.
	:param n_classes: number of the classes.
	:param output_path: output path. if it is None, this function return result array(ndarray)
	:param colors: refer to 'class_colors' format. Default: random assigned color.
	:return: if out_path is None, return result array(ndarray)
	"""
	if isinstance(input, six.string_types):
		seg = cv2.imread(input, flags=cv2.IMREAD_GRAYSCALE)
	elif type(input) is np.ndarray:
		assert len(input.shape) == 2, "Input should be h,w "
		seg = input

	height = seg.shape[0]
	width = seg.shape[1]

	seg_img = np.zeros((height, width, 3))

	for c in range(n_classes):
		seg_arr = seg[:, :] == c
		seg_img[:, :, 0] += ((seg_arr) * colors[c][0]).astype('uint8')
		seg_img[:, :, 1] += ((seg_arr) * colors[c][1]).astype('uint8')
		seg_img[:, :, 2] += ((seg_arr) * colors[c][2]).astype('uint8')

	if output_path:
		cv2.imwrite(output_path, seg_img)
	else:
		return seg_img

def predict(model, input_path, output_path, colors=class_colors):
	"""
	This function can save a predicted result on the color from the trained model.

	:param model: a network model.
	:param input_path: the input file path.
	:param output_path: the output file path.
	:param colors: refer to 'class_colors' format. Default: random assigned color.
	:return: model result.
	"""
	model.eval()

	img = cv2.imread(input_path, flags=cv2.IMREAD_COLOR)
	ori_height = img.shape[0]
	ori_width = img.shape[1]

	model_width = model.img_width
	model_height = model.img_height

	if model_width != ori_width or model_height != ori_height:
		img = cv2.resize(img, (model_width, model_height), interpolation=cv2.INTER_NEAREST)


	data = img.transpose((2, 0, 1))
	data = data[None, :, :, :]
	data = torch.from_numpy(data).float()

	if next(model.parameters()).is_cuda:
		if not torch.cuda.is_available():
			raise ValueError("A model was trained via .cuda(), but this system can not support cuda.")
		data = data.cuda()

	score = model(data)

	lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
	lbl_pred = lbl_pred.transpose((1, 2, 0))
	n_classes = np.max(lbl_pred)
	lbl_pred = lbl_pred.reshape(model_height, model_width)

	seg_img = convert_seg_gray_to_color(lbl_pred, n_classes, colors=colors)

	if model_width != ori_width or model_height != ori_height:
		seg_img = cv2.resize(seg_img, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)

	if not exist(parent(output_path)):
		mkdir(parent(output_path))

	cv2.imwrite(output_path, seg_img)

	return score
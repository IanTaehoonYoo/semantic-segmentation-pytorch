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

random.seed(0)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

def predict(model, input_name, output_name, colors=class_colors):
	model.eval()

	img = cv2.imread(input_name, flags=cv2.IMREAD_COLOR)
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

	seg_img = np.zeros((model_height, model_width, 3))

	for c in range(n_classes):
		seg_arr = lbl_pred[:, :] == c
		seg_arr = seg_arr.reshape(model_height, model_width)
		seg_img[:, :, 0] += ((seg_arr)* colors[c][0]).astype('uint8')
		seg_img[:, :, 1] += ((seg_arr) * colors[c][1]).astype('uint8')
		seg_img[:, :, 2] += ((seg_arr) * colors[c][2]).astype('uint8')


	if model_width != ori_width or model_height != ori_height:
		seg_img = cv2.resize(seg_img, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)

	cv2.imwrite(output_name, seg_img)

	return score
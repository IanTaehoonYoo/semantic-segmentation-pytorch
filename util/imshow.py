"""
imshow function. This is useful to see an tiled image.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1, OpenCV-Python 4.1.1.26, PIL
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import os
if os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"') != 0: # non gui support
	print("non gui system. use Agg instead")
	import matplotlib # https://stackoverflow.com/questions/43003758/
	matplotlib.use("Agg") # matplotlib-is-throwing-segmentation-fault-when-running-on-non-gui-machineweb-se
import matplotlib.pyplot as plt
from PIL import Image
import pathlib

def parent(path):
	path = pathlib.Path(path)
	return str(path.parent)

def exist(path):
	return os.path.exists(str(path))

def mkdir(path):
	pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def normalize_img(img, vmin=None, vmax=None):
	"""
	:param img:	Tensor, np
	:param vmin:
	:param vmax:
	:return:		Tensor, np, float32, return the same dimension
	"""
	if isinstance(img, np.ndarray):
		img = img.astype(np.float32)
		if vmin is None:
			vmin = img.min()
		if vmax is None:
			vmax = img.max()
		img = np.clip(img, vmin, vmax)
		img = (img - vmin) / (vmax - vmin)
		img = np.clip(img, 0, 1)  # numeric error 대비
		return img
	elif isinstance(img, torch.Tensor):
		img = img.type(torch.float32)
		if vmin is None:
			vmin = img.min()
		if vmax is None:
			vmax = img.max()
		img = torch.clamp(img, vmin, vmax)
		img = (img - vmin) / (vmax - vmin)
		img = torch.clamp(img, 0, 1)
		return img
	else:
		raise ValueError

def imshow(*args, nx=None, vmin=None, vmax=None, path=None, is_color=False, normalize_uint8=False, title=None):
	"""
		:param args: available as follow
			Tensor:
				float32, bcyx | cyx
			np:
				uint8, byxc | yxc
				float32, byxc | yxc
			PIL image
		:param nx:					image counts on cols.
									(default) if counts bigger than ten, nx is fixed ten.
		:param vmin:				(default) minimum value of the image.
		:param vmax:				(default) maximum value of the image.
		:param path:				saved path. if it is null, not save.
		:param is_color:			gray or color.
		:param normalize_uint8:	    if it is [np.uint8], this param decides to divide 255.
		:param title:	    		firgure name
	"""
	# args is turned into one list.

	imgs = []
	for arg in args:
		if isinstance(arg, list):
			imgs += arg
		else:
			imgs.append(arg)

	assert len(imgs) >= 1

	for i in range(len(imgs)):
		if isinstance(imgs[i], np.ndarray):
			# if normalize_uint8 is true, values is normalized [0,1]
			if normalize_uint8 and imgs[i].dtype == np.uint8:
				imgs[i] = np.clip(imgs[i].astype(np.float32) / 255, 0, 1)
			else:
				imgs[i] = imgs[i].astype(np.float32)
			if len(imgs[i].shape) == 2:  # np, yx --> 1yx1
				imgs[i] = imgs[i][None, :, :, None]
			elif len(imgs[i].shape) == 3:  # np, yxc --> 1yxc
				imgs[i] = imgs[i][None, :, :, :]
			elif len(imgs[i].shape) == 4:
				pass
			else:
				raise ValueError
		elif isinstance(imgs[i], torch.Tensor):
			imgs[i] = imgs[i].cpu().detach().numpy()
			imgs[i] = imgs[i].astype(np.float32)
			if len(imgs[i].shape) == 2:  # Tensor, yx --> 1yx1
				imgs[i] = imgs[i][None, :, :, None]
			elif len(imgs[i].shape) == 3:  # Tensor, cyx --> 1yxc
				imgs[i] = imgs[i][None, :, :, :]
				imgs[i] = np.transpose(imgs[i], [0, 2, 3, 1])
			elif len(imgs[i].shape) == 4:
				imgs[i] = np.transpose(imgs[i], [0, 2, 3, 1])
			else:
				raise ValueError
		elif isinstance(imgs[i], Image.Image):
			imgs[i] = np.array(imgs[i]).astype(np.float32)
			imgs[i] = np.clip(imgs[i] / 255, 0, 1)
			if len(imgs[i].shape) == 2:  # PIL img, yx --> 1yx1
				imgs[i] = imgs[i][None, :, :, None]
			elif len(imgs[i].shape) == 3:  # PIL img, yxc --> 1yxc
				imgs[i] = imgs[i][None, :, :, :3]  # if it has alpha, it is trimmed.
			else:
				raise ValueError
		else:
			raise ValueError

	# imgs's byxc must be matched
	img = np.concatenate(imgs)

	# check color
	b, y, x, c = img.shape
	if is_color and c != 3 and c != 4:
		raise ValueError

	# np, bcyx, float32
	img = np.transpose(img, [0, 3, 1, 2])

	# set nx automatically from image counts
	num_img = img.shape[0] if is_color else img.shape[0] * img.shape[1]
	if nx is None:
		if num_img < 10:
			nx = num_img
		else:
			nx = 10

	if not is_color:  # gray
		ny = int(np.ceil(np.float32(b * c) / nx))
		img = img.reshape(b * c, y, x)
		black = np.zeros([ny * nx - b * c, y, x], np.float32)
		img = np.concatenate([img, black])  # ny*nx,y,x
		img = img.reshape(ny, nx, y, x)
		img = img.transpose(0, 2, 1, 3)  # ny,y,nx,x
		img = img.reshape(ny * y, nx * x)  # ny*y,nx*x complete
	else:  # color: if img is color image, it'll create three channels of the gray image.
		ny = int(np.ceil(np.float32(b) / nx))
		black = np.zeros([ny * nx - b, c, y, x], np.float32)
		img = np.concatenate([img, black])  # ny*nx,c,y,x
		img = img.reshape(ny, nx, c, y, x)
		img = img.transpose(0, 3, 1, 4, 2)  # ny,y,nx,x,c
		img = img.reshape(ny* y, nx * x, c)  # ny*y,nx*x,c complete

	# rescale
	img = normalize_img(img, vmin, vmax)

	# plot or save:
	if path == None:
		plt.figure(num=title)
		if is_color:
			fig = plt.imshow(img, interpolation="nearest", vmin=0, vmax=1)
		else:
			fig = plt.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.pause(0.001)
		plt.show(block=False)
		plt.tight_layout()
	else:
		if not exist(parent(path)):
			mkdir(parent(path))
		img = np.clip(img * 255, 0, 255).astype(np.uint8)
		img = Image.fromarray(img)
		img.save(path)


def imshowc(*args, nx=None, vmin=0, vmax=1, path=None, normalize_uint8=True, title=None):
	"""
		Drawing a color image.

		:param args: available as follow
			Tensor:
				float32, bcyx | cyx
			np:
				uint8, bcyx | cyx
				float32, bcyx | cyx
			PIL image
		:param nx:					image counts on cols.
									(default) if counts bigger than ten, nx is fixed ten.
		:param vmin:				(default) minimum value of the image.
		:param vmax:				(default) maximum value of the image.
		:param path:				saved path. if it is null, not save.
		:param is_color:			gray or color.
		:param normalize_uint8:	    if it is [np.uint8], this param decides to divide 255.
		:param title:	    		firgure name
	"""
	imshow(*args, nx=nx, vmin=vmin, vmax=vmax, path=path, is_color=True, normalize_uint8=normalize_uint8, title=title)

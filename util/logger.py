"""
The logger class. This class write logs to Tensorboard mostly.
Also, it can save images and will be stored in '.runs' path

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import pathlib
import datetime
from util.checkpoint import *
import copy

class Logger:

	def __init__(self, model_name, data_name):
		self.model_name = model_name
		self.data_name = data_name

		self.comment = '{}_{}'.format(model_name, data_name)
		self.data_subdir = '{}/{}'.format(model_name, data_name)

		current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")

		train_log_dir = r'runs/' + current_time + self.comment + r'/train'
		test_log_dir = r'runs/' + current_time + self.comment + r'/test'

		self.hdl_chkpoint = CheckpointHandler()

		# TensorBoard
		self.writer_train = SummaryWriter(train_log_dir, comment=self.comment)
		self.writer_test = SummaryWriter(test_log_dir, comment=self.comment)

	def log_train(self, scalar, title, epoch, n_batch, num_batches):
		step = Logger._step(epoch, n_batch, num_batches)
		self.writer_train.add_scalar(
			'{}/{}'.format(self.comment, title), scalar, step)

	def log_test(self, scalar, title, epoch, n_batch, num_batches):
		step = Logger._step(epoch, n_batch, num_batches)
		self.writer_test.add_scalar(
			'{}/{}'.format(self.comment, title), scalar, step)

	def concatenate_images(self, *args, input_axis='bcyx', normalize_uint8=False):
		"""
		This function concatenate images and return the result.

		:param images: 			available follow
								Tensor: float32
								np: uint8, int64, float32
		:param input_axis: 		if the input_axis is 'byxc', it transpose axis to 'bcyx'.
								available (bcyx | byxc | cyx | yxc | byx)
		:param normalize_uint8:	if dtype is [np.uint8], values are divided by 255.
		:return: images
		"""
		imgs = []
		for arg in args:
			if isinstance(arg, list):
				imgs += arg
			else:
				imgs.append(arg)

		assert len(imgs) >= 1

		for i in range(len(imgs)):
			if isinstance(imgs[i], np.ndarray):
				if imgs[i].dtype == np.uint8 or imgs[i].dtype == np.int64:
					if normalize_uint8:
						imgs[i] = np.clip(imgs[i].astype(np.float32) / 255, 0.0, 1.0)
					else:
						imgs[i] = imgs[i].astype(np.float32)
				imgs[i] = torch.from_numpy(imgs[i])

			if imgs[i].dtype != torch.float32:
				imgs[i] = imgs[i].float()

			if len(imgs[i].shape) == 2:  # Tensor, yx --> 11yx
				imgs[i] = imgs[i][None, None, :, :]
			elif len(imgs[i].shape) == 3:  # Tensor, cyx --> 1cyx
				imgs[i] = imgs[i][None, :, :, :]

			# swap axis to 'bcyx'
			if input_axis == 'byxc' or input_axis == 'yxc':
				imgs[i] = imgs[i].transpose(1, 3)
				imgs[i] = imgs[i].transpose(1, 2)
			elif input_axis == 'byx':
				imgs[i] = imgs[i].transpose(0, 1)
				
			if imgs[i].shape[1] == 1:
				imgs[i] = imgs[i].repeat((1, 3, 1, 1)) / 3.0

		return torch.cat(imgs)

	def log_images_train(self, images, epoch, n_batch, num_batches, input_axis='bcyx',
				   nrows=8, padding=2, pad_value=1, normalize=True, normalize_uint8=False):
		"""
		This function writes images to Tensorboard and save the file at [./data/images/]

		:param images: 			available follow
								Tensor: float32
								np: uint8, int64, float32
		:param epoch: 			epoch.
		:param n_batch:			batch index.
		:param num_batches: 	batch counts.
		:param nrows: 			grid's rows on the image.
		:param padding: 		amount of padding.
		:param pad_value: 		padding scalar value, the range [0, 1].
		:param input_axis: 		if the input_axis is 'byxc', it transpose axis to 'bcyx', available as follow
								(bcyx | byxc | cyx | yxc | byx)
		:param normalize: 		normalize image to the range [0, 1].
		:param normalize_uint8:	if dtype is [np.uint8], values are divided by 255.
		"""

		img_name, grid, step = self._log_images(images, epoch, n_batch, num_batches,
												input_axis, nrows, padding, pad_value, normalize,
						 						comment='train')

		# Add images to tensorboard
		self.writer_train.add_image(img_name, grid, step)

	def log_images_test(self, images, epoch, n_batch, num_batches, input_axis='bcyx',
				   nrows=8, padding=2, pad_value=1, normalize=True, normalize_uint8=False):
		"""
		This function writes images to Tensorboard and save the file at [./data/images/]

		:param images: 			available follow
								Tensor: float32
								np: uint8, int64, float32
		:param epoch: 			epoch.
		:param n_batch:			batch index.
		:param num_batches: 	batch counts.
		:param nrows: 			grid's rows on the image.
		:param padding: 		amount of padding.
		:param pad_value: 		padding scalar value, the range [0, 1].
		:param input_axis: 		if the input_axis is 'byxc', it transpose axis to 'bcyx', available as follow
								(bcyx | byxc | cyx | yxc | byx)
		:param normalize: 		normalize image to the range [0, 1].
		:param normalize_uint8:	if dtype is [np.uint8], values are divided by 255.
		"""

		img_name, grid, step = self._log_images(images, epoch, n_batch, num_batches,
												input_axis, nrows, padding, pad_value, normalize,
						 						comment='test')

		# Add images to tensorboard
		self.writer_test.add_image(img_name, grid, step)

	def _log_images(self, images, epoch, n_batch, num_batches, input_axis='bcyx',
				   nrows=8, padding=2, pad_value=1, normalize=True, normalize_uint8=False, comment=''):

		if isinstance(images, np.ndarray):
			if images.dtype == np.uint8 or images.dtype == np.int64:
				if normalize_uint8:
					images = np.clip(images.astype(np.float32) / 255, 0.0, 1.0)
				else:
					images = images.astype(np.float32)
			images = torch.from_numpy(images)

		if len(images.shape) == 2:  # Tensor, yx --> 11yx
			images = images[None, None, :, :]
		elif len(images.shape) == 3:  # Tensor, cyx --> 1cyx
			images = images[None, :, :, :]

		# swap axis to 'bcyx'
		if input_axis == 'byxc' or input_axis == 'yxc':
			images = images.transpose(1, 3)
			images = images.transpose(1, 2)
		elif input_axis == 'byx':
			images = images.transpose(0, 1)

		step = Logger._step(epoch, n_batch, num_batches)
		img_name = '{}/images{}'.format(self.comment, '')

		# Make grid from image tensor
		if images.shape[0] < nrows:
			nrows = images.shape[0]

		grid = vutils.make_grid(images, nrow=nrows, normalize=normalize,
								scale_each=True, pad_value=pad_value, padding=padding)

		# Save plots
		self._save_torch_images(grid, epoch, n_batch, comment)

		return img_name, grid, step

	def _save_torch_images(self, grid, epoch, n_batch, comment=''):
		out_dir = './runs/images/{}'.format(self.data_subdir)
		Logger._make_dir(out_dir)

		# Save squared
		fig = plt.figure()
		plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
		plt.axis('off')
		if comment:
			fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, comment, epoch, n_batch))
		else:
			fig.savefig('{}/epoch_{}_batch_{}.png'.format(out_dir, comment, epoch, n_batch))
		plt.close()

	def store_checkpoint_var(self, key, value):
		self.hdl_chkpoint.store_var(key, value)

	def save_model(self, model, file_name):
		out_dir = './runs/models/{}'.format(self.data_subdir)
		if not Logger._exist(out_dir):
			Logger._make_dir(out_dir)

		self.hdl_chkpoint.save_checkpoint('{}/{}'.format(out_dir, file_name))

	def save_model_and_optimizer(self, model, optim, file_name):
		out_dir = './runs/models/{}'.format(self.data_subdir)
		if not Logger._exist(out_dir):
			Logger._make_dir(out_dir)

		self.hdl_chkpoint.save_checkpoint('{}/{}'.format(out_dir, file_name), model, optim)

	def load_model(self, model, file_name):
		dir = './runs/models/{}'.format(self.data_subdir)
		assert Logger._exist(dir)

		self.hdl_chkpoint = self.hdl_chkpoint.load_checkpoint('{}/{}'.format(dir, file_name))

		model.load_state_dict(self.hdl_chkpoint.model_state_dict)
		if hasattr(self.hdl_chkpoint, '__dict__'):
			for k in self.hdl_chkpoint.__dict__:
				if k == 'model_state_dict' or k == 'optimizer_state_dict':
					continue
				attr_copy = copy.deepcopy(getattr(self.hdl_chkpoint, k))
				setattr(model, k, attr_copy)

	# def load_model_and_optimizer(self, model, optim, file_name):
	# 	dir = './runs/models/{}'.format(self.data_subdir)
	# 	assert Logger._exist(dir)
	#
	# 	self.hdl_chkpoint = self.hdl_chkpoint.load_checkpoint('{}/{}'.format(dir, file_name))
	#
	# 	model.load_state_dict(self.hdl_chkpoint.model_state_dict)
	# 	optim.load_state_dict(self.hdl_chkpoint.optimizer_state_dict)
	# 	if hasattr(self.hdl_chkpoint, '__dict__'):
	# 		for k in self.hdl_chkpoint.__dict__:
	# 			if k == 'model_state_dict' or k == 'optimizer_state_dict':
	# 				continue
	# 			attr_copy = copy.deepcopy(getattr(self.hdl_chkpoint, k))
	# 			setattr(model, k, attr_copy)

	def close(self):
		self.writer.close()

	@staticmethod
	def _step(epoch, n_batch, num_batches):
		return epoch * num_batches + n_batch

	@staticmethod
	def _make_dir(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

	@staticmethod
	def _parent(path):
		path = pathlib.Path(path)
		return str(path.parent)

	@staticmethod
	def _exist(path):
		return os.path.exists(str(path))

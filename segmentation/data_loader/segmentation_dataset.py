"""
Dataset class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1, OpenCV-Python 4.1.1.26
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, print_function, division
import os
import numpy as np
import time
import torch
from torch.utils.data import Dataset
import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DataLoaderError(Exception):
    pass

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter

TQDM_COLS = 80

class SegmentationDataset(Dataset):
	""" Segmentation dataset"""
	def __init__(self, images_dir, segs_dir, n_classes, transform=None):
		"""
		input images must be matched.

		:param images_dir: path to the image directory
		:param segs_dir: path to the annotation image directory
		:param n_classes: a number of the classes
		:param transform: optional transform to be applied on an image
		"""
		super(SegmentationDataset, self).__init__()

		self.images_dir = images_dir
		self.segs_dir = segs_dir
		self.transform = transform
		self.n_classes = n_classes

		self.pairs_dir = self._get_image_pairs_(self.images_dir, self.segs_dir)
		verified = self._verify_segmentation_dataset()
		assert verified

	def __len__(self):
		return len(self.pairs_dir)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		im = cv2.imread(self.pairs_dir[idx][0], flags=cv2.IMREAD_COLOR)
		lbl = cv2.imread(self.pairs_dir[idx][1], flags=cv2.IMREAD_GRAYSCALE)

		sample = {'image': im, 'labeled': lbl}

		if self.transform:
			sample = self.transform(sample)

		return sample
	def _verify_segmentation_dataset(self):
		try:
			if not len(self.pairs_dir):
				print("Couldn't load any data from self.images_dir: "
					  "{0} and segmentations path: {1}"
					  .format(self.images_dir, self.segs_dir))
				return False

			return_value = True
			for im_fn, seg_fn in tqdm(self.pairs_dir, ncols=TQDM_COLS):
				img = cv2.imread(im_fn)
				seg = cv2.imread(seg_fn)
				# Check dimensions match
				if not img.shape == seg.shape:
					return_value = False
					print("The size of image {0} and its segmentation {1} "
						  "doesn't match (possibly the files are corrupt)."
						  .format(im_fn, seg_fn))
				else:
					max_pixel_value = np.max(seg[:, :, 0])
					if max_pixel_value >= self.n_classes:
						return_value = False
						print("The pixel values of the segmentation image {0} "
							  "violating range [0, {1}]. "
							  "Found maximum pixel value {2}"
							  .format(seg_fn, str(self.n_classes - 1), max_pixel_value))

			time.sleep(0.0001)
			if return_value:
				print("Dataset verified! ")
			else:
				print("Dataset not verified!")
			return return_value
		except DataLoaderError as e:
			print("Found error during data loading\n{0}".format(str(e)))
			return False

	def _get_image_pairs_(self, img_path1, img_path2):
		""" Check two images have the same name and get all the images
		:param img_path1: directory
		:param img_path2: directory
		:return: pair paths
		"""

		AVAILABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

		files1 = []
		files2 = {}
		
		for dir_entry in os.listdir(img_path1):
			if os.path.isfile(os.path.join(img_path1, dir_entry)) and \
					os.path.splitext(dir_entry)[1] in AVAILABLE_IMAGE_FORMATS:
				file_name, file_extension = os.path.splitext(dir_entry)
				files1.append((file_name, file_extension,
									os.path.join(img_path1, dir_entry)))

		for dir_entry in os.listdir(img_path2):
			if os.path.isfile(os.path.join(img_path2, dir_entry)) and \
				os.path.splitext(dir_entry)[1] in AVAILABLE_IMAGE_FORMATS:
				file_name, file_extension = os.path.splitext(dir_entry)
				full_dir_entry = os.path.join(img_path2, dir_entry)
				if file_name in files2:
					raise DataLoaderError("img_path2 with filename {0}"
										  " already exists and is ambiguous to"
										  " resolve with path {1}."
										  " Please remove or rename the latter."
										  .format(file_name, full_dir_entry))

				files2[file_name] = (file_extension, full_dir_entry)

		return_value = []
		# Match two paths
		for image_file, _, image_full_path in files1:
			if image_file in files2:
				return_value.append((image_full_path,
									 files2[image_file][1]))
			else:
				# Error out
				raise DataLoaderError("No corresponding images "
									  "found for image {0}."
									  .format(image_full_path))

		return return_value
			



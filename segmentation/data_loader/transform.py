"""
The transform method for the SegmentationDataset

Library:	Tensowflow 2.2.0, pyTorch 1.5.1, OpenCV-Python 4.1.1.26
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import torch
import cv2

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labeled = sample['image'], sample['labeled']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        lbl = cv2.resize(labeled, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return {'image': img, 'labeled': lbl}

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        image, labeled = sample['image'], sample['labeled']

        if torch.rand(1) < self.p:
            image = cv2.flip(image, 1)
            labeled = cv2.flip(labeled, 1)

        return {'image': image, 'labeled': labeled}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class MakeSegmentationArray(object):
    """Make segmentation array from the annotation image"""

    def __init__(self, n_classes):
        assert isinstance(n_classes, int)

        self.n_classes = n_classes

    def __call__(self, sample):
        annotation = sample['annotation']
        assert annotation.dtype != int

        h, w = annotation.shape[:2]

        seg_labels = np.zeros((self.n_classes, h, w), dtype=annotation.dtype)

        for label in range(self.n_classes):
            seg_labels[label, :, :] = (annotation == label)

        return {'image': sample['image'], 'annotation': seg_labels}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, lbl = sample['image'], sample['labeled']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'annotation': torch.from_numpy(lbl).long()}
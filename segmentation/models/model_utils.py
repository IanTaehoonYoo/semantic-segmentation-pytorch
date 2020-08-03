"""
Model util class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""

def set_segmentation_model_params(model, width, height):
	model.img_width = width
	model.img_height = height

	return model

"""
CheckpointHandler.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""

import torch

class CheckpointHandler:

    def store_var(self, var_name, value, exist_fail=False):
        if exist_fail is True and hasattr(self, var_name):
            raise Exception("var_name='{}' already exists".format(var_name))
        else:
            setattr(self, var_name, value)

    def get_var(self, var_name):
        if hasattr(self, var_name):
            value = getattr(self, var_name)
            return value
        else:
            return False

    def save_checkpoint(self, checkpoint_path, model, optimizer=None):
        if type(model) == torch.nn.DataParallel:
            # converting a DataParallel model to be able load later without DataParallel
            self.model_state_dict = model.module.state_dict()
        else:
            self.model_state_dict = model.state_dict()

        if optimizer:
            self.optimizer_state_dict = optimizer.state_dict()

        torch.save(self, checkpoint_path)

    @staticmethod
    def load_checkpoint(checkpoint_path, map_location='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return checkpoint
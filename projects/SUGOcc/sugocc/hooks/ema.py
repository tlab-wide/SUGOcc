# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
import torch
import torch.nn as nn
from copy import deepcopy
import stat
from typing import Dict, Optional

from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from mmengine.runner import load_state_dict
from mmengine.dist import master_only
from mmengine.registry import HOOKS
from mmengine.hooks.hook import DATA_BATCH, Hook

def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)

__all__ = ['ModelEMA']

class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in the model
    state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.shadow = {}
        self.backup = {}
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        self.model = model
        self.ema_state = {}
        state_dict = model.module.module.state_dict(keep_vars=True) if is_parallel(
            model.module) else model.module.state_dict(keep_vars=True)
        module = model.module.module if is_parallel(model.module) else model.module
        for k, v in state_dict.items():
            self.ema_state[k] = v.detach().clone()
            # if v.dtype.is_floating_point:
            #     self.ema_state[k] = v.detach().clone()

    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema_state.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

@HOOKS.register_module()
class MEGVIIEMAHook(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, init_updates=0, decay=0.9990, resume=None):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay

    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        
        runner.ema_model = ModelEMA(runner.model, self.decay)
        
        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema_state, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']

    def after_train_iter(self, 
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None):
        runner.ema_model.update(runner, runner.model.module)

    def after_train_epoch(self, runner):
        # if self.is_last_epoch(runner):   # 只保存最后一个epoch的ema权重.
        self.save_checkpoint(runner)

    def before_val_epoch(self, runner):
        state_dict = runner.model.module.state_dict() if is_parallel(
            runner.model) else runner.model.state_dict()
        self.backup_state = {k: v.detach().clone() for k, v in state_dict.items()}
        
        ema_state_dict = runner.ema_model.ema_state
        for k, v in state_dict.items():
            if k in ema_state_dict:
                v.copy_(ema_state_dict[k])
    
    def after_val_epoch(self, runner, metrics = None):
        if self.backup_state is None:
            return
        state_dict = runner.model.module.state_dict() if is_parallel(
            runner.model) else runner.model.state_dict()
        for k, v in state_dict.items():
            if k in self.backup_state:
                v.copy_(self.backup_state[k])
        self.backup_state = None

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema_state
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')

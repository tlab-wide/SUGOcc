from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import spconv.pytorch as spconv
import numpy as np
import time
from einops import rearrange
# from mmcv.runner import BaseModule
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
# from mmdet3d.models.builder import MODELS, NECKS
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
# from projects.HSOcc.hrocc.models.modules.sparse_block import sparse_cat, SparseConvBlock, SparseBasicBlock
# from projects.HSOcc.hrocc.models.modules.sparse_block import SparseBasicBlock
# from spconv.pytorch import functional as fsp
from .mink import *
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
    
class DenseNet(BaseModule):
    def __init__(self, 
                 in_channels=128,
                 num_blocks=9,
                 out_channels=192,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(
                MSCABlock(dim=in_channels)
            )
        self.init_weights()
    
    def forward(self, x, voxel_shape=[16,16,2]):

        tensor_stride = x.tensor_stride[0]
        coordinate_manager = x.coordinate_manager
        x = x.dense(
            shape=torch.Size([(x.C[:,0].max()+1).int().item(), x.features.shape[1], 
                              voxel_shape[0], 
                              voxel_shape[1], 
                              voxel_shape[2]]),
            min_coordinate=torch.IntTensor([0, 0, 0]),
        )[0]

        for layer in self.layers:
            x = layer(x)

        temp_x = ME.to_sparse(x)
        coords = temp_x.C.clone()
        coords[:, 1:] = coords[:, 1:] * tensor_stride

        x = ME.SparseTensor(
            features=temp_x.features,
            coordinates=coords,
            tensor_stride=tensor_stride,
            coordinate_manager=coordinate_manager
        )
        return x

@MODELS.register_module()
class MinkowskiUNetEncoder(BaseModule):
    def __init__(self, 
                 in_channels=[128, 256, 512, 1024],
                 out_channels=[128, 256, 512, 1024],
                 num_blocks=[2,2,2,2],
                 voxel_size=[128,128,16],
                 num_dense_blocks=3,
                 kernel_size=3,
                 cross_kernel=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_size = voxel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s1 = nn.Sequential(
            nn.Identity(),
            *[ResidualBlock(in_channels[0], in_channels[0], ks=kernel_size,
                           cross_kernel=cross_kernel, expand_coords=False) for _ in range(num_blocks[0])]
        )
        self.s1s2 = nn.Sequential(
            BasicConvolutionBlock(in_channels[0], in_channels[1], ks=2, stride=2),
            *[ResidualBlock(in_channels[1], in_channels[1], ks=kernel_size, 
                           cross_kernel=cross_kernel) for _ in range(num_blocks[1])]
        )
        
        self.s2s4 = nn.Sequential(
            BasicConvolutionBlock(in_channels[1], in_channels[2], ks=2, stride=2),
            *[ResidualBlock(in_channels[2], in_channels[2], ks=kernel_size,
                           cross_kernel=cross_kernel) for _ in range(num_blocks[2])]
        )

        self.s4s8 = nn.Sequential(
            BasicConvolutionBlock(in_channels[2], in_channels[3], ks=2, stride=2),
            *[ResidualBlock(in_channels[3], in_channels[3], ks=kernel_size,
                           cross_kernel=cross_kernel) for _ in range(num_blocks[3])]
        )
        self.voxel_size = voxel_size
        self.dense_net = DenseNet(
            in_channels=in_channels[3],
            num_blocks=num_dense_blocks,
            out_channels=in_channels[3],
        )
        self.init_weights()
    
    def forward(self, x):
        _s = time.time()
        if not isinstance(x, ME.SparseTensor):
            B, C, X, Y, Z = x.shape
            x = ME.to_sparse(x)
        else:
            B = x.C[:,0].max().int().item() + 1
            C = x.F.shape[1]
            X, Y, Z = self.voxel_size

        x1 = self.s1(x)
        x2 = self.s1s2(x1)
        x3 = self.s2s4(x2)
        x4 = self.s4s8(x3)

        x4 = self.dense_net(x4, voxel_shape=[X//8, Y//8, Z//8])
        return [x1, x2, x3, x4]
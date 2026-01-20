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
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from projects.SUGOcc.sugocc.models.backbones.mink import *
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from projects.SUGOcc.sugocc.utils.semkitti import semantic_kitti_class_frequencies
from projects.SUGOcc.sugocc.utils.nusc import nusc_class_frequencies
from projects.SUGOcc.sugocc.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.SUGOcc.sugocc.utils.lovasz_softmax import lovasz_softmax

class MinkowskiUpAndPruneBlock(BaseModule):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 pruning_ratio=0.3,
                 process_block_num=1,
                 process_kernel_size=1,
                 process_cross_kernel=False,
                 nclasses=20,
                 empty_idx=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pruning_ratio = pruning_ratio
        self.empty_idx = empty_idx
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(
            nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiLeakyReLU(inplace=True),
            ))
        self.cls_layers = nn.ModuleList()
        self.cls_layers.append(
            ME.MinkowskiConvolution(
                in_channels=out_channels,
                out_channels=nclasses,
                kernel_size=1,
                stride=1,
                bias=True,
                dimension=3,))
        
        self.process = nn.Sequential(
            *[ResidualBlock(out_channels, out_channels, 
                        ks=process_kernel_size, drop_path=0.2, cross_kernel=process_cross_kernel) for _ in range(process_block_num)]
        )
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x, short_cut):
        seg = None
        x = self.upsample_layers[0](x)
            
        x = self.process(x + short_cut)

        mask = torch.zeros((x.F.shape[0],), dtype=torch.bool).to(x.F.device)
        
        seg = self.cls_layers[0](x)
        seg_prob = F.softmax(seg.F, dim=-1)

        mask = mask | ((1 - seg_prob[:,self.empty_idx]) > self.pruning_ratio)
        if mask.sum() == 0:
            mask = torch.ones_like(mask).bool().to(x.device)
        x = self.pruning(x, mask)
        return x, None, seg
    
@MODELS.register_module()
class SparseGenerativePixelDecoder(BaseModule):
    """Generative Pixel Decoder for Sparse 3D Features

    Args:
        in_channels (list[int]): Input channels of each scale feature.
        out_channels (int): Output channels of the decoder.
        norm_cfg (dict): Config dict for normalization layer.
        upsample_cfg (dict): Config dict for upsampling layer.
        init_cfg (dict, optional): Init config for `BaseModule`. Defaults to None.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 process_block_num=1,
                 process_kernel_size=1,
                 nclasses=20,
                 process_cross_kernel=False,
                 pruning_ratio=[0.5, 0.5, 0.5],
                 empty_idx=0,
                 lss_downsample=[2,2,2],
                 dataset='semantickitti',
                 init_cfg=None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.empty_idx = empty_idx
        self.lss_downsample = lss_downsample
        self.upsample_blocks = nn.ModuleList()
        for i, in_channel in enumerate(self.in_channels):
            if i == len(self.in_channels) - 1:
                continue
            up_layer = MinkowskiUpAndPruneBlock(
                in_channels=self.in_channels[len(self.in_channels)-1-i],
                out_channels=self.in_channels[len(self.in_channels)-2-i],
                pruning_ratio=pruning_ratio[i],
                process_block_num=process_block_num,
                process_kernel_size=process_kernel_size,
                process_cross_kernel=process_cross_kernel,
                empty_idx = self.empty_idx,
                nclasses = nclasses,
            )
            self.upsample_blocks.append(up_layer)

        self.out_convs = nn.ModuleList()
        for i in range(len(out_channels)):
            if in_channels[i] != out_channels[i]:
                self.out_convs.append(
                    nn.Sequential(
                        ME.MinkowskiConvolution(
                            in_channels[i], 
                            out_channels[i], 
                            kernel_size=1, 
                            stride=1,
                            dimension=3,),
                        ME.MinkowskiBatchNorm(out_channels[i]),
                        ME.MinkowskiLeakyReLU(inplace=True), 
                    )
                )
            else:
                self.out_convs.append(None)

        if dataset == 'semantickitti':
            kitti_class_weights = 1 / np.log(semantic_kitti_class_frequencies)
            norm_kitti_class_weights = kitti_class_weights / kitti_class_weights[0]
            norm_kitti_class_weights = norm_kitti_class_weights.tolist()
            norm_kitti_class_weights.append(0.1)
            self.class_weight = norm_kitti_class_weights
        elif dataset == 'nuscenes':
            nuscenes_class_weights = 1 / np.log(nusc_class_frequencies)
            norm_nuscenes_class_weights = nuscenes_class_weights / nuscenes_class_weights[0]
            norm_nuscenes_class_weights = norm_nuscenes_class_weights.tolist()
            norm_nuscenes_class_weights.append(0.1)
            self.class_weight = norm_nuscenes_class_weights
        else:
            self.class_weight = [1.0] * nclasses + [0.1]

    def forward(self, x):
        prune_loss = []
        current_feat = x[-1]
        outputs = [self.out_convs[-1](current_feat) if self.out_convs[-1] else current_feat]
        final_outs = [outputs[0]]
        seg_logits = []
        for i, up_layer in enumerate(self.upsample_blocks):
            current_feat, prune_loss_i, seg_i = up_layer(current_feat, short_cut=x[len(x)-2-i])
            if self.out_convs[len(self.out_convs)-2-i]:
                out = self.out_convs[len(self.out_convs)-2-i](current_feat)
            else:
                out = current_feat
            final_outs.insert(0, out)
            prune_loss.append(prune_loss_i)
            seg_logits.insert(0, seg_i)

        return final_outs, seg_logits

    def get_pruning_loss(self, seg_logits, gt_occ):
        pruning_loss = 0.0
        for logits in seg_logits:
            down_gt = gt_occ[int(math.log2(logits.tensor_stride[0]))+self.lss_downsample[0]//2].clone()
            valid = down_gt[logits.C[:, 0].long(),
                        logits.C[:,1].long()//(logits.tensor_stride[0]),
                        logits.C[:,2].long()//(logits.tensor_stride[0]),
                        logits.C[:,3].long()//(logits.tensor_stride[0])]
            class_weights_tensor = torch.tensor(self.class_weight[:-1]).type_as(logits.F)
            pruning_loss += CE_ssc_loss(logits.F, 
                                        valid.long(), 
                                        class_weights_tensor, 
                                        ignore_index=255)
            pruning_loss += sem_scal_loss(logits.F,
                                        valid.long(),
                                        ignore_index=255,)
            pruning_loss += geo_scal_loss(logits.F,
                                        valid.long(),
                                        ignore_index=255,
                                        non_empty_idx=self.empty_idx)
            pruning_loss += lovasz_softmax(torch.softmax(logits.F, dim=1),
                                            valid.long(),
                                            ignore=255,)
        return pruning_loss
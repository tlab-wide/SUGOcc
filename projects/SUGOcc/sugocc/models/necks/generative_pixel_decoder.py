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
                 voxel_range=[128, 128, 16],
                 process_block_num=1,
                 process_kernel_size=1,
                 process_cross_kernel=False,
                 nclasses=20,
                 empty_idx=0,
                 lss_downsample=[2,2,2],
                 dataset = 'semantickitti',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pruning_ratio = pruning_ratio
        self.voxel_range = voxel_range
        self.empty_idx = empty_idx
        self.lss_downsample = lss_downsample
        self.dataset = dataset
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers .append(
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
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x, short_cut, gt_occ=None):
        pruning_loss = 0.0

        seg = None
        x = self.upsample_layers[0](x)
            
        x = self.process(x + short_cut)

        mask = torch.zeros((x.F.shape[0],), dtype=torch.bool).to(x.F.device)
        
        seg = self.cls_layers[0](x)
        seg_prob = F.softmax(seg.F, dim=-1)

        mask = mask | ((1 - seg_prob[:,self.empty_idx]) > self.pruning_ratio)

        if gt_occ is not None:
            down_gt = gt_occ[int(math.log2(x.tensor_stride[0]))+self.lss_downsample[0]//2].clone()
            valid = down_gt[x.C[:, 0].long(),
                        x.C[:,1].long()//(x.tensor_stride[0]),
                        x.C[:,2].long()//(x.tensor_stride[0]),
                        x.C[:,3].long()//(x.tensor_stride[0])]
            nonempty = (valid != self.empty_idx)&(valid != 255)
            if mask.sum() == 0:
                mask = mask + nonempty
            weight = 1
            class_weights_tensor = torch.tensor(self.class_weight[:-1]).type_as(seg.F)
            pruning_loss += CE_ssc_loss(seg.F, 
                                        valid.long(), 
                                        class_weights_tensor, 
                                        ignore_index=255) * weight
            pruning_loss += sem_scal_loss(seg.F,
                                        valid.long(),
                                        ignore_index=255,) * weight
            pruning_loss += geo_scal_loss(seg.F,
                                        valid.long(),
                                        ignore_index=255,) * weight
            pruning_loss += lovasz_softmax(torch.softmax(seg.F, dim=1),
                                            valid.long(),
                                            ignore=255,) * weight
        x = self.pruning(x, mask)
        return x, pruning_loss, seg
    
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
                 is_pruning=[True, True, True],
                 process_cross_kernel=False,
                 generative=True,
                 soft_pruning=False,
                 pruning_ratio=[0.5, 0.5, 0.5],
                 voxel_size=[128,128,16],
                 dense_output=True,
                 norm_cfg=dict(type='BN1d'),
                 empty_idx=0,
                 lss_downsample=[2,2,2],
                 dataset='semantickitti',
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.upsample_cfg = upsample_cfg
        self.voxel_size = voxel_size
        self.is_pruning = is_pruning
        self.dense_output = dense_output
        self.empty_idx = empty_idx
        self.lss_downsample = lss_downsample
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        for i, in_channel in enumerate(self.in_channels):
            if i == len(self.in_channels) - 1:
                continue
            up_layer = MinkowskiUpAndPruneBlock(
                in_channels=self.in_channels[len(self.in_channels)-1-i],
                out_channels=self.in_channels[len(self.in_channels)-2-i],
                pruning_ratio=pruning_ratio[i],
                voxel_range=self.voxel_size,
                process_block_num=process_block_num,
                process_kernel_size=process_kernel_size,
                process_cross_kernel=process_cross_kernel,
                empty_idx = self.empty_idx,
                lss_downsample=self.lss_downsample,
                nclasses = nclasses,
                dataset=dataset,
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

    def forward(self, x, img_metas=None, gt_occ=None):
        prune_loss = []

        current_feat = x[-1]
        outputs = [self.out_convs[-1](current_feat) if self.out_convs[-1] else current_feat]
        final_outs = [outputs[0].dense(
                shape=torch.Size([int(outputs[0].C[:, 0].max()+1), outputs[0].F.shape[1],
                                  self.voxel_size[0]//outputs[0].tensor_stride[0],
                                  self.voxel_size[1]//outputs[0].tensor_stride[1],
                                  self.voxel_size[2]//outputs[0].tensor_stride[2]]),
                min_coordinate=torch.IntTensor([0, 0, 0]),
            )[0]]
        prune_seg_logits = []
        for i, up_layer in enumerate(self.upsample_blocks):
            current_feat, prune_loss_i, seg_i = up_layer(current_feat, short_cut=x[len(x)-2-i], gt_occ=gt_occ)
            if self.out_convs[len(self.out_convs)-2-i]:
                out = self.out_convs[len(self.out_convs)-2-i](current_feat)
            else:
                out = current_feat

            if self.dense_output:
                final_outs.insert(0, out.dense(
                    shape=torch.Size([int(out.C[:, 0].max()+1), out.F.shape[1],
                                      self.voxel_size[0]//out.tensor_stride[0],
                                      self.voxel_size[1]//out.tensor_stride[1],
                                      self.voxel_size[2]//out.tensor_stride[2]]),
                    min_coordinate=torch.IntTensor([0, 0, 0]),
                )[0])
            else:
                final_outs.insert(0, out)
            prune_loss.append(prune_loss_i)
            prune_seg_logits.insert(0, seg_i)

        return final_outs, prune_loss, prune_seg_logits

@MODELS.register_module()
class CascadeDensifyPixelDecoder(BaseModule):
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
                 norm_cfg=dict(type='BN1d'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.upsample_cfg = upsample_cfg

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.process_blocks = nn.ModuleList()
        upsample_cfg=dict(type='deconv3d', bias=False)
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
        conv_cfg=dict(type='Conv3d', bias=False)
        for i, in_channel in enumerate(self.in_channels):
            if i == len(self.in_channels) - 1:
                continue
            up_layer = nn.Sequential(
                build_upsample_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[len(self.in_channels)-1-i],
                    out_channels=self.in_channels[len(self.in_channels)-2-i],
                    kernel_size=2,
                    stride=2,
                ),
                build_norm_layer(norm_cfg, self.in_channels[len(self.in_channels)-2-i])[1],
                nn.ReLU(inplace=True),
            )
            self.upsample_blocks.append(up_layer)
            self.process_blocks.append(
                nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        in_channels=2*self.in_channels[len(self.in_channels)-2-i],
                        out_channels=self.in_channels[len(self.in_channels)-2-i],
                        kernel_size=3,
                        stride=1,
                        padding=0),
                    build_norm_layer(norm_cfg, self.in_channels[len(self.in_channels)-2-i])[1],
                    nn.ReLU(inplace=True),
                )
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(out_channels)):
            if in_channels[i] != out_channels[i]:
                self.out_convs.append(
                    nn.Sequential(
                        build_conv_layer(
                            conv_cfg,
                            in_channels=in_channels[i],
                            out_channels=out_channels[i],
                            kernel_size=1,
                            stride=1,
                            padding=0),
                        build_norm_layer(norm_cfg, out_channels[i])[1],
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.out_convs.append(None)

    def forward(self, x, img_metas=None, gt_occ=None):
        prune_loss = []
        B = int(x[0].C[:, 0].max()+1)
        inputs = []
        for i in range(len(x)):
            ch = x[i].features.shape[1]
            # np.save(f"8occ_down_coords_{self.downsample_rate[i]}.npy", outputs[i].coordinates.cpu().numpy())
            d = x[i].dense(
                shape=torch.Size([int(x[i].C[:, 0].max()+1), ch,
                                  128//x[i].tensor_stride[0],
                                  128//x[i].tensor_stride[1],
                                  16//x[i].tensor_stride[2]]),
                min_coordinate=torch.IntTensor([0, 0, 0]),
            )[0]
            inputs.append(d)
        outputs = [inputs[-1]]
        for i, up_layer in enumerate(self.upsample_blocks):
            feat = torch.cat([up_layer(outputs[0]), inputs[len(inputs)-2-i]], dim=1)
            feat = self.process_blocks[i](feat)
            outputs.insert(0, feat)

        for i in range(len(self.out_convs)):
            if self.out_convs[i]:
                outputs[i] = self.out_convs[i](outputs[i])
        
        return outputs, prune_loss
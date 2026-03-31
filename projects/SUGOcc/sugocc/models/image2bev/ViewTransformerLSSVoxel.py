# Copyright (c) Phigent Robotics. All rights reserved.
from calendar import c
import dis
import torch
import time
import os
import torch.utils.checkpoint as cp
from projects.SUGOcc.sugocc.ops.occ_pooling import occ_pool, occ_avg_pool
from projects.SUGOcc.sugocc.ops.bev_pool_v2 import bev_pool_v2
from projects.SUGOcc.sugocc.ops.bev_pool_v3 import bev_pool_v3
from projects.SUGOcc.sugocc.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss, multiscale_supervision
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import pdb
from .ViewTransformerLSSBEVDepth import *
from mmengine.registry import MODELS
import MinkowskiEngine as ME
import torch
import math

class DepthDistanceSineEncoding(torch.nn.Module):
    """
    1D sine/cosine positional encoding based on distance to the foreground depth d* (along the D dimension).
    Inputs:
        prob: [B, D, H, W]  depth probability (no normalization required)
        mask: [B, H, W] or None. Non-zero indicates ignored positions, zero indicates valid positions.
              If provided, distances at invalid pixels are zeroed out; an optional valid channel can be appended.
    Args:
        num_feats: number of channels per sine or cosine side; total output channels = 2 * num_feats
        temperature: frequency temperature, following DETR/Transformer convention
        normalize: whether to scale distances to [0, scale]
        scale: used together with normalize; commonly set to 2π
        distance_unit: converts index difference to real-world scale (e.g., 0.1m per bin)
        clamp_max: optional upper bound on distance to suppress high-frequency jitter
        add_valid_channel: whether to append 1 validity channel (0/1) at the end for downstream use
    Outputs:
        pos: [B, 2*num_feats (+1 if add_valid_channel), D, H, W]
    """
    def __init__(self,
                 num_feats: int = 64,
                 temperature: float = 10000.0,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 distance_unit: float = 1.0,
                 clamp_max: float = None,
                 add_valid_channel: bool = False):
        super().__init__()
        self.num_feats = int(num_feats)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.scale = float(scale)
        self.distance_unit = float(distance_unit)
        self.clamp_max = None if clamp_max is None else float(clamp_max)
        self.add_valid_channel = bool(add_valid_channel)
        self.eps = 1e-6  # avoid division by zero

        dim_t = torch.arange(self.num_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_feats)
        self.register_buffer('dim_t', dim_t)

        self.register_buffer('sin_idx', torch.arange(0, self.num_feats, 2))
        self.register_buffer('cos_idx', torch.arange(1, self.num_feats, 2))


    def forward(self, prob: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # prob: [B, D, H, W]
        assert prob.dim() == 4, "prob must be [B, D, H, W]"
        B, D, H, W = prob.size()
        device = prob.device
        dtype = torch.float32

        # 1) foreground depth index d* = argmax_d P
        # keepdim=True for broadcasting to [B, D, H, W]
        # 2) compute distance |d - d*|
        d_index = torch.arange(D, device=device, dtype=dtype)  # [1, D, 1, 1]
        # _s = time.time()
        d_star = torch.einsum('bdhw, d -> bhw', prob, d_index) # [B, H, W]
        dist = (d_index.view(1, D, 1, 1) - d_star.unsqueeze(1)) # [B, D, H, W]

        # 3) optional: convert index difference to real-world scale
        if self.distance_unit != 1.0:
            dist = dist * self.distance_unit

        # 4) optional: normalize to [0, scale] (consistent with 2D/3D sine encoding style)
        if self.normalize:
            # maximum possible distance: ~(D-1) in index units
            max_dist = float(D - 1) * (self.distance_unit if self.distance_unit != 1.0 else 1.0)
            dist = dist / max(max_dist, self.eps) * self.scale

        # 5) optional: clamp distance upper bound for numerical stability
        if self.clamp_max is not None:
            dist = dist.clamp(max=self.clamp_max)

        pos = dist.unsqueeze(-1) / self.dim_t.view(1, 1, 1, 1, -1)  # [B, D, H, W, C]
        # 7) interleave sin/cos -> [B, D, H, W, 2C], then permute to [B, 2C, D, H, W]
        # use stack + view for ONNX compatibility

        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=5)
        pos = pos.view(B, D, H, W, -1)               # [B, D, H, W, 2*C]
        pos = pos.permute(0, 4, 1, 2, 3) # [B, 2*C, D, H, W]

        # 8) optional: mask handling (non-zero = ignored, consistent with reference)
        if mask is not None:
            # cast to int, 0=valid, 1=ignored
            mask_i = mask.to(torch.int)
            # zero out encodings at invalid pixels to avoid interference
            valid = (1 - mask_i).to(dtype)  # [B, H, W], 1=valid, 0=ignored
            valid = valid.view(B, 1, 1, H, W)  # broadcast to [B, 2C, D, H, W]
            pos = pos * valid
            if self.add_valid_channel:
                pos = torch.cat([pos, valid], dim=1)  # append validity channel

        return pos  # [B, 2*num_feats (+1), D, H, W]

@MODELS.register_module()
class CM_DepthNet(BaseModule):
    """
        Camera parameters aware depth net
    """
    def __init__(self,
                 in_channels=512, #256
                 context_channels=64, #numC_Trans
                 depth_channels=118,
                 mid_channels=512,
                 use_dcn=True,
                 downsample=16,
                 stereo=False,
                 grid_config=None,
                 bias=0.0,
                 input_size=None,
                 loss_depth_weight=3.0,
                 with_cp=False,
                 se_depth_map=False,
                 sid=False,
                 aspp_mid_channels=-1,         
                 cam_channel=27,
                 num_class=18, #nusc
        ):
        super(CM_DepthNet, self).__init__()
        self.fp16_enable=False
        self.sid=sid
        self.with_cp = with_cp
        self.downsample = downsample
        self.grid_config = grid_config
        self.loss_depth_weight = loss_depth_weight
        self.stereo = stereo
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.se_depth_map = se_depth_map
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(cam_channel)
        self.depth_mlp = Mlp(cam_channel, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(cam_channel, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample_net = None
        self.num_class = num_class

        self.depth_stereo=stereo
        if stereo:
            cost_volumn_channels=depth_channels
            depth_conv_input_channels += cost_volumn_channels
            downsample_net = nn.Conv2d(depth_conv_input_channels,
                                    mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(int(2)):
                cost_volumn_net.extend([
                    nn.Conv2d(cost_volumn_channels, cost_volumn_channels, kernel_size=3,
                            stride=2, padding=1),
                    nn.BatchNorm2d(cost_volumn_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            
            self.bias = bias
        
            self.cv_frustum = self.create_frustum(grid_config['dbound'],
                                                    input_size,
                                                    downsample=self.downsample//4)
        depth_conv_list = [
           BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample_net),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        aspp_mid_channel = True
        if aspp_mid_channels < 0:
            aspp_mid_channels = mid_channels
            aspp_mid_channel = False

        depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels, aspp_mid_channel=aspp_mid_channel))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.D =depth_channels
        self.class_predictor = nn.Sequential(
                nn.Conv2d(self.context_channels , self.context_channels * 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.context_channels * 2),
                nn.ReLU(),
                nn.Conv2d(self.context_channels * 2, self.context_channels * 2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.context_channels * 2),
                nn.ReLU(),
                nn.Conv2d(self.context_channels * 2, self.num_class, kernel_size=1, stride=1, padding=0)
                )

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        return torch.stack((x, y, d), -1)
    
    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        # pass
        frustum =self.cv_frustum.to(metas['post_trans'].device)

        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)
        neg_mask = points[..., 2, 0] < 1e-3
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points = points[..., :2, :] / points[..., 2:3, :]

        points = metas['post_rots'][...,:2,:2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][...,:2].view(B, N, 1, 1, 1, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        grid = grid.view(B * N, D * H, W, 2)
        return grid


    def calculate_cost_volumn(self, metas):
        prev, curr = metas['cv_feat_list']
        
        group_size = 4
        _, c, hf, wf = curr.shape
        hi, wi = hf * 4, wf * 4
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = self.cv_frustum.shape
        # self.gen_grid(metas, B, N, D, H, W, hi, wi)
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)

        prev = prev.view(B * N, -1, H, W)
        curr = curr.view(B * N, -1, H, W)
        cost_volumn = 0
      
        for fid in range(curr.shape[1] // group_size):
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            warp_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              warp_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)
            cost_volumn += cost_volumn_tmp
       
        if not self.bias == 0:
            invalid = warp_prev[:, 0, ...].view(B * N, D, H, W) == 0
           
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias
           
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        
        
        return cost_volumn
    
    def forward(self, x, mlp_input, stereo_metas=None, img_metas=None):
        x = x.to(torch.float32)
        # B * N, C, H, W = x.shape
        # x = x.view(B * N, C, H, W)
        cost_volumn = None
        if self.depth_stereo:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn_ = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn_ = self.calculate_cost_volumn(stereo_metas)
                if cost_volumn is not None:
                    cost_volumn=torch.cat((cost_volumn,cost_volumn_),dim=1)
                else:
                    cost_volumn=cost_volumn_
        if cost_volumn is not None:
            cost_volumn = self.cost_volumn_net(cost_volumn)
            
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1])) 
        
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x) 
        context_se = self.context_mlp(mlp_input)[..., None, None]
        if self.with_cp and x.requires_grad:
            context = cp.checkpoint(self.context_se, x, context_se)
        else:
            context = self.context_se(x, context_se) 
        context = self.context_conv(context) 
        depth_se = self.depth_mlp(mlp_input)[..., None, None] 
        depth = self.depth_se(x, depth_se)
        if cost_volumn is not None:
            depth = torch.cat([depth, cost_volumn], dim=1)
        if self.with_cp and depth.requires_grad:
            depth = cp.checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)

        seg = self.class_predictor(context)
        return torch.cat([depth, context], dim=1), seg

@MODELS.register_module()
class SegAndDepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, cam_channels=27, num_classes=20, **kwargs):
        super(SegAndDepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        
        self.bn = nn.BatchNorm1d(cam_channels)
        
        self.context_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels, aspp_mid_channel=True),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
        )
        self.depth_out = nn.Conv2d(mid_channels,
            depth_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            padding=0)
        self.seg_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.seg_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.seg_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels, aspp_mid_channel=True),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
        )
        self.seg_out = nn.Conv2d(mid_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            bias=True,
            padding=0)

    # @auto_fp16()
    def forward(self, x, mlp_input, **kwargs):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        depth_out = self.depth_out(depth)

        seg_se = self.seg_mlp(mlp_input)[..., None, None]
        seg = self.seg_se(x, seg_se)
        seg = self.seg_conv(seg)
        seg_out = self.seg_out(seg)
        
        return torch.cat([depth_out, context], dim=1), seg_out
    
@MODELS.register_module()
class ViewTransformerLiftSplatShootVoxel(ViewTransformerLSSBEVDepth):
    def __init__(
            self, 
            loss_depth_weight,
            point_cloud_range=None,
            lss_downsample=[2,2,2],
            empty_idx=0,
            num_classes=20,
            num_cam=1,
            seg_pruning_ratio=0.1,
            depth_pruning_ratio=0.1,    
            dataset='semantickitti',
            depthnet_cfg=dict(),
            **kwargs,
        ):
        super(ViewTransformerLiftSplatShootVoxel, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
        self.cam_depth_range = self.grid_config['dbound']
        self.point_cloud_range = point_cloud_range
        self.lss_downsample = lss_downsample
        self.empty_idx = empty_idx
        self.num_classes = num_classes
        self.dataset = dataset
        self.num_cam = num_cam
        self.create_grid_infos(self.grid_config["xbound"],
                               self.grid_config["ybound"],
                               self.grid_config["zbound"])
        if self.dataset == 'semantickitti':
            depth_net = dict(
                type="SegAndDepthNet",
                in_channels=self.numC_input,
                mid_channels=self.numC_input,
                context_channels=self.numC_Trans,
                depth_channels=self.D,
                cam_channels=self.cam_channels,
                num_classes=self.num_classes,
            )
        else:
            depth_net = dict(
                type="CM_DepthNet",
                in_channels=self.numC_input,
                mid_channels=self.numC_input,
                context_channels=self.numC_Trans,
                depth_channels=self.D,
                cam_channel=self.cam_channels,
                num_class=self.num_classes,
                **depthnet_cfg,
            )

        self.depth_net = MODELS.build(depth_net)
        self.opacity_embedding = DepthDistanceSineEncoding(self.numC_Trans//self.num_cam-(self.numC_Trans//self.num_cam)%2)
        self.criterion_seg = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        self.seg_pruning_ratio = seg_pruning_ratio
        self.depth_pruning_ratio = depth_pruning_ratio
        self.initial_flag = True
        self.voxel_num = 0
        self.test_num=0
    
    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])
        
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        # [min - step / 2, min + step / 2] creates min depth
        
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths_vals, gt_depths.float()
    
    # @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return self.loss_depth_weight *depth_loss
    
    def get_seg_loss(self, x, gt_occ, rots, trans, intrins, 
                     post_rots, post_trans, bda):
        
        BN, C, H, W = x.shape
        
        if self.dataset == 'semantickitti':
            geom = self.get_geometry(
                rots=rots, trans=trans, intrins=intrins, 
                post_rots=post_rots, post_trans=post_trans, bda=bda)
            projects_seg_gt = self.get_seg_gt_from_occ(geom, gt_occ[self.lss_downsample[0]//2])
            labels = projects_seg_gt.clone().view(BN, -1, H, W)
            index = ((labels!=self.empty_idx)&(labels!=255)).long().argmax(dim=1)
        elif self.dataset == 'nuscenes':
            geom = self.get_ego_coor(
                rots, trans, intrins, post_rots, post_trans, bda)
            projects_seg_gt = self.get_seg_gt_from_occ(geom, gt_occ[self.lss_downsample[0]//2])
            labels = projects_seg_gt.clone().view(BN, -1, H, W)
            index = ((labels!=self.empty_idx)&(labels!=255)).long().argmax(dim=1)
        labels = torch.gather(labels, dim=1, index=index.unsqueeze(1)).squeeze(1)

        seg_loss = self.criterion_seg(
            x,
            labels.long()
        )
        return self.loss_depth_weight * seg_loss

    def get_seg_gt_from_occ(self, geom_feats, gt_occ):
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        # flatten indices
        geom_feats = ((geom_feats - self.grid_lower_bound.to(geom_feats.device)) / self.grid_interval.to(geom_feats.device)).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=geom_feats.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        train_gt_mask = None
        target = gt_occ.clone()
        if gt_occ is not None:
            valid = (geom_feats[..., 0] >= 0) & (geom_feats[..., 0] < self.grid_size[0]) \
               & (geom_feats[..., 1] >= 0) & (geom_feats[..., 1] < self.grid_size[1]) \
               & (geom_feats[..., 2] >= 0) & (geom_feats[..., 2] < self.grid_size[2])
            train_gt_mask = (torch.ones_like(valid)*255).long()
            train_gt_mask[valid] = target[
                geom_feats[valid][..., 3].long(),
                geom_feats[valid][..., 0].long(),
                geom_feats[valid][..., 1].long(),
                geom_feats[valid][..., 2].long()
            ].long()
            train_gt_mask = train_gt_mask.view(B, N, D, H, W)

        return train_gt_mask

    def get_seg_prob(self, x):
        B, C, H, W = x.shape
        return x.softmax(dim=1)
    
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - self.grid_lower_bound.to(x.device)) / self.grid_interval.to(x.device))
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        topk = (x.abs().sum(dim=1) >0)
        x = x[topk]
        geom_feats = geom_feats[topk]
        
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.grid_size[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.grid_size[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.grid_size[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        feats_final = x
        geom_feats_final = geom_feats.long()

        final = occ_pool(feats_final, geom_feats_final.long(), B, self.grid_size[2], self.grid_size[0], self.grid_size[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final
    
    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                     bda):

        B, N, _, _ = sensor2ego.shape
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)

        post_rots_inv = torch.inverse(post_rots)
        
        post_rots_inv= post_rots_inv.view(B, N, 1, 1, 1, 3, 3)
        
        points = post_rots_inv.matmul(points.unsqueeze(-1))
        # points = torch.einsum('bnij,bndhwj->bndhwi', torch.inverse(post_rots), points)
        
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)
        
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points
    
    def voxel_pooling_v2(self, coor, depth, feat):
        if not self.accelerate:
            ranks_bev, ranks_depth, ranks_feat, \
                interval_starts, interval_lengths = \
                self.voxel_pooling_prepare_v2(coor)
        else:
            ranks_bev = self.ranks_bev
            ranks_depth = self.ranks_depth
            ranks_feat = self.ranks_feat
            interval_starts = self.interval_starts
            interval_lengths = self.interval_lengths
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat) 
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy

        feat = feat.permute(0, 1, 3, 4, 2) 
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # if self.collapse_z:
        #     bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_v3(self, coor, depth, feat, dist_embed):
        if not self.accelerate:
            ranks_bev, ranks_depth, ranks_feat, \
                interval_starts, interval_lengths = \
                self.voxel_pooling_prepare_v3(coor, depth)
        else:
            ranks_bev = self.ranks_bev
            ranks_depth = self.ranks_depth
            ranks_feat = self.ranks_feat
            interval_starts = self.interval_starts
            interval_lengths = self.interval_lengths
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat) 
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        dist_embed = dist_embed.permute(0, 1, 3, 4, 5, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])

        bev_feat = bev_pool_v3(depth, feat, dist_embed, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)

        # if self.collapse_z:
        #     bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat


    def voxel_pooling_prepare_v2(self, coor):
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device) 
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device) 
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()

        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3) 
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1) 

        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()
    
    def voxel_pooling_prepare_v3(self, coor, mask):
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device) 
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device) 
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()

        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        mask = mask.view(num_points)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1) 

        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2]) & mask
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()
    
    def init_acceleration_v2(self, coor):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor(*input[1:7])  
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def get_mlp_input_nus(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
                                dim=-1)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input
    
    def forward(self, input, img_metas=None, stereo_metas=None):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        x, seg_out = self.depth_net(x, mlp_input, stereo_metas=stereo_metas)     

        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]

        depth_prob = self.get_depth_dist(depth_digit)
        seg_prob = self.get_seg_prob(seg_out)
           
        hazard = torch.cumsum(depth_prob, dim=1)

        seg_mask = (1-seg_prob[:,self.empty_idx,...] > self.seg_pruning_ratio).unsqueeze(1).repeat(1, self.D, 1, 1)
        mask = (hazard > self.depth_pruning_ratio)&(seg_mask)
        
        opacity_embeds = self.opacity_embedding(
            depth_prob,
        ).to(img_feat.dtype)  # B*N, D, H, W, C
        
        # Splat
        if self.dataset == 'semantickitti':
            img_feat = img_feat.unsqueeze(2) + opacity_embeds
            volume = mask.unsqueeze(1) * img_feat
            volume = volume.view(B, N, -1, self.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)  # B, N, D, H, W, C

            geom = self.get_geometry(
                rots=rots, trans=trans, intrins=intrins, 
                post_rots=post_rots, post_trans=post_trans, bda=bda)

            bev_feat = self.voxel_pooling(geom, volume)
        elif self.dataset == 'nuscenes':
            geom = self.get_ego_coor(*input[1:7])
                
            opacity_embeds = opacity_embeds.view(B, N, -1, self.D, H, W)
            dist_embed = opacity_embeds

            bev_feat = self.voxel_pooling_v3(
                geom, mask.view(B, N, self.D, H, W),
                img_feat.view(B, N, self.numC_Trans, H, W),
                dist_embed.view(B, N, self.numC_Trans, self.D, H, W)).permute(0, 1, 4, 3, 2)

        return bev_feat, depth_prob, seg_out
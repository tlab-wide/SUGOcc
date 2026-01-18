# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import time
# from mmdet3d.models.builder import NECKS
# from mmdet3d.ops.bev_pool import bev_pool
from projects.SUGOcc.sugocc.ops.occ_pooling import occ_pool, occ_avg_pool
from projects.SUGOcc.sugocc.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss, multiscale_supervision
# from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import pdb
# from mmdet3d.models import builder
from .ViewTransformerLSSBEVDepth import *
from mmdet3d.registry import MODELS
import MinkowskiEngine as ME
import torch
import math

class DepthDistanceSineEncoding(torch.nn.Module):
    """
    基于到前景深度 d* 的距离的一维正弦/余弦编码（沿 D 维）。
    输入:
        prob: [B, D, H, W]  深度概率(无需归一化)
        mask: [B, H, W] 或 None。非零为忽略位置，零为有效位置（与你给的代码一致）。
              若提供，将对无效像素把距离置零，并额外输出一个valid通道可选。
    参数:
        num_feats: 每个正弦或余弦一侧的通道数；总输出通道 = 2 * num_feats
        temperature: 频率温度，参考 DETR/Transformer
        normalize: 是否把距离缩放到 [0, scale]
        scale: 与 normalize 搭配；常用 2π
        distance_unit: 把索引差转换到真实尺度（如每层=0.1米）
        clamp_max: 可选距离上限，控制高频抖动
        add_valid_channel: 是否在最后附加 1 个有效性通道（0/1），方便下游使用
    输出:
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
        self.eps = 1e-6  # 防除零

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

        # 1) 前景深度索引 d* = argmax_d P
        # keepdim=True 方便广播到 [B, D, H, W]
        # 2) 计算距离 |d - d*|
        d_index = torch.arange(D, device=device, dtype=dtype)  # [1, D, 1, 1]
        _s = time.time()
        d_star = torch.einsum('bdhw, d -> bhw', prob, d_index) # [B, H, W]
        dist = (d_index.view(1, D, 1, 1) - d_star.unsqueeze(1)) # [B, D, H, W]

        # 3) 可选：按需求把索引差换算到真实尺度
        if self.distance_unit != 1.0:
            dist = dist * self.distance_unit

        # 4) 可选：归一化到 [0, scale]（与2D/3D正弦编码风格一致）
        if self.normalize:
            # 最大可能距离：若以索引差计，则 ~ (D-1)
            max_dist = float(D - 1) * (self.distance_unit if self.distance_unit != 1.0 else 1.0)
            dist = dist / max(max_dist, self.eps) * self.scale

        # 5) 可选：截断距离上限，数值更稳
        if self.clamp_max is not None:
            dist = dist.clamp(max=self.clamp_max)

        pos = dist.unsqueeze(-1) / self.dim_t.view(1, 1, 1, 1, -1)  # [B, D, H, W, C]
        # 7) 交替堆叠 sin/cos -> [B, D, H, W, 2C]，再转到 [B, 2C, D, H, W]
        # 用 stack + view 保持 ONNX 友好（与你参考代码一致）
        
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=5)
        pos = pos.view(B, D, H, W, -1)               # [B, D, H, W, 2*C]
        pos = pos.permute(0, 4, 1, 2, 3) # [B, 2*C, D, H, W]
        
        # 8) 可选：mask 处理（与参考代码一致：非零为忽略）
        if mask is not None:
            # 统一成 int，0=valid, 1=ignored
            mask_i = mask.to(torch.int)
            # 有时希望把无效像素的编码清零（避免干扰）
            valid = (1 - mask_i).to(dtype)  # [B, H, W], 1=valid, 0=ignored
            valid = valid.view(B, 1, 1, H, W)  # 便于广播到 [B, 2C, D, H, W]
            pos = pos * valid
            if self.add_valid_channel:
                pos = torch.cat([pos, valid], dim=1)  # 附加一个有效性通道

        return pos  # [B, 2*num_feats (+1), D, H, W]

@MODELS.register_module()
class SegAndDepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, cam_channels=27, num_classes=20):
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
            ASPP(mid_channels, mid_channels),
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
            ASPP(mid_channels, mid_channels),
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
    def forward(self, x, mlp_input):
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
            **kwargs,
        ):
        super(ViewTransformerLiftSplatShootVoxel, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
        self.cam_depth_range = self.grid_config['dbound']
        self.point_cloud_range = point_cloud_range
        self.lss_downsample = lss_downsample
        self.empty_idx = empty_idx
        self.num_classes = num_classes
        self.create_grid_infos(self.grid_config["xbound"],
                               self.grid_config["ybound"],
                               self.grid_config["zbound"])
        depth_net = dict(
            type="SegAndDepthNet",
            in_channels=self.numC_input,
            mid_channels=self.numC_input,
            context_channels=self.numC_Trans,
            depth_channels=self.D,
            cam_channels=self.cam_channels,
            num_classes=self.num_classes,
        )
        self.depth_net = MODELS.build(depth_net)
        self.opacity_embedding = DepthDistanceSineEncoding(self.numC_Trans)
        self.criterion_seg = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    
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
        
        return depth_loss
    
    def get_seg_loss(self, x, gt_occ, rots, trans, intrins, 
                     post_rots, post_trans, bda):
        geom = self.get_geometry(
            rots=rots, trans=trans, intrins=intrins, 
            post_rots=post_rots, post_trans=post_trans, bda=bda)
        BN, C, H, W = x.shape
        projects_seg_gt = self.get_seg_gt_from_occ(geom, gt_occ[self.lss_downsample[0]//2])
        labels = projects_seg_gt.clone().view(BN, -1, H, W)
        index = (labels!=self.empty_idx).long().argmax(dim=1)
        labels = torch.gather(labels, dim=1, index=index.unsqueeze(1)).squeeze(1)
        seg_loss = self.criterion_seg(
            x,
            labels.long()
        )
        return seg_loss

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
            train_gt_mask = (torch.ones_like(valid)*self.empty_idx).long()
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
    
    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape

        x = x.view(B * N, C, H, W)
        x, seg_out = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)
        seg_prob = self.get_seg_prob(seg_out)
        geom = self.get_geometry(
            rots=rots, trans=trans, intrins=intrins, 
            post_rots=post_rots, post_trans=post_trans, bda=bda)
                    
        hazard = torch.cumsum(depth_prob.detach(), dim=1)
        seg_mask = (seg_prob.detach()[:,self.empty_idx,...] < 0.9).unsqueeze(1).repeat(1, self.D, 1, 1)
        mask = (hazard > 0.1)&(seg_mask)
        opacity_embeds = self.opacity_embedding(
            depth_prob.detach(),
        ).to(img_feat.dtype)  # B*N, D, H, W, C

        img_feat = img_feat.unsqueeze(2) + opacity_embeds
        volume = mask.unsqueeze(1) * img_feat
        volume = volume.view(B, N, -1, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)  # B, N, D, H, W, C

        # Splat
        bev_feat = self.voxel_pooling(geom, volume)

        return bev_feat, depth_prob, seg_out
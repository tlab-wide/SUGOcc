import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.cnn.bricks import DropPath

class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        # Global coords does not require coords_key
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid())
        self.pooling = ME.MinkowskiGlobalPooling()
 

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
 

        return ME.SparseTensor(
            y.F * x.F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        # return self.broadcast_mul(x, y)
        
class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class BasicGenerativeDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=D,
                expand_coordinates=True
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )
     

    def forward(self, x):
        return self.net(x)

class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=D
            ),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiLeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.F.shape[0],) + (1,) * (x.F.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output_F = x.F.div(keep_prob) * random_tensor
    output = ME.SparseTensor(output_F, 
                             coordinate_map_key=x.coordinate_map_key,
                             coordinate_manager=x.coordinate_manager,)
    return output

class MinkDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(MinkDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3, drop_path=0., use_se=False, 
                 cross_kernel=False, expand_coords=False, voxel_range=[128, 128, 16]):
        super().__init__()
        self.cross_kernel = cross_kernel
        self.expand_coords = expand_coords
        self.voxel_range = voxel_range
        kernel_generator = ME.KernelGenerator(
            ks,
            stride,
            dilation,
            region_type=ME.RegionType.HYPER_CROSS,
            dimension=D)
        self.drop_path = MinkDropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.cross_kernel:
            self.net = nn.Sequential(
                ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=1,
                                    expand_coordinates=expand_coords,
                                    kernel_generator=kernel_generator if self.cross_kernel else None,
                                    dimension=D),
                ME.MinkowskiBatchNorm(outc),
                ME.MinkowskiLeakyReLU(inplace=True),
                ME.MinkowskiConvolution(outc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=1,
                                    expand_coordinates=expand_coords,
                                    kernel_generator=kernel_generator if self.cross_kernel else None,
                                    dimension=D),
                ME.MinkowskiBatchNorm(outc),
                ME.MinkowskiLeakyReLU(inplace=True),
                ME.MinkowskiConvolution(outc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    expand_coordinates=expand_coords,
                                    kernel_generator=kernel_generator if self.cross_kernel else None,
                                    stride=1,
                                    dimension=D),
                ME.MinkowskiBatchNorm(outc),
            )
        else:
            self.net = nn.Sequential(
                ME.MinkowskiConvolution(inc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    stride=stride,
                                    expand_coordinates=expand_coords,
                                    dimension=D),
                ME.MinkowskiBatchNorm(outc),
                ME.MinkowskiLeakyReLU(inplace=True),
                ME.MinkowskiConvolution(outc,
                                    outc,
                                    kernel_size=ks,
                                    dilation=dilation,
                                    expand_coordinates=expand_coords,
                                    stride=1,
                                    dimension=D),
                ME.MinkowskiBatchNorm(outc),
                ME.MinkowskiLeakyReLU(inplace=True),
            )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
            )

        self.relu = ME.MinkowskiLeakyReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning() if expand_coords else nn.Identity()
        self.use_se = use_se
        if use_se:
            self.se = SELayer(outc, reduction=2)

    def forward(self, x):
        skip = self.downsample(x)
        y = self.net(x)
        if self.expand_coords:
            mask = (y.C[:, 1] >= 0) & (y.C[:, 2] >= 0) & (y.C[:, 3] >= 0) & \
                   (y.C[:, 1] < self.voxel_range[0]) & (y.C[:, 2] < self.voxel_range[1]) & (y.C[:, 3] < self.voxel_range[2])
            y = self.pruning(y, mask)
        if self.use_se:
            y = self.se(y)
        out = self.relu(skip + y)
        return out

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv3d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv3d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            AtrousSeparableConvolution(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-3:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, in_channels, 1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
class SPCDense3Dv2(nn.Module):
    def __init__(self, init_size=16):
        super(SPCDense3Dv2, self).__init__()

        conv_layer = nn.Conv3d

        ### Completion sub-network
        bias = False
        act = nn.Identity()
        chs = [init_size, init_size * 1, init_size * 1, init_size * 1]
        self.a_conv1 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_1 = nn.BatchNorm3d(chs[1])

        self.a_conv2 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_2 = nn.BatchNorm3d(chs[1])
        self.a_conv3 = nn.Sequential(
            conv_layer(chs[1], chs[1], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias), act
        )
        self.bn_3 = nn.BatchNorm3d(chs[1])
        self.a_conv4 = nn.Sequential(
            conv_layer(chs[1], chs[1], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias), act
        )
        self.bn_4 = nn.BatchNorm3d(chs[1])

        self.a_conv5 = nn.Sequential(
            conv_layer(chs[1], chs[1], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_5 = nn.BatchNorm3d(chs[1])
        self.a_conv6 = nn.Sequential(
            conv_layer(chs[1], chs[1], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias), act
        )
        self.bn_6 = nn.BatchNorm3d(chs[1])
        self.a_conv7 = nn.Sequential(
            conv_layer(chs[1], chs[1], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias), act
        )
        self.bn_7 = nn.BatchNorm3d(chs[1])
        self.ch_conv1 = nn.Sequential(
            nn.Conv3d(chs[1], chs[0], kernel_size=1, stride=1, bias=bias), act
        )
        self.bn_ch_conv1 = nn.BatchNorm3d(chs[0])

        self.res_1 = nn.Sequential(
            conv_layer(chs[0], chs[0], (3, 3, 1), 1, padding=(1, 1, 0), bias=bias), act
        )
        self.bn_res_1 = nn.BatchNorm3d(chs[0])
        self.res_2 = nn.Sequential(
            conv_layer(chs[0], chs[0], (5, 5, 3), 1, padding=(2, 2, 1), bias=bias), act
        )
        self.bn_res_2 = nn.BatchNorm3d(chs[0])
        self.res_3 = nn.Sequential(
            conv_layer(chs[0], chs[0], (7, 7, 5), 1, padding=(3, 3, 2), bias=bias), act
        )
        self.bn_res_3 = nn.BatchNorm3d(chs[0])

    def forward(self, x_dense):
        ### Completion sub-network by dense convolution

        x1 = F.relu(self.bn_1(self.a_conv1(x_dense)))

        x2 = F.relu(self.bn_2(self.a_conv2(x1)))
        x3 = F.relu(self.bn_3(self.a_conv3(x1)))
        x4 = F.relu(self.bn_4(self.a_conv4(x1)))

        t1 = x2 + x3 + x4

        x5 = F.relu(self.bn_5(self.a_conv5(t1)))
        x6 = F.relu(self.bn_6(self.a_conv6(t1)))
        x7 = F.relu(self.bn_7(self.a_conv7(t1)))

        x = x1 + x2 + x3 + x4 + x5 + x6 + x7
        y0 = F.relu(self.bn_ch_conv1(self.ch_conv1(x)))
        y1 = F.relu(self.bn_res_1(self.res_1(x_dense)))
        y2 = F.relu(self.bn_res_2(self.res_2(x_dense)))
        y3 = F.relu(self.bn_res_3(self.res_3(x_dense)))
        x = x1 + y0 + y1 + y2 + y3
        # return F.relu(x)
        return x

class AttentionModule3D(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, (5, 5, 3), padding=(2, 2, 1), groups=dim)
        self.conv0_1 = nn.Conv3d(dim, dim, (1, 7, 1), padding=(0, 3, 0), groups=dim)
        self.conv0_2 = nn.Conv3d(dim, dim, (7, 1, 1), padding=(3, 0, 0), groups=dim)
        self.conv0_3 = nn.Conv3d(dim, dim, (1, 1, 3), padding=(0, 0, 1), groups=dim)

        self.conv1_1 = nn.Conv3d(dim, dim, (1, 11, 1), padding=(0, 5, 0), groups=dim)
        self.conv1_2 = nn.Conv3d(dim, dim, (11, 1, 1), padding=(5, 0, 0), groups=dim)
        self.conv1_3 = nn.Conv3d(dim, dim, (1, 1, 3), padding=(0, 0, 1), groups=dim)

        self.conv2_1 = nn.Conv3d(
            dim, dim, (1, 15, 1), padding=(0, 7, 0), groups=dim)
        self.conv2_2 = nn.Conv3d(
            dim, dim, (15, 1, 1), padding=(7, 0, 0), groups=dim)
        self.conv2_3 = nn.Conv3d(
            dim, dim, (1, 1, 3), padding=(0, 0, 1), groups=dim)
        self.conv3 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_0 = self.conv0_3(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_1 = self.conv1_3(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_2 = self.conv2_3(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention3D(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule3D(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
    
class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
    
class MSCABlock(BaseModule):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention3D(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # print(x.shape)
        B, C, X, Y, Z = x.shape
        # x = x.permute(0, 2, 1).view(B, C, X, Y, Z)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, X, Y, Z)
        return x
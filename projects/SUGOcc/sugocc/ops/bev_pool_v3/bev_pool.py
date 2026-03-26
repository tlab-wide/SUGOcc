# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

from . import bev_pool_v3_ext

__all__ = ['bev_pool_v3']


class QuickCumsumCuda(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """
    @staticmethod
    def forward(ctx, depth, feat, depth_embed, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()     # (N_points, ),
        depth = depth.contiguous().float()  # (B, N, D, fH, fW)
        feat = feat.contiguous().float()    # (B, N, fH, fW, C)
        depth_embed = depth_embed.contiguous().float()  # (B, N, D, fH, fW, C)
        ranks_depth = ranks_depth.contiguous().int()    # (N_points, ),
        ranks_feat = ranks_feat.contiguous().int()      # (N_points, ),
        interval_lengths = interval_lengths.contiguous().int()  # (N_pillar, )
        interval_starts = interval_starts.contiguous().int()    # (N_pillar, )

        out = feat.new_zeros(bev_feat_shape)    # (B, D_Z, D_Y, D_X, C)

        bev_pool_v3_ext.bev_pool_v3_forward(
            depth,
            feat,
            depth_embed,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, depth_embed, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, depth_embed, ranks_feat, ranks_depth = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
            1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        depth_embed_grad = depth_embed.new_zeros(depth_embed.shape)
        out_grad = out_grad.contiguous()
        bev_pool_v3_ext.bev_pool_v3_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth_embed_grad,
            depth,
            feat,
            depth_embed,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, None, \
            None, None, None


def bev_pool_v3(depth, feat, depth_embed, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """
    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_depth: (N_points, ),
        ranks_feat:  (N_points, ),
        ranks_bev:   (N_points, ),
        bev_feat_shape: (B, D_Z, D_Y, D_X, C)
        interval_starts: (N_pillar, )
        interval_lengths: (N_pillar, )

    Returns:
        x: bev feature in shape (B, C, Dz, Dy, Dx)
    """
    x = QuickCumsumCuda.apply(depth, feat, depth_embed, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts,
                              interval_lengths)      # (B, Dz, Dy, Dx, C)
    x = x.permute(0, 4, 1, 2, 3).contiguous()        # (B, C, Dz, Dy, Dx)
    return x
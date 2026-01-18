# Copyright (c) OpenMMLab. All rights reserved.
from encodings.punycode import decode_generalized_number
from matplotlib.pyplot import cla
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule

from mmdet.models import DetrTransformerDecoderLayer
from mmengine.model import caffe2_xavier_init
from mmcv.cnn import Conv2d, Conv3d
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version
# caffe2_xavier_init
import fvcore.nn.weight_init as weight_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule, ModuleList

from mmdet.models import build_assigner, build_sampler
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.registry import MODELS

from .base.mmdet_utils import (sample_valid_coords_with_frequencies,
                          get_uncertain_point_coords_3d_with_frequency,
                          preprocess_occupancy_gt, point_sample_3d)

from .base.anchor_free_head import AnchorFreeHead
from .base.maskformer_head import MaskFormerHead
from projects.SUGOcc.sugocc.utils.semkitti import semantic_kitti_class_frequencies
from projects.SUGOcc.sugocc.utils.nusc import nusc_class_frequencies
from projects.SUGOcc.sugocc.utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.SUGOcc.sugocc.utils.lovasz_softmax import lovasz_softmax
from einops import rearrange



# Sparse Mask2former Head for 3D Occupancy Segmentation
@MODELS.register_module()
class SparseOCRMask2OccHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=20,
                 dataset='semantickitti',
                 final_occ_size=[512, 512, 40],
                 lss_downsample=[2, 2, 2],
                 num_queries=100,
                 num_transformer_feat_level=3,
                 enforce_decoder_input_project=False,
                 img_feat_channels=256,
                 transformer_decoder=None,
                 positional_encoding=None,
                 pooling_attn_mask=True,
                 sample_weight_gamma=0.25,
                 dn_num=100,
                 noise_scale=0.4,
                 empty_idx=0,
                 with_cp=False,
                 align_corners=True,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 grid_config=None,
                 data_config=None,
                 norm_cfg=dict(type='GN', num_groups=20, requires_grad=True),
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.with_cp = with_cp
        self.dn_num = dn_num
        self.noise_scale = noise_scale
        self.lss_downsample = lss_downsample
        self.dataset = dataset
        self.empty_idx = empty_idx
        ''' Transformer Decoder Related '''
        # number of multi-scale features for masked attention
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution, align the channel of input features
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv3d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
                
        self.decoder_positional_encoding = MODELS.build(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        self.global_prototypes = nn.Embedding(self.num_classes, feat_channels)
        self.label_enc = nn.Embedding(self.num_classes, feat_channels)
        self.global_initialized = torch.zeros(self.num_classes).cuda().bool()
        # from low resolution to high resolution
        # self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)
        ''' Pixel Decoder Related, skipped '''
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.sample_weight_gamma = sample_weight_gamma
        self.class_weight = loss_cls.class_weight
        # create class_weights for semantic_kitti
        if self.dataset == 'semantickitti':
            class_frequencies = semantic_kitti_class_frequencies
        elif self.dataset == 'nuscenes':
            class_frequencies = nusc_class_frequencies
        else:
            class_frequencies = [1.0 for _ in range(self.num_occupancy_classes)]
            # normalize the class weights based on semantic kitti class frequencies
        class_weights = 1 / np.log(class_frequencies)
        norm_class_weights = class_weights / class_weights[self.empty_idx]
        norm_class_weights = norm_class_weights.tolist()
        # append the class_weight for background
        norm_class_weights.append(self.class_weight[-1])
        # print("********************12313******", self.class_weight[-1])
        self.class_weight = norm_class_weights
        loss_cls.class_weight = self.class_weight
        sample_weights = 1 / class_frequencies
        sample_weights = sample_weights / sample_weights.min()
        self.baseline_sample_weights = sample_weights
        self.get_sampling_weights()    
        
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        self.pooling_attn_mask = pooling_attn_mask
        
        # align_corners
        self.align_corners = align_corners

        # for sparse segmentation
        self.empty_token = nn.Embedding(1, out_channels)
        self.final_occ_size = final_occ_size

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_transformer_feat_level):
            if feat_channels != img_feat_channels:
                self.input_proj.append(
                    ConvModule(
                        img_feat_channels, feat_channels, kernel_size=1, act_cfg=None))
                weight_init.c2_xavier_fill(self.input_proj[-1].conv)
            else:
                self.input_proj.append(nn.Sequential())

        self.coarse_occ_pred = nn.ModuleList()
        self.refine_attns = nn.ModuleList()
        self.query_inprojs = nn.ModuleList()
        for i in range(self.num_transformer_decoder_layers):
            # self.refine_attns.append(SparseVoxelRefinedAttn(
            #     embed_dims=feat_channels,
            #     num_heads=8,
            #     num_levels=self.num_transformer_feat_level,
            #     num_points=4,
            #     mlp_ratio=4,
            #     grid_config=grid_config,
            #     data_config=data_config,
            # ))

            self.coarse_occ_pred.append(nn.Sequential(
                Conv3d(feat_channels, feat_channels//2, kernel_size=1, stride=1, padding=0),
                # nn.GroupNorm(16, feat_channels//2),
                build_norm_layer(norm_cfg, feat_channels//2)[1],
                nn.ReLU(inplace=True),
                Conv3d(feat_channels//2, num_occupancy_classes, kernel_size=1, stride=1, padding=0)
            ))

            self.query_inprojs.append(nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels)))

    def get_sampling_weights(self):
        if type(self.sample_weight_gamma) is list:
            # dynamic sampling weights
            min_gamma, max_gamma = self.sample_weight_gamma
            sample_weight_gamma = np.random.uniform(low=min_gamma, high=max_gamma)
        else:
            sample_weight_gamma = self.sample_weight_gamma
        
        self.sample_weights = self.baseline_sample_weights ** sample_weight_gamma
        
    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv3d):
                caffe2_xavier_init(m, bias=0)
        
        if hasattr(self, "pixel_decoder"):
            self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas,)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        # create sampling weights
        point_indices, point_coords = sample_valid_coords_with_frequencies(self.num_points, 
                gt_labels=gt_labels, gt_masks=gt_masks, sample_weights=self.sample_weights)
        
        point_coords = point_coords[..., [2, 1, 0]]
        mask_points_pred = point_sample_3d(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1), align_corners=self.align_corners).squeeze(1)
        
        # shape (num_gts, num_points)
        gt_points_masks = gt_masks.view(num_gts, -1)[:, point_indices]

        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label target
        labels = gt_labels.new_full((num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = labels.new_ones(num_queries).type_as(cls_score)
        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_score)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries, ))
        mask_weights[pos_inds] = class_weights_tensor[labels[pos_inds]]
        # print(mask_weights.shape)
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    # @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
                gt_masks_list, img_metas, **kwargs):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)
        
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas, **kwargs):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, x, y, z).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, x, y, z).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                gt_labels_list, gt_masks_list, img_metas)
        
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum(),
        )
        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_weights = mask_weights[mask_weights > 0]
        
        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        ''' 
        randomly sample K points for supervision, which can largely improve the 
        efficiency and preserve the performance. oversample_ratio = 3.0, importance_sample_ratio = 0.75
        '''
        with torch.no_grad():
            point_indices, point_coords = get_uncertain_point_coords_3d_with_frequency(
                mask_preds.unsqueeze(1), None, gt_labels_list, gt_masks_list, 
                self.sample_weights, self.num_points, self.oversample_ratio, 
                self.importance_sample_ratio)
            
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = torch.gather(mask_targets.view(mask_targets.shape[0], -1), 
                                        dim=1, index=point_indices)
        
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample_3d(
            mask_preds.unsqueeze(1), point_coords[..., [2, 1, 0]], align_corners=self.align_corners).squeeze(1)
        
        # dice loss
        num_total_mask_weights = reduce_mean(mask_weights.sum())
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, 
                        weight=mask_weights, avg_factor=num_total_mask_weights)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_weights = mask_weights.view(-1, 1).repeat(1, self.num_points)
        mask_point_weights = mask_point_weights.reshape(-1)
        
        num_total_mask_point_weights = reduce_mean(mask_point_weights.sum())
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            weight=mask_point_weights,
            avg_factor=num_total_mask_point_weights)

        return loss_cls, loss_mask, loss_dice

    def forward_head(self, 
                     decoder_out, 
                     mask_feature, 
                     nonempty_mask=None):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries, x, y, z).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        # print(decoder_out.shape, mask_feature.shape, attn_mask_target_size, coords.shape)
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(decoder_out)

        # return cls_pred, mask_pred, None
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)
        
        
        empty_pred = torch.einsum('bqc, bc -> bq', 
                                    mask_embed, 
                                    self.empty_token.weight.view(1, -1))  # B, Q
        mask_3d = empty_pred.reshape(empty_pred.shape[0], -1, 1, 1, 1)
        mask_3d = torch.where(nonempty_mask.unsqueeze(1).repeat(1, mask_pred.shape[1], 1, 1, 1), mask_pred, mask_3d)
        
        # print(mask_3d.shape)
        return cls_pred, mask_3d

    def preprocess_gt(self, gt_occ, img_metas):
        
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        targets = multi_apply(preprocess_occupancy_gt, gt_occ, num_class_list, img_metas)
        
        labels, masks = targets
        return labels, masks
    
    def generate_prototypes(self, mask_features, coarse_occ):
        """Generate prototypes from mask features and coarse occupancy predictions.

        Args:
            mask_features (Tensor): Mask features from pixel decoder.
                Shape (B, C, H, W, D).
            coarse_occ (Tensor): Coarse occupancy predictions.
                Shape (B, num_occupancy_classes, H, W, D).
        """
        assert mask_features.shape[-3:] == coarse_occ.shape[-3:]
        bs, c, x, y, z = mask_features.shape
        _, k, _, _, _ = coarse_occ.shape

        nonempty_mask = (mask_features.abs().sum(dim=1) > 0).view(bs, -1)  # B, 1, N

        coarse_occ = coarse_occ.view(bs, k, -1) # B, K, N
        mask_features = mask_features.reshape(bs, c, -1).permute(0, 2, 1)  # B, N, C

        prototype_list = []
        for b in range(bs):
            indices = nonempty_mask[b].nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                # prototypes_b
                feat = mask_features[b]  # [N, C]
                coarse = coarse_occ[b]  # [K, N]
                spatial_probs = F.softmax(coarse, dim=1)  # K, N
                prototypes_b = torch.matmul(spatial_probs, feat)  # K, C
                prototype_list.append(prototypes_b)
                continue
            feats = mask_features[b, indices, :]  # [num_high_confidence, C]
            coarse = coarse_occ[b, :, indices]  # [K, num_high_confidence]
            spatial_probs = F.softmax(coarse, dim=1)  # K, num_high_confidence
            prototypes_b = torch.matmul(spatial_probs, feats)  # K, C
            prototype_list.append(prototypes_b)
        prototypes = torch.stack(prototype_list, dim=0)  # B, K,
        return prototypes  # B, K, C
    
    def prepare_for_dn(self, targets, batch_size):
        scalar, noise_scale = self.dn_num, self.noise_scale
        known = [(torch.ones_like(t)).cuda() for t in targets["labels"]]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]
        if max(known_num) > 0:
            scalar = scalar // (int(max(known_num)))
        else:
            scalar = 0
        if scalar == 0:
            input_query_label = None
            attn_mask = None
            mask_dict = None
            return input_query_label, attn_mask, mask_dict
        unmask_label = torch.cat(known)
        labels = torch.cat([t for t in targets['labels']])
        batch_idx = torch.cat([
            torch.full_like(t.long(), i)
            for i, t in enumerate(targets['labels'])
        ])

        # known
        known_indice = torch.nonzero(unmask_label)
        known_indice = known_indice.view(-1)

        # noise
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        # known_bboxs = boxes.repeat(scalar, 1)
        known_labels_expaned = known_labels.clone()
        # known_bbox_expand = known_bboxs.clone()

        # noise on the label
        if noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(
                -1)  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0,
                self.num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        m = known_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m)
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)

        padding_label = torch.zeros(pad_size, self.feat_channels).cuda()
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        # map
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([
                torch.tensor(range(num)) for num in known_num
            ])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i for i in range(scalar)
            ]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(),
                                map_known_indice)] = input_label_embed
        tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1),
                            single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad *
                            (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1),
                            single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad *
                            (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_labels': (known_labels,),
            'know_idx': know_idx,
            'pad_size': pad_size,
            'scalar': scalar,
        }
        return input_query_label, attn_mask, mask_dict
    
    def dn_post_process(self, outputs_class, mask_dict,
                        outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        outputs_class = torch.stack(outputs_class)
        outputs_mask = torch.stack(outputs_mask)

        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]

        output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
        outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        return list(output_known_class)+list(outputs_class), list(output_known_mask)+list(outputs_mask)
        
    def forward(self, 
            voxel_feats,
            img_metas,
            targets=None,
            **kwargs,
        ):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 5D-tensor (B, C, X, Y, Z).
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        """
        batch_size = len(img_metas)
        mask_features = voxel_feats[0]
        mask_features = mask_features.dense(
            shape=torch.Size([
                int(mask_features.C[:, 0].max() + 1), mask_features.F.shape[1],
                self.final_occ_size[0]//(mask_features.tensor_stride[0]*self.lss_downsample[0]),
                self.final_occ_size[1]//(mask_features.tensor_stride[1]*self.lss_downsample[1]),
                self.final_occ_size[2]//(mask_features.tensor_stride[2]*self.lss_downsample[2])
            ]),
            min_coordinate=torch.IntTensor([0, 0, 0]),
        )[0]  # B, C, W, H, D
        B, _, W, H, D = mask_features.shape
        # multi_scale_memorys = voxel_feats[:0:-1]
        nonempty_mask = (mask_features.abs().sum(dim=1) > 0)  # B, W, H, 

        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        scale = self.num_queries//self.num_classes
        global_prototypes = self.global_prototypes.weight.unsqueeze(1).repeat((scale, batch_size, 1))
        query_feat = self.query_inprojs[0](query_feat + global_prototypes)

        input_query_label, dn_attn_mask, mask_dict = None, None, None
        if targets is not None and self.dn_num > 0:
            input_query_label, dn_attn_mask, mask_dict = self.prepare_for_dn(targets, batch_size)
            query_feat = torch.cat([input_query_label.transpose(0,1), query_feat], dim=0)
            # dn_query
            # query_embed = torch.cat([query_embed.clone()[:self.dn_num], query_embed], dim=0)
        else:
            input_query_label, dn_attn_mask, mask_dict = None, None, None

        cls_pred_list = []
        mask_pred_list = []
        coarse_occ_list = []

        cls_pred, mask_pred = self.forward_head(
            query_feat, 
            mask_features, 
            nonempty_mask=nonempty_mask)

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        for i in range(self.num_transformer_decoder_layers):
            '''
            if the attn_mask is all True (ignore everywhere), simply change it to all False (attend everywhere) 
            '''
            coarse_occ = self.coarse_occ_pred[i](mask_features)  # B, C, H, W, D
            coarse_occ_list.append(coarse_occ)
            local_prototypes = self.generate_prototypes(mask_features, coarse_occ)
            if self.training:
                with torch.no_grad():
                    nonempty_prototypes_mask = local_prototypes.mean(dim=0).sum(dim=-1) != 0  # K
                    no_assign_flag = nonempty_prototypes_mask * self.global_initialized
                    if no_assign_flag.sum() != 0:
                        self.global_prototypes.weight[no_assign_flag] = local_prototypes.mean(dim=0)[no_assign_flag].detach()
                        self.global_initialized[no_assign_flag] = False
                    proto_assign_flag = self.global_initialized == False
                    assign_flag = proto_assign_flag * nonempty_prototypes_mask
                    if assign_flag.shape[0] != 0:
                        self.global_prototypes.weight[assign_flag] = self.global_prototypes.weight[assign_flag] * (1-0.01) + local_prototypes.mean(dim=0)[assign_flag].detach()*0.01
        
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            # attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=local_prototypes.permute(1, 0, 2),
                value=local_prototypes.permute(1, 0, 2),
                query_pos=None,
                # key_pos=decoder_positional_encodings[level_idx],
                attn_masks=[None, dn_attn_mask],
                query_key_padding_mask=None,
                key_padding_mask=None)
            
            cls_pred, mask_pred = self.forward_head(
                query_feat, mask_features, 
                nonempty_mask=nonempty_mask)
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        
        if mask_dict is not None:
            cls_pred_list, mask_pred_list = self.dn_post_process(cls_pred_list, mask_dict, mask_pred_list)
        
        '''
        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        '''

        return cls_pred_list, mask_pred_list, coarse_occ_list

    def forward_train(self,
            voxel_feats,
            img_metas,
            gt_occ,
            **kwargs,
        ):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(
            gt_occ[self.lss_downsample[0]//2], 
            img_metas)
        
        targets = {
            "labels": gt_labels,
            "masks": gt_masks,
        }
        # forward
        all_cls_scores, all_mask_preds, coarse_occ = self.forward(voxel_feats, img_metas, targets=targets)

        # loss
        loss_dict = {}
        for i, coarse_occ in enumerate(coarse_occ):
            losses_voxel = self.loss_voxel(coarse_occ, gt_occ[self.lss_downsample[0]//2], f"coarse_{i}")
            loss_dict.update(losses_voxel)
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, 
                           gt_masks, img_metas, sparse_mask=None)
        loss_dict.update(losses)
        
        return loss_dict
    
    def format_results(self, mask_cls_results, mask_pred_results):
        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        output_voxels = torch.einsum("bqc, bqxyz->bcxyz", mask_cls, mask_pred)
        
        return output_voxels

    def simple_test(self, 
            voxel_feats,
            img_metas,
            **kwargs,
        ):
        all_cls_scores, all_mask_preds, _ = self.forward(voxel_feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=self.final_occ_size,
            mode='trilinear',
            align_corners=self.align_corners,
        )

        output_voxels = self.format_results(mask_cls_results, mask_pred_results)
        res = {
            'output_voxels': [output_voxels],
            'output_voxel_refine': None,
            'output_points': None,
        }

        return res
    
    def loss_voxel(self, output_voxels, target_voxels, tag='coarse'):
        gt_occ = target_voxels.clone()
        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = gt_occ.shape[2] // H
        if ratio != 1:
            gt_occ = gt_occ.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = gt_occ.sum(-1) == self.empty_idx
            gt_occ = gt_occ.to(torch.int64)
            occ_space = gt_occ[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            gt_occ[~empty_mask] = occ_space
            gt_occ = torch.mode(gt_occ, dim=-1)[0]
            gt_occ[gt_occ<0] = 255
            gt_occ = gt_occ.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(gt_occ).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        class_weights_tensor = torch.tensor(self.class_weight[:-1]).type_as(output_voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = CE_ssc_loss(output_voxels, gt_occ, class_weights_tensor, ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = sem_scal_loss(output_voxels, gt_occ, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = geo_scal_loss(output_voxels, gt_occ, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = lovasz_softmax(torch.softmax(output_voxels, dim=1), gt_occ, ignore=255)

        return loss_dict
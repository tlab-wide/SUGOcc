import torch
import torch.nn.functional as F
import collections 

from mmdet3d.models import Base3DSegmentor
from mmdet3d.structures import PointData
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
# from mmdet.models import DETECTORS
# from mmcv.runner import force_fp32, auto_fp16
from projects.SUGOcc.sugocc.utils import fast_hist_crop
# from .bevdepth import BEVDepth

import numpy as np
import time
import pdb

@MODELS.register_module()
class SUGOcc(Base3DSegmentor):
    def __init__(self, 
                 data_preprocessor=None, 
                 img_backbone=None,
                 img_neck=None,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 pts_bbox_head=None,
                 loss_norm=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.loss_norm = loss_norm
        self.nonempty_num = 0
        self.sample_num = 0
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        else:
            self.img_neck = None
            

        if img_view_transformer is not None:
            self.img_view_transformer = MODELS.build(img_view_transformer)
        else:
            self.img_view_transformer = None
        
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = MODELS.build(img_bev_encoder_backbone)
        else:
            self.img_bev_encoder_backbone = torch.nn.Identity()
        
        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)
        else:
            self.img_bev_encoder_neck = None

        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = MODELS.build(pts_bbox_head)

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        _s = time.time()
        x = self.img_backbone(imgs)
        if isinstance(x, dict):
            img_feats = list(x.get('feats', x))
            for i in range(len(img_feats)):
                img_feats[i] = img_feats[i].contiguous()
        else:
            img_feats = x

        if self.img_neck is not None:
            _s = time.time()
            fused_feats = self.img_neck(img_feats)
            if type(fused_feats) in [list, tuple]:
                fused_feats = fused_feats[0]
        _, output_dim, ouput_H, output_W = fused_feats.shape
        fused_feats = fused_feats.view(B, N, output_dim, ouput_H, output_W)
        
        return fused_feats

    def bev_encoder(self, x):
        voxel_seg_logits = []
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        if self.img_bev_encoder_backbone:
            x = self.img_bev_encoder_backbone(x)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['bev_encoder'].append(t1 - t0)
        if self.img_bev_encoder_neck:
            x, voxel_seg_logit = self.img_bev_encoder_neck(x)
            voxel_seg_logits.extend(voxel_seg_logit)
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['bev_neck'].append(t2 - t1)
        
        return x, voxel_seg_logits

    def extract_img_feat(self, img):
        """Extract features of images."""
        # print(len(img[0]))
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        x = self.image_encoder(img[0])
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]

        x, depth, seg = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)

        x, voxel_seg_logits = self.bev_encoder(x)
        if type(x) is not list:
            x = [x]
        
        return x, depth, seg, voxel_seg_logits
    
    def extract_feat(self, points, img):
        """Extract features from images and points."""
        voxel_feats, depth, seg, voxel_seg_logits  = self.extract_img_feat(img)
        return (voxel_feats, depth, seg, voxel_seg_logits)

    def _forward(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_inputs = batch_inputs['img_inputs']
        gt_occ = []
        for gt in batch_inputs['gt_occ']:
            gt_occ.append(torch.stack(gt, dim=0))
        for i in range(len(img_inputs)):
            img_inputs[i] = torch.stack(img_inputs[i], dim=0)
        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        outs = self.forward_test(img_metas=img_metas, img_inputs=img_inputs, gt_occ=gt_occ)
        return outs

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        img_inputs = batch_inputs['img_inputs']
        gt_occ = []
        for gt in batch_inputs['gt_occ']:
            gt_occ.append(torch.stack(gt, dim=0))
        for i in range(len(img_inputs)):
            img_inputs[i] = torch.stack(img_inputs[i], dim=0)
        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        losses = self.forward_train(img_metas=img_metas, img_inputs=img_inputs, gt_occ=gt_occ)
        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        if batch_data_samples is None:
            batch_data_samples = batch_inputs['data_samples']
            batch_inputs = batch_inputs['inputs']
        img_inputs = batch_inputs['img_inputs']
        for i in range(len(img_inputs)):
            img_inputs[i] = torch.stack(img_inputs[i], dim=0) 
        gt_occ = []
        for gt in batch_inputs['gt_occ']:
            gt_occ.append(torch.stack(gt, dim=0)) 
        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        outs = self.forward_test(img_metas=img_metas, img_inputs=img_inputs)
        return self.postprocess_result(outs, batch_data_samples)

    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        pass

    def postprocess_result(self, seg_logits_list,
                           batch_data_samples: SampleList) -> SampleList:
        for i in range(len(seg_logits_list)):
            seg_logits = seg_logits_list[i]
            batch_data_samples[i].set_data({
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': seg_logits})
            })
        return batch_data_samples

    # @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(
            self,
            pts_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            points_uv=None,
            transform=None,
            **kwargs,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        losses = self.pts_bbox_head.forward_train(
            voxel_feats=pts_feats,
            img_metas=img_metas,
            gt_occ=gt_occ,
            points=points_occ,
            points_uv=points_uv,
            transform=transform,
            **kwargs,
        )
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['mask2former_head'].append(t1 - t0)
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            points_uv=None,
            **kwargs,
        ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        voxel_feats, depth, seg, voxel_seg_logits  = self.extract_feat(
            points, img=img_inputs)

        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()

        losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[7], depth)
        losses['loss_seg'] = self.img_view_transformer.get_seg_loss(seg, gt_occ, *img_inputs[1:7])
        losses["loss_pruning"] = self.img_bev_encoder_neck.get_pruning_loss(voxel_seg_logits, gt_occ)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
        
        transform = img_inputs[1:] if img_inputs is not None else None

        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ, 
                                                  points_occ, img_metas, 
                                                  points_uv=points_uv, 
                                                  transform=transform, **kwargs)
        losses.update(losses_occupancy)
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)
        
        if self.record_time:
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)            
            print(out_res)

        return losses
        
    def forward_test(self,
            img_metas=None,
            img_inputs=None,
            **kwargs,
        ):
        res = self.simple_test(img_metas, img_inputs, **kwargs)
        return res
    

    def simple_test(self, img_metas, img=None, rescale=False, points_occ=None, points_uv=None):

        x = self.image_encoder(img[0])
        
        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        x, _, _ = self.img_view_transformer([x] + geo_inputs)
        
        x, _ = self.bev_encoder(x)
        if type(x) is not list:
            x = [x]

        transform = img[1:] if img is not None else None
        output = self.pts_bbox_head.simple_test(
            voxel_feats=x,
            points=points_occ,
            img_metas=img_metas,
            points_uv=points_uv,
            transform=transform,
        )
        return [output['output_voxels'][0]]

    def post_process_semantic(self, pred_occ):
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        score, clses = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        return clses

    def simple_evaluation_semantic(self, pred, gt, img_metas):
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        gt = gt[:, 3].astype(np.int)
        unique_label = np.arange(16)
        hist = fast_hist_crop(pred, gt, unique_label)
        
        return hist
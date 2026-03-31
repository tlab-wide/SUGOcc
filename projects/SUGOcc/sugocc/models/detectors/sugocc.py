import torch
import torch.nn.functional as F
import collections 

from mmdet3d.models import Base3DSegmentor
from mmdet3d.structures import PointData
from mmengine.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from projects.SUGOcc.sugocc.utils import fast_hist_crop

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
                 downsample=16,
                 grid_config=None,
                 do_use_history=False,
                 dataset='semantickitti',
                 **kwargs):
        super().__init__(**kwargs)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.loss_norm = loss_norm
        self.nonempty_num = 0
        self.sample_num = 0
        self.dataset = dataset
        self.do_use_history = do_use_history
        self.num_frame = 1 if not do_use_history else 2
        self.history_stereo_feat = None
        self.history_img_metas = None
        self.history_img_inputs = None
        self.history_voxel_feat = None
        self.grid_config = grid_config
        self.downsample = downsample
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

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        # assert len(inputs) == 7 # nusc:7, semantickitti:10
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[:7]

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)    # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())   # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()     # (B, N_views, 4, 4)

        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def get_curr2adjsensor(self, curr_inputs, history_inputs, start_of_sequence):
        B, N, C, H, W = curr_inputs[0].shape

        sensor2egos, ego2globals, intrins, post_rots, post_trans = curr_inputs[1:6]
        sensor2egos_hist, ego2globals_hist, intrins_hist, post_rots_hist, post_trans_hist = history_inputs[1:6]
        sensor2egos = sensor2egos.view(B, 1, N, 4, 4).contiguous()
        ego2globals = ego2globals.view(B, 1, N, 4, 4).contiguous()
        sensor2egos_hist = sensor2egos_hist.view(B, 1, N, 4, 4).contiguous()
        ego2globals_hist = ego2globals_hist.view(B, 1, N, 4, 4).contiguous()

        if start_of_sequence.sum() > 0:
            sensor2egos_hist[start_of_sequence] = sensor2egos[start_of_sequence]
            ego2globals_hist[start_of_sequence] = ego2globals[start_of_sequence]
            intrins_hist[start_of_sequence] = intrins[start_of_sequence]
            post_rots_hist[start_of_sequence] = post_rots[start_of_sequence]
            post_trans_hist[start_of_sequence] = post_trans[start_of_sequence]

        sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
        sensor2egos_curr = sensor2egos_cv.double()
        ego2globals_curr = ego2globals_cv.double()
        sensor2egos_adj = sensor2egos_hist.double()
        ego2globals_adj = ego2globals_hist.double()
        curr2adjsensor = \
            torch.inverse(ego2globals_adj @ sensor2egos_adj) \
            @ ego2globals_curr @ sensor2egos_curr
        curr2adjsensor = curr2adjsensor.float().squeeze(1)

        return [history_inputs[0], sensor2egos_hist, ego2globals_hist, intrins_hist, post_rots_hist, post_trans_hist, *history_inputs[6:]], curr2adjsensor

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        stereo_feat=None
        _s = time.time()
        x = self.img_backbone(imgs)
        if isinstance(x, dict):
            img_feats = list(x.get('feats', x))
            for i in range(len(img_feats)):
                img_feats[i] = img_feats[i].contiguous()
        else:
            if stereo:
                stereo_feat = x[0]
                x = x[1:]
            img_feats = x

        if self.img_neck is not None:
            _s = time.time()
            fused_feats = self.img_neck(img_feats)
            if type(fused_feats) in [list, tuple]:
                fused_feats = fused_feats[0]
        _, output_dim, ouput_H, output_W = fused_feats.shape
        fused_feats = fused_feats.view(B, N, output_dim, ouput_H, output_W)
        
        return fused_feats, stereo_feat

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

    def extract_img_feat(self, img, img_metas=None):
        """Extract features of images."""

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        if self.dataset == 'nuscenes':
            img = self.prepare_inputs(img)
        x, stereo_feat = self.image_encoder(img[0], stereo=self.dataset == 'nuscenes')
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        if self.do_use_history:       
            if self.history_stereo_feat is None:
                self.history_stereo_feat = stereo_feat
                self.history_img_metas = img_metas
                self.history_img_inputs = img

            start_of_sequence = torch.BoolTensor([
                img_meta['flag'] != self.history_img_metas[i]['flag'] \
                    for i, img_meta in enumerate(img_metas)
            ])
            if start_of_sequence.sum() > 0:
                bs = img[0].shape[0]
                tem = stereo_feat.reshape(bs,stereo_feat.shape[0]//bs,*stereo_feat.shape[1:])[start_of_sequence].detach()
                self.history_stereo_feat=self.history_stereo_feat.reshape(bs,self.history_stereo_feat.shape[0]//bs,*self.history_stereo_feat.shape[1:])
                self.history_stereo_feat[start_of_sequence]=tem
                self.history_stereo_feat=self.history_stereo_feat.reshape(*stereo_feat.shape)
                
            prev_stereo_feat = self.history_stereo_feat
            prev_img = self.history_img_inputs
            prev_img, curr2adjsensor = self.get_curr2adjsensor(img, prev_img, start_of_sequence)
            self.history_stereo_feat = stereo_feat
            self.history_img_metas = img_metas
            self.history_img_inputs = img

            stereo_metas = dict(k2s_sensor=curr2adjsensor,
                intrins=prev_img[3],
                post_rots=prev_img[4],
                post_trans=prev_img[5],
            #  frustum=self.cv_frustum.to(stereo_feat.device),
                cv_downsample=4,
                downsample=self.downsample,
                grid_config=self.grid_config,
                cv_feat_list=[prev_stereo_feat, stereo_feat])
        else:
            stereo_metas = None
        
        if self.dataset == "semantickitti":
            # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
            rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
            
            mlp_input = self.img_view_transformer.get_mlp_input(rots, 
                                                                trans, 
                                                                intrins, 
                                                                post_rots, 
                                                                post_trans, 
                                                                bda)
        else:
            sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img[1:7]
            mlp_input = self.img_view_transformer.get_mlp_input_nus(sensor2egos, 
                                                                    ego2globals, 
                                                                    intrins, 
                                                                    post_rots, 
                                                                    post_trans, 
                                                                    bda)
        geo_inputs = [*img[1:7], mlp_input]
        x, depth, seg = self.img_view_transformer([x] + geo_inputs, img_metas=img_metas, stereo_metas=stereo_metas)


        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)

        x, voxel_seg_logits = self.bev_encoder(x)
        ## TODO fuse x with self.history_voxel_feat
        if type(x) is not list:
            x = [x]
        if self.do_use_history:
            self.history_voxel_feat = x[0].detach()
        return x, depth, seg, voxel_seg_logits
    
    def extract_feat(self, points, img, img_metas=None):
        """Extract features from images and points."""
        voxel_feats, depth, seg, voxel_seg_logits  = self.extract_img_feat(img, img_metas=img_metas)
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
            points, img=img_inputs, img_metas=img_metas)

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
        voxel_feats, depth, seg, voxel_seg_logits  = self.extract_feat(
            points_occ, img=img, img_metas=img_metas)
        
        if type(voxel_feats) is not list:
            voxel_feats = [voxel_feats]

        transform = img[1:] if img is not None else None
        output = self.pts_bbox_head.simple_test(
            voxel_feats=voxel_feats,
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
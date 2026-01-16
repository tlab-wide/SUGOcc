# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence
import mmcv
import numpy as np
import torch
import mmcv
import numpy as np
import torch.distributed as dist
import os
import shutil
import pickle
import time
import pdb
from torchmetrics.metric import Metric

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS

LABEL2CAT = {
    0: 'unlabeled', 
    1: 'car', 
    2: 'bicycle', 
    3: 'motorcycle',
    4: 'truck', 
    5: 'other-vehicle',
    6: 'person', 
    7: 'bicyclist', 
    8: 'motorcyclist', 
    9: 'road', 
    10: 'parking', 
    11: 'sidewalk',
    12: 'other-ground', 
    13: 'building', 
    14: 'fence', 
    15: 'vegetation', 
    16: 'trunk', 
    17: 'terrain',
    18: 'pole', 
    19: 'traffic-sign',
}

class SSCMetrics(Metric):
    def __init__(self, class_names=None, compute_on_step=False):
        # super().__init__(compute_on_step=compute_on_step)
        super().__init__()
        
        if class_names is None:
            class_names = [
                'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
                'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
                'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
                'pole', 'traffic-sign'
            ]
        
        self.class_names = class_names
        self.n_classes = len(class_names)
        
        self.add_state('tps', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        self.add_state('fps', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        self.add_state('fns', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        
        self.add_state('completion_tp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fn', default=torch.zeros(1), dist_reduce_fx='sum')
    
    def compute_single(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        # y_true = y_true * mask
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        # # evaluate semantic completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        
        ret = (tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy(), tp_sum.cpu().numpy(), fp_sum.cpu().numpy(), fn_sum.cpu().numpy())
        
        return ret
        
    def update(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn
        
        # # evaluate semantic completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum
    
    def compute(self):
        precision = self.completion_tp / (self.completion_tp + self.completion_fp)
        recall = self.completion_tp / (self.completion_tp + self.completion_fn)
        iou = self.completion_tp / \
                (self.completion_tp + self.completion_fp + self.completion_fn)
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        
        output = {
            "precision": precision,
            "recall": recall,
            "iou": iou.item(),
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": iou_ssc[1:].mean().item(),
        }
        
        return output

    def get_score_completion(self, predict, target, nonempty=None):
        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = torch.zeros_like(predict)
        b_true = torch.zeros_like(target)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = torch.sum((y_true == 1) & (y_pred == 1))
            fp = torch.sum((y_true != 1) & (y_pred == 1))
            fn = torch.sum((y_true == 1) & (y_pred != 1))
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        tp_sum = torch.zeros(_C).type_as(predict)
        fp_sum = torch.zeros(_C).type_as(predict)
        fn_sum = torch.zeros(_C).type_as(predict)

        for idx in range(_bs):
            y_true = target[idx]  # GT
            y_pred = predict[idx]
            
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                valid_mask = (nonempty_idx == 1) & (y_true != 255)
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]
            
            for j in range(_C):  # for each class
                tp = torch.sum((y_true == j) & (y_pred == j))
                fp = torch.sum((y_true != j) & (y_pred == j))
                fn = torch.sum((y_true == j) & (y_pred != j))
                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum

@METRICS.register_module()
class CustomOccMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super(CustomOccMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_pts_seg']
            pred_3d['pts_semantic_mask'] = pred_3d['pts_semantic_mask'].argmax(dim=1)
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, pred_3d))

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, 'results')
        mmcv.mkdir_or_exist(submission_prefix)
        ignore_index = self.dataset_meta['ignore_index']
        # need to map network output to original label idx
        cat2label = np.zeros(len(self.dataset_meta['label2cat'])).astype(
            np.int64)
        for original_label, output_idx in self.dataset_meta['label2cat'].items(
        ):
            if output_idx != ignore_index:
                cat2label[output_idx] = original_label

        for i, (eval_ann, result) in enumerate(results):
            sample_idx = eval_ann['point_cloud']['lidar_idx']
            pred_sem_mask = result['semantic_mask'].numpy().astype(np.int64)
            pred_label = cat2label[pred_sem_mask]
            curr_file = f'{submission_prefix}/{sample_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        ssc_metric = SSCMetrics()
        ssc_results = []
        for eval_ann, single_pred_results in results:
            ssc_results_i = ssc_metric.compute_single(
                y_pred=single_pred_results['pts_semantic_mask'].long(), y_true=eval_ann['gt_occ'].unsqueeze(0).long())
            ssc_results.append(ssc_results_i)
        completion_tp = sum([x[0] for x in ssc_results])
        completion_fp = sum([x[1] for x in ssc_results])
        completion_fn = sum([x[2] for x in ssc_results])
        
        tps = sum([x[3] for x in ssc_results])
        fps = sum([x[4] for x in ssc_results])
        fns = sum([x[5] for x in ssc_results])
        
        precision = completion_tp / (completion_tp + completion_fp)
        recall = completion_tp / (completion_tp + completion_fn)
        iou = completion_tp / \
                (completion_tp + completion_fp + completion_fn)
        iou_ssc = tps / (tps + fps + fns + 1e-5)
        
        class_ssc_iou = iou_ssc.tolist()
        res_dic = {
            "SC_Precision": precision,
            "SC_Recall": recall,
            "SC_IoU": iou,
            "SSC_mIoU": iou_ssc[1:].mean(),
        }
        class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)
        
        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']
        
        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)

        return eval_results

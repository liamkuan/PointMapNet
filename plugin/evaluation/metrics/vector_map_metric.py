# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
import mmengine
import torch
import numpy as np
from mmengine import print_log
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmdet3d.registry import METRICS
from plugin.datasets.map_utils.mean_ap import eval_map, format_res_gt_by_classes
from plugin import eval_chamfer


@METRICS.register_module()
class VectorMapMetric(BaseMetric):

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 prefix: Optional[str] = None,
                 pc_range=[-30, -15, -5.0, 30, 15, 3.0],
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None,
                 eval_cfg=None) -> None:
        self.default_prefix = 'out/evaluation'
        super(VectorMapMetric, self).__init__(collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=True,
                use_lidar=False,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.pc_range = pc_range
        self.eval_cfg = eval_cfg
        self.eval_cfg['ann_file'] = ann_file
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample, gt_labels, gt_lines in zip(data_samples, data_batch['inputs']['gt_labels_3d'],
                                                    data_batch['inputs']['gt_bboxes_3d']):
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['sample_idx'] = data_sample['sample_idx']
            result['vectors'] = []
            for line, label in zip(pred_3d['pred_lines'], pred_3d['pred_logits']):
                label = torch.sigmoid(label)
                _, type = torch.max(label, dim=0)
                type = int(type)
                confidence_level = float(label[type])

                line[..., 0] = line[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                line[..., 1] = line[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]

                # if confidence_level > 0.5:
                result['vectors'].append({
                    'pts': line.numpy(),
                    'pts_num': 20,
                    'type': type,
                    'confidence_level': confidence_level
                })
            result['groundTruth'] = []
            for line, label in zip(gt_lines.fixed_num_sampled_points, gt_labels):
                result['groundTruth'].append({
                    'pts': line.cpu().numpy(),
                    'pts_num': 20,
                    'type': int(label),
                    'confidence_level': 1,
                })
            result['pred_instances_3d'] = pred_3d
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        # load annotations
        result_path = self.format_results(results)

        self.eval_cfg['result_path'] = result_path

        mean_ap = eval_chamfer(self.eval_cfg, update=True, logger=logger)

        result_dict = {
            'mAP': mean_ap,
        }

        print('VectormapNet Evaluation Results:')
        print(result_dict)

        return result_dict

    def format_results(self, results):
        meta = self.modality
        submissions = {
            'meta': meta,
            'results': {},
            "groundTruth": {},  # for validation
        }

        for case in mmengine.track_iter_progress(results):
            '''
                vectorized_line {
                    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
                    "pts_num":           <int>,            -- Number of points in this line.
                    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
                    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
                }
            '''
            vector_lines = case['vectors']
            submissions['results'][case['sample_idx']] = {}
            submissions['results'][case['sample_idx']]['vectors'] = vector_lines

            if 'groundTruth' in case:
                submissions['groundTruth'][case['sample_idx']] = {}
                vector_lines = case['groundTruth']
                submissions['groundTruth'][case['sample_idx']]['vectors'] = vector_lines

        # Use pickle format to minimize submission file size.
        print('Done!')
        mmengine.mkdir_or_exist(self.prefix)
        res_path = os.path.join(self.prefix, 'results_vectormap.pkl')
        mmengine.dump(submissions, res_path)

        return res_path


@METRICS.register_module()
class VectorMapMetricV1(BaseMetric):

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality=None,
                 map_classes=None,
                 fixed_num=20,
                 pc_range=None,
                 prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None,
                 eval_cfg=None) -> None:
        self.default_prefix = 'out/evaluation'
        super(VectorMapMetricV1, self).__init__(collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=True,
                use_lidar=False,
            )
        if map_classes is None:
            map_classes = ['divider', 'ped_crossing', 'boundary']
        if pc_range is None:
            pc_range = [-32., -16., -3.0, 32., 16., 3.0]
        self.map_classes = map_classes
        self.fixed_num = fixed_num
        self.eval_use_same_gt_sample_num_flag = True
        self.pc_range = pc_range
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.eval_cfg = eval_cfg
        self.eval_cfg['ann_file'] = ann_file
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample, gt_labels, gt_bboxes in zip(data_samples, data_batch['inputs']['gt_labels_3d'],
                                                     data_batch['inputs']['gt_bboxes_3d']):
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['sample_idx'] = data_sample['sample_idx']
            result['vectors'] = []
            for line, label, score in zip(pred_3d['pts'].numpy(), pred_3d['labels'].numpy(),
                                          pred_3d['scores'].numpy()):
                result['vectors'].append({
                    'pts': line.tolist(),
                    'pts_num': len(line),
                    'cls_name': self.map_classes[label],
                    'type': int(label),
                    'confidence_level': float(score)
                })
            result['groundTruth'] = []
            gt_labels = gt_labels.cpu().numpy()
            gt_vecs = gt_bboxes.instance_list
            for vec, label in zip(gt_vecs, gt_labels):
                result['groundTruth'].append({
                    'pts': np.array(list(vec.coords))[..., :2].tolist(),
                    'pts_num': len(list(vec.coords)),
                    'cls_name': self.map_classes[label],
                    'type': int(label),
                })
            # result['pred_instances_3d'] = pred_3d
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        result_path = self.format_results(results)

        result_path = os.path.abspath(result_path)
        detail = dict()

        print_log('Formating results & gts by classes')
        submissions = mmengine.load(result_path)
        gen_results = submissions['results']
        annotations = submissions['groundTruth']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     gen_results,
                                                     annotations,
                                                     cls_names=self.map_classes,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range)

        metric = 'chamfer'
        print_log('-*' * 10 + f'use metric:{metric}' + '-*' * 10)

        thresholds = [0.5, 1.0, 1.5]

        cls_aps = np.zeros((len(thresholds), len(self.map_classes)))

        for i, thr in enumerate(thresholds):
            print('-*' * 10 + f'threshhold:{thr}' + '-*' * 10)
            mAP, cls_ap = eval_map(
                gen_results,
                annotations,
                cls_gens,
                cls_gts,
                threshold=thr,
                cls_names=self.map_classes,
                logger=MMLogger.get_current_instance(),
                num_pred_pts_per_instance=self.fixed_num,
                pc_range=self.pc_range,
                metric=metric)
            for j in range(len(self.map_classes)):
                cls_aps[i, j] = cls_ap[j]['ap']

        for i, name in enumerate(self.map_classes):
            print('{}: {}'.format(name, cls_aps.mean(0)[i]))
            detail['NuscMap_{}/{}_AP'.format(metric, name)] = cls_aps.mean(0)[i]
        print('map: {}'.format(cls_aps.mean(0).mean()))
        detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

        for i, name in enumerate(self.map_classes):
            for j, thr in enumerate(thresholds):
                if metric == 'chamfer':
                    detail['NuscMap_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]
                elif metric == 'iou':
                    if thr == 0.5 or thr == 0.75:
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]

        return detail

    def format_results(self, results):
        meta = self.modality
        submissions = {
            'meta': meta,
            'results': [],
            "groundTruth": [],  # for validation
        }

        for case in mmengine.track_iter_progress(results):
            '''
                vectorized_line {
                    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
                    "pts_num":           <int>,            -- Number of points in this line.
                    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
                    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
                }
            '''
            vector_lines = case['vectors']
            submissions['results'].append({
                'sample_token': case['sample_idx'],
                'vectors': vector_lines
            })

            if 'groundTruth' in case:
                submissions['groundTruth'].append({
                    'sample_token': case['sample_idx'],
                    'vectors': case['groundTruth']
                })

        # Use pickle format to minimize submission file size.
        print('Done!')
        mmengine.mkdir_or_exist(self.prefix)
        res_path = os.path.join(self.prefix, 'results_vectormap.pkl')
        mmengine.dump(submissions, res_path)

        return res_path

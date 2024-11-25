import copy
from typing import List
import mmcv
import os
from os import path as osp
import mmengine
import numpy as np
from mmdet3d.registry import DATASETS
from .evaluation.precision_recall.average_precision_gen import eval_map
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString
import json
from .map_utils.mean_ap import format_res_gt_by_classes
from .base_dataset import BaseMapDataset


@DATASETS.register_module()
class CustomAV2LocalMapDataset(BaseMapDataset):
    r"""NuScenes Dataset.

    This datset add static map elements
    """
    MAPCLASSES = ('divider',)

    def __init__(self,
                 map_ann_file=None,
                 queue_length=1,
                 code_size=2,
                 bev_size=(200, 200),
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False,
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.map_ann_file = map_ann_file

        self.code_size = code_size
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.is_vis_on_test = False

    def load_data_list(self) -> List[dict]:
        annotations = mmengine.load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']
        self.id2map = annotations['id2map']
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)
        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                data_list.append(data_info)
            elif isinstance(data_info, list):
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')

        return class_names

    def prepare_data(self, index: int):
        ori_input_dict = copy.deepcopy(self.get_data_info(index))
        if ori_input_dict is None:
            return None
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=ori_input_dict['log_id'] + ori_input_dict['timestamp'],
            timestamp=ori_input_dict['timestamp'],
            e2g_translation=ori_input_dict['e2g_translation'],
            e2g_rotation=ori_input_dict['e2g_rotation'],
            log_id=ori_input_dict['log_id'],
            scene_token=ori_input_dict['log_id'],
            map_elements=self.id2map[ori_input_dict['log_id']]
        )
        if self.modality['use_camera']:
            image_paths = []
            cam_intrinsics = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_types = []
            for cam_type, cam_info in ori_input_dict['cams'].items():
                if cam_type == "ring_front_center":
                    continue
                image_paths.append(cam_info['img_path'])
                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["intrinsics"]
                # input_dict["camera_intrinsics"].append(camera_intrinsics)

                # ego2img, ego = lidar
                lidar2cam_rt = cam_info['extrinsics']
                intrinsic = cam_info['intrinsics']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)
                cam_intrinsics.append(viewpad)
                cam_types.append(cam_type)

            input_dict.update(
                dict(
                    img_filenames=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    cam_type=cam_types,
                ))

        if not self.test_mode:
            # annos = self.get_ann_info(index)
            input_dict['ann_info'] = dict()

        translation = input_dict['e2g_translation']
        can_bus = np.ones(18)
        # can_bus.extend(translation.tolist())
        can_bus[:3] = translation
        rotation = Quaternion._from_matrix(input_dict['e2g_rotation'])
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus

        example = self.pipeline(input_dict)

        if len(example['inputs']['gt_labels_3d']) == 0:
            return None

        return example

    def _format_gt(self):
        gt_annos = []
        # import pdb;pdb.set_trace()
        print('Start to convert gt map format...')
        assert self.map_ann_file is not None
        if (not os.path.exists(self.map_ann_file)):
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['token']
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, self.data_infos[sample_id])
                gt_labels = gt_sample_dict['gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['gt_bboxes_3d'].data.instance_list
                # import pdb;pdb.set_trace()
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords))[:, :self.code_size],
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors'] = gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(nusc_submissions, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        assert self.map_ann_file is not None
        pred_annos = []
        mapped_class_names = self.MAPCLASSES
        # import pdb;pdb.set_trace()
        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['token']
            pred_anno['sample_token'] = sample_token
            pred_vec_list = []
            for i, vec in enumerate(vecs):
                name = mapped_class_names[vec['label']]
                anno = dict(
                    # sample_token=sample_token,
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=name,
                    type=vec['label'],
                    confidence_level=vec['score'])
                pred_vec_list.append(anno)
                # annos.append(nusc_anno)
            # nusc_annos[sample_token] = annos
            pred_anno['vectors'] = pred_vec_list
            pred_annos.append(pred_anno)

        if not os.path.exists(self.map_ann_file):
            self._format_gt()
        else:
            print(f'{self.map_ann_file} exist, not update')
        # with open(self.map_ann_file,'r') as f:
        #     GT_anns = json.load(f)
        # gt_annos = GT_anns['GTs']
        nusc_submissions = {
            'meta': self.modality,
            'results': pred_annos,
            # 'GTs': gt_annos
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'nuscmap_results.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def to_gt_vectors(self,
                      gt_dict):
        # import pdb;pdb.set_trace()
        gt_labels = gt_dict['gt_labels_3d'].data
        gt_instances = gt_dict['gt_bboxes_3d'].data.instance_list

        gt_vectors = []

        for gt_instance, gt_label in zip(gt_instances, gt_labels):
            pts, pts_num = sample_pts_from_line(gt_instance, patch_size=self.patch_size)
            gt_vectors.append({
                'pts': pts,
                'pts_num': pts_num,
                'type': int(gt_label)
            })
        vector_num_list = {}
        for i in range(self.NUM_MAPCLASSES):
            vector_num_list[i] = []
        for vec in gt_vectors:
            if vector['pts_num'] >= 2:
                vector_num_list[vector['type']].append(
                    (LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))
        return gt_vectors

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        result_path = osp.abspath(result_path)
        # import pdb;pdb.set_trace()
        detail = dict()

        print('Formating results & gts by classes')
        with open(result_path, 'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']
        with open(self.map_ann_file, 'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     gen_results,
                                                     annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            print('-*' * 10 + f'use metric:{metric}' + '-*' * 10)

            if metric == 'chamfer':
                thresholds = [0.5, 1.0, 1.5]
            elif metric == 'iou':
                thresholds = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                print('-*' * 10 + f'threshhold:{thr}' + '-*' * 10)
                mAP, cls_ap = eval_map(
                    gen_results,
                    annotations,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.MAPCLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['AV2Map_{}/{}_AP'.format(metric, name)] = cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['AV2Map_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['AV2Map_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['AV2Map_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]

        return detail

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict


def output_to_vecs(detection):
    box3d = detection['boxes_3d'].numpy()
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    pts = detection['pts_3d'].numpy()

    vec_list = []
    # import pdb;pdb.set_trace()
    for i in range(box3d.shape[0]):
        vec = dict(
            bbox=box3d[i],  # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list


def sample_pts_from_line(line,
                         fixed_num=-1,
                         sample_dist=1,
                         normalize=False,
                         patch_size=None,
                         padding=False,
                         num_samples=250, ):
    if fixed_num < 0:
        distances = np.arange(0, line.length, sample_dist)
        if line.has_z:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   3)
        else:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)
    else:
        # fixed number of points, so distance is line.length / fixed_num
        distances = np.linspace(0, line.length, fixed_num)
        if line.has_z:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   3)
        else:
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)

    if normalize:
        sampled_points[:, :2] = sampled_points[:, :2] / np.array([patch_size[1], patch_size[0]])

    num_valid = len(sampled_points)

    if not padding or fixed_num > 0:
        # fixed num sample can return now!
        return sampled_points, num_valid

    # fixed distance sampling need padding!
    num_valid = len(sampled_points)

    if fixed_num < 0:
        if num_valid < num_samples:
            padding = np.zeros((num_samples - len(sampled_points), sampled_points.shape[-1]))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        else:
            sampled_points = sampled_points[:num_samples, :]
            num_valid = num_samples

        if normalize:
            sampled_points[:, :2] = sampled_points[:, :2] / np.array([patch_size[1], patch_size[0]])
            num_valid = len(sampled_points)

    return sampled_points[:, :2], num_valid
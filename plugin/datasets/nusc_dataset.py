import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import to_tensor
from mmdet3d.registry import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

from .base_dataset import BaseMapDataset
from .evaluation.precision_recall.average_precision_gen import eval_chamfer


def add_rotation_noise(extrinsics, std=0.01, mean=0.0):
    #n = extrinsics.shape[0]
    noise_angle = torch.normal(mean, std=std, size=(3,))
    # extrinsics[:, 0:3, 0:3] *= (1 + noise)
    sin_noise = torch.sin(noise_angle)
    cos_noise = torch.cos(noise_angle)
    rotation_matrix = torch.eye(4).view(4, 4)
    #  rotation_matrix[]
    rotation_matrix_x = rotation_matrix.clone()
    rotation_matrix_x[1, 1] = cos_noise[0]
    rotation_matrix_x[1, 2] = sin_noise[0]
    rotation_matrix_x[2, 1] = -sin_noise[0]
    rotation_matrix_x[2, 2] = cos_noise[0]

    rotation_matrix_y = rotation_matrix.clone()
    rotation_matrix_y[0, 0] = cos_noise[1]
    rotation_matrix_y[0, 2] = -sin_noise[1]
    rotation_matrix_y[2, 0] = sin_noise[1]
    rotation_matrix_y[2, 2] = cos_noise[1]

    rotation_matrix_z = rotation_matrix.clone()
    rotation_matrix_z[0, 0] = cos_noise[2]
    rotation_matrix_z[0, 1] = sin_noise[2]
    rotation_matrix_z[1, 0] = -sin_noise[2]
    rotation_matrix_z[1, 1] = cos_noise[2]

    rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

    rotation = torch.from_numpy(extrinsics.astype(np.float32))
    rotation[:3, -1] = 0.0
    # import pdb;pdb.set_trace()
    rotation = rotation_matrix @ rotation
    extrinsics[:3, :3] = rotation[:3, :3].numpy()
    return extrinsics


def add_translation_noise(extrinsics, std=0.01, mean=0.0):
    # n = extrinsics.shape[0]
    noise = torch.normal(mean, std=std, size=(3,))
    extrinsics[0:3, -1] += noise.numpy()
    return extrinsics


@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    def __init__(self,
                 ann_file,
                 data_root,
                 modality=dict(
                     use_camera=True,
                     use_lidar=False,
                     use_radar=False,
                     use_map=True,
                     use_external=False,
                 ),
                 pipeline=None,
                 coord_dim=3,
                 work_dir=None,
                 noise='None',
                 noise_std=0,
                 eval_cfg: dict = dict(),
                 **kwargs,
                 ):
        super().__init__(
            ann_file=ann_file,
            modality=modality,
            data_root=data_root,
            pipeline=pipeline,
            **kwargs,
        )
        self.coord_dim = coord_dim
        self.eval_cfg = eval_cfg
        # dummy flag to fit with mmdet
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.work_dir = work_dir
        self.noise = noise
        self.noise_std = noise_std

    def prepare_data(self, index: int):
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = copy.deepcopy(self.get_data_info(index))
        if ori_input_dict is None:
            return None
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=ori_input_dict['token'],
            # pts_filename=ori_input_dict['lidar_path'],
            lidar_points={"lidar_path": ori_input_dict['lidar_path']},
            # sweeps=ori_input_dict['sweeps'],
            ego2global_translation=ori_input_dict['ego2global_translation'],
            ego2global_rotation=ori_input_dict['ego2global_rotation'],
            lidar2ego_translation=ori_input_dict['lidar2ego_translation'],
            lidar2ego_rotation=ori_input_dict['lidar2ego_rotation'],
            prev_idx=ori_input_dict['prev'],
            next_idx=ori_input_dict['next'],
            scene_token=ori_input_dict['scene_token'],
            can_bus=ori_input_dict['can_bus'],
            frame_idx=ori_input_dict['frame_idx'],
            timestamp=ori_input_dict['timestamp'],
            map_location=ori_input_dict['map_location'],
        )
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float64)
        lidar2ego[:3, :3] = Quaternion(ori_input_dict["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = ori_input_dict["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            for cam_type, cam_info in ori_input_dict['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                lidar2cam_rt_t = lidar2cam_rt.T

                if self.noise == 'rotation':
                    lidar2cam_rt_t = add_rotation_noise(lidar2cam_rt_t, std=self.noise_std)
                elif self.noise == 'translation':
                    lidar2cam_rt_t = add_translation_noise(
                        lidar2cam_rt_t, std=self.noise_std)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt_t)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt_t)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    cam_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    img_filenames=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if self.modality["use_radar"]:
            input_dict['radars'] = ori_input_dict['radars']
        # rotation = Quaternion(input_dict['ego2global_rotation'])
        # translation = input_dict['ego2global_translation']
        # can_bus = input_dict['can_bus']
        # can_bus[:3] = translation
        # can_bus[3:7] = rotation
        # patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # if patch_angle < 0:
        #     patch_angle += 360
        # can_bus[-2] = patch_angle / 180 * np.pi
        # can_bus[-1] = patch_angle

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = input_dict['ego2global_translation']
        lidar2global = ego2global @ lidar2ego
        input_dict['lidar2global'] = lidar2global

        example = self.pipeline(input_dict)

        if len(example['inputs']['gt_labels_3d']) == 0:
            return None
        return example

    def format_results(self, results, name, prefix=None, patch_size=(60, 30), origin=(0, 0)):

        meta = self.modality
        submissions = {
            'meta': meta,
            'results': {},
            "groundTruth": {},  # for validation
        }
        patch_size = np.array(patch_size)
        origin = np.array(origin)

        for case in mmcv.track_iter_progress(results):
            '''
                vectorized_line {
                    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
                    "pts_num":           <int>,            -- Number of points in this line.
                    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
                    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
                }
            '''

            if case is None:
                continue

            vector_lines = []
            for i in range(case['nline']):
                vector = case['lines'][i] * patch_size + origin
                vector_lines.append({
                    'pts': vector,
                    'pts_num': len(case['lines'][i]),
                    'type': case['labels'][i],
                    'confidence_level': case['scores'][i],
                })
                submissions['results'][case['token']] = {}
                submissions['results'][case['token']]['vectors'] = vector_lines

            if 'groundTruth' in case:

                submissions['groundTruth'][case['token']] = {}
                vector_lines = []
                for i in range(case['groundTruth']['nline']):
                    line = case['groundTruth']['lines'][i] * \
                           patch_size + origin

                    vector_lines.append({
                        'pts': line,
                        'pts_num': len(case['groundTruth']['lines'][i]),
                        'type': case['groundTruth']['labels'][i],
                        'confidence_level': 1.,
                    })
                submissions['groundTruth'][case['token']
                ]['vectors'] = vector_lines

        # Use pickle format to minimize submission file size.
        print('Done!')
        mmcv.mkdir_or_exist(prefix)
        res_path = os.path.join(prefix, '{}.pkl'.format(name))
        mmcv.dump(submissions, res_path)

        return res_path

    def evaluate(self,
                 results,
                 logger=None,
                 name=None,
                 **kwargs):
        '''
        Args:
            results (list[Tensor]): List of results.
            eval_cfg (Dict): Config of test dataset.
            output_format (str): Model output format, should be either 'raster' or 'vector'.

        Returns:
            dict: Evaluation results.
        '''

        print('len of the results', len(results))
        name = 'results_nuscence' if name is None else name
        result_path = self.format_results(
            results, name, prefix=self.work_dir, patch_size=self.eval_cfg.patch_size, origin=self.eval_cfg.origin)

        self.eval_cfg.evaluation_cfg['result_path'] = result_path
        self.eval_cfg.evaluation_cfg['ann_file'] = self.ann_file

        mean_ap = eval_chamfer(
            self.eval_cfg.evaluation_cfg, update=True, logger=logger)

        result_dict = {
            'mAP': mean_ap,
        }

        print('VectormapNet Evaluation Results:')
        print(result_dict)

        return result_dict

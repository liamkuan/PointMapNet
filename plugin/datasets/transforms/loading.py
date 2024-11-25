import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List

import mmcv
import mmengine
import numpy as np
import torch
from av2.map.drivable_area import DrivableArea
from av2.map.lane_segment import LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from av2.map.pedestrian_crossing import PedestrianCrossing
from mmdet3d.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform
import av2.utils.raster as raster_utils
from mmengine import get
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion


@TRANSFORMS.register_module()
class LoadMultiViewImagesFromFiles(BaseTransform):

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filenames']
        img_bytes = [get(name, backend_args=None) for name in filename]
        img = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in img]
        results['ori_shape'] = [i.shape for i in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [i.shape for i in img]
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__} (to_float32={self.to_float32}, ' \
               f"color_type='{self.color_type}')"


@TRANSFORMS.register_module()
class AV2LoadVectorMap(BaseTransform):

    def __init__(self,
                 dim=3,
                 map_range=[-60, -30, 60, 30],
                 class2label={
                     'ped_crossing': 0,
                     'divider': 1,
                     'others': -1,
                     # 'centerline': 3,
                 }):
        self.dim = dim
        self.class2label = class2label
        self.map_range = map_range

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        map_data_path = results['map_data_path']
        map_dict = mmengine.load(map_data_path)
        # if 'drivable_boundary' not in self.class2label.keys():
        #     new_lines = []
        #     new_labels = []
        #     for line, label in zip(map_dict['lines'], map_dict['labels']):
        #         if int(label) < 2:
        #             new_lines.append(line)
        #             new_labels.append(label)
        #     map_dict['lines'] = new_lines
        #     map_dict['labels'] = torch.stack(new_labels, dim=0)
        results['target'] = map_dict
        results['maps'] = map_dict['maps']
        del results['target']['maps']
        return results


@TRANSFORMS.register_module()
class AV2LoadRasterizeMap(BaseTransform):

    def __init__(self, densities=1, map_range=[-60, -30, 60, 30]):
        self.densities = densities
        self.map_range = map_range

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        map_path = results['map_path']
        avm = ArgoverseStaticMap.from_map_dir(Path(map_path).absolute(), build_raster=True)
        ego2global = results['ego2global']
        lane_segments = avm.get_scenario_lane_segments()
        ped_crossings = avm.get_scenario_ped_crossings()
        drivable_areas = avm.get_scenario_vector_drivable_areas()
        layers = [self._get_mask(lane_segments, ego2global), self._get_mask(ped_crossings, ego2global),
                  self._get_mask(drivable_areas, ego2global)]
        results['maps'] = torch.stack(layers, dim=0)
        return results

    def _get_mask(self, areas, ego2global):
        polygons = []
        for da in areas:
            if type(da) is LaneSegment:
                points = torch.cat(
                    [torch.from_numpy(da.polygon_boundary), torch.ones([da.polygon_boundary.shape[0], 1])],
                    dim=-1).unsqueeze(-1)
            elif type(da) is PedestrianCrossing:
                points = torch.cat(
                    [torch.from_numpy(da.polygon), torch.ones([da.polygon.shape[0], 1])],
                    dim=-1).unsqueeze(-1)
            elif type(da) is DrivableArea:
                points = torch.cat([torch.from_numpy(da.xyz), torch.ones([da.xyz.shape[0], 1])], dim=-1).unsqueeze(-1)
            else:
                return None
            points = points.to(torch.float32)
            polygon = torch.inverse(ego2global) @ points
            polygon = polygon.squeeze(-1)[:, :2]
            polygon = polygon + torch.tensor([self.map_range[2], self.map_range[3]])
            polygon = polygon * self.densities
            polygons.append(polygon.numpy())
        w = self.map_range[3] - self.map_range[1]
        h = self.map_range[2] - self.map_range[0]
        da_array = raster_utils.get_mask_from_polygons(polygons, w * self.densities, h * self.densities)
        return torch.as_tensor(da_array, dtype=torch.float32)


@TRANSFORMS.register_module()
class LoadRadarPoints(BaseTransform):
    """Load Radar Points.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:lse.
        use_color (bool): Whether to use color features. Defaults to False.
    """
    def _load_points(self, radar_info: dict) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            radar_info (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        points = RadarPointCloud.from_file(radar_info['data_path']).points.transpose(1, 0).astype(np.float32)
        points = points[:, :3]
        points = torch.from_numpy(points)
        points = torch.cat((points, torch.ones_like(points[..., :1])), -1).unsqueeze(-1)

        radar2lidar = np.eye(4)
        radar2lidar[:3, :3] = radar_info['sensor2lidar_rotation']
        radar2lidar[:3, 3] = radar_info['sensor2lidar_translation']
        radar2lidar = radar2lidar.astype(np.float32)
        lidar_points = torch.matmul(torch.from_numpy(radar2lidar), points)

        return lidar_points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        for i, img in enumerate(results['img']):
            results['img'][i] = np.pad(img, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=(0,))

        lidar_points = [self._load_points(item) for _, item in results["radars"].items()]
        lidar_points = torch.cat(lidar_points, dim=0)
        
        lidar2img = torch.as_tensor(
            np.array(results["lidar2img"]), dtype=torch.float32
        )

        num_cam = lidar2img.shape[0]
        lidar_points = lidar_points.unsqueeze(0).repeat(num_cam, 1, 1, 1)

        lidar2img = lidar2img.view(num_cam, 1, 4, 4).repeat(1, lidar_points.shape[1], 1, 1)
        # 从LIDAR坐标到相机坐标的转换
        cam_points = torch.matmul(lidar2img, lidar_points).squeeze(-1)
        eps = 1e-5
        # 第三维代表缩放系数，除以缩放系数得到真实的uv
        masks = cam_points[..., 2:3] > eps
        cam_points[..., 0:2] = cam_points[..., 0:2] / torch.maximum(
            cam_points[..., 2:3],
            torch.ones_like(cam_points[..., 2:3]) * eps,
        )
        masks = (
                masks
                & (cam_points[..., 1:2] > 0.0)
                & (cam_points[..., 1:2] < results["img_shape"][0][0])
                & (cam_points[..., 0:1] < results["img_shape"][0][1])
                & (cam_points[..., 0:1] > 0.0)
        )
        masks = masks.squeeze(-1)
        for mask, cam_point, img in zip(masks, cam_points, results['img']):
            index = torch.nonzero(mask).squeeze(-1)
            cam_point = torch.index_select(cam_point, dim=0, index=index)
            for c_point in cam_point:
                x = int(c_point[1].item())
                y = int(c_point[0].item())
                img[x, y, 3] = c_point[2]
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

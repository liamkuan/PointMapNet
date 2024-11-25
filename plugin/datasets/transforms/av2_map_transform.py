from os import path as osp
from typing import Dict, Optional, Union, Tuple, List

import torch
import numpy as np
from mmcv import to_tensor, BaseTransform
from mmdet3d.registry import TRANSFORMS
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from pathlib import Path
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3


class LiDARInstanceLines(object):
    def __init__(self,
                 instance_line_list,
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,
                 patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
            dtype=torch.float32)
        instance_se_points_tensor[:, 0] = torch.clamp(instance_se_points_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 1] = torch.clamp(instance_se_points_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_se_points_tensor[:, 2] = torch.clamp(instance_se_points_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 3] = torch.clamp(instance_se_points_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(
            dtype=torch.float32)
        instance_bbox_tensor[:, 0] = torch.clamp(instance_bbox_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 1] = torch.clamp(instance_bbox_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_bbox_tensor[:, 2] = torch.clamp(instance_bbox_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 3] = torch.clamp(instance_bbox_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            # instance_array = np.array(list(instance.coords))
            # interpolated_instance = interp_utils.interp_arc(t=self.fixed_num, points=instance_array)
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances])
            if instance.has_z:
                sampled_points = sampled_points.reshape(-1, 3)
            else:
                sampled_points = sampled_points.reshape(-1, 2)
            # import pdb;pdb.set_trace()
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
            dtype=torch.float32)

        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """
        return torch.Tensor([N,fixed_num,3]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            if instance.has_z:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 3)
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
            dtype=torch.float32)

        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        # instance_points_tensor = instance_points_tensor if is_3d else instance_points_tensor[:, :, :2]
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_torch(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            # distances = np.linspace(0, instance.length, self.fixed_num)
            # sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0, 2, 1)
            sampled_pts = torch.nn.functional.interpolate(poly_pts, size=(self.fixed_num), mode='linear',
                                                          align_corners=True)
            sampled_pts = sampled_pts.permute(0, 2, 1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list, dim=0)
        instance_points_tensor = instance_points_tensor.to(
            dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num - shift_pts.shape[0], fixed_num, shift_pts.shape[-1]],
                                     self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num - shift_pts.shape[0], pts_num, shift_pts.shape[-1]], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                               coords_num)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(multi_shifts_pts_tensor[:, :, 0], min=-self.max_x,
                                                           max=self.max_x)
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(multi_shifts_pts_tensor[:, :, 1], min=-self.max_y,
                                                           max=self.max_y)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num - multi_shifts_pts_tensor.shape[0], self.fixed_num,
                                      multi_shifts_pts_tensor.shape[-1]], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                               coords_num)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                               coords_num)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape
            # import pdb;pdb.set_trace()
            if shifts_num > 2 * final_shift_num:
                index = np.random.choice(shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index + shift_num]
                multi_shifts_pts = np.concatenate((flip0_shifts_pts, flip1_shifts_pts), axis=0)

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(multi_shifts_pts_tensor[:, :, 0], min=-self.max_x,
                                                           max=self.max_x)
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(multi_shifts_pts_tensor[:, :, 1], min=-self.max_y,
                                                           max=self.max_y)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < 2 * final_shift_num:
                padding = torch.full([final_shift_num * 2 - multi_shifts_pts_tensor.shape[0], self.fixed_num,
                                      multi_shifts_pts_tensor.shape[-1]], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(flip_pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num * 2, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num * 2 - shift_pts.shape[0], pts_num, shift_pts.shape[-1]],
                                     self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num - shift_pts.shape[0], fixed_num, shift_pts.shape[-1]],
                                     self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor


@TRANSFORMS.register_module(force=True)
class VectorizedAV2LocalMap(BaseTransform):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'divider': 0,
        'ped_crossing': 1,
        'boundary': 2,
        'others': -1
    }

    def __init__(self,
                 data_root,
                 patch_size,
                 test_mode=False,
                 map_classes=['divider', 'ped_crossing', 'boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000, ):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()
        # self.data_root = dataroot
        self.test_mode = test_mode
        if self.test_mode:
            self.data_root = osp.join(data_root, "val")
        else:
            self.data_root = osp.join(data_root, "train")

        self.loader = AV2SensorDataLoader(data_dir=Path(data_root), labels_dir=Path(data_root))

        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        location = results['log_id']
        e2g_translation = results['e2g_translation']
        e2g_rotation = results['e2g_rotation']
        map_elements = results['map_elements']
        anns_results = self.gen_vectorized_samples(location, map_elements, e2g_translation, e2g_rotation)
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                gt_vecs_pts_loc = gt_vecs_pts_loc
        results['gt_labels_3d'] = gt_vecs_label
        results['gt_bboxes_3d'] = gt_vecs_pts_loc
        return results

    def gen_vectorized_samples(self, location, map_elements, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        av2 lidar2global the same as ego2global
        location the same as log_id
        '''
        # avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion._from_matrix(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # import pdb;pdb.set_trace()
        vectors = []
        city_SE2_ego = SE3(lidar2global_rotation, lidar2global_translation)
        ego_SE3_city = city_SE2_ego.inverse()
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_divider_geom(patch_box, patch_angle, map_elements[vec_class], ego_SE3_city)
                line_instances_list = self.line_geoms_to_instances(line_geom)
                for divider in line_instances_list:
                    vectors.append((divider, self.CLASS2LABEL.get('divider', -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_ped_geom(patch_box, patch_angle, map_elements[vec_class], ego_SE3_city)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_boundary_geom(patch_box, patch_angle, map_elements[vec_class], ego_SE3_city)
                poly_bound_list = self.bound_poly_geoms_to_instances(polygon_geom)
                for bound in poly_bound_list:
                    vectors.append((bound, self.CLASS2LABEL.get('boundary', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        # filter out -1
        filtered_vectors = []
        gt_pts_loc_3d = []
        gt_pts_num_3d = []
        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)
        # import pdb;pdb.set_trace()
        gt_instance = LiDARInstanceLines(gt_instance, self.sample_dist,
                                         self.num_samples, self.padding, self.fixed_num, self.padding_value,
                                         patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,

        )
        # import pdb;pdb.set_trace()
        return anns_results

    def proc_polygon(self, polygon, ego_SE3_city):
        # import pdb;pdb.set_trace()
        interiors = []
        exterior_cityframe = np.array(list(polygon.exterior.coords))
        exterior_egoframe = ego_SE3_city.transform_point_cloud(exterior_cityframe)
        for inter in polygon.interiors:
            inter_cityframe = np.array(list(inter.coords))
            inter_egoframe = ego_SE3_city.transform_point_cloud(inter_cityframe)
            interiors.append(inter_egoframe[:, :])

        new_polygon = Polygon(exterior_egoframe[:, :], interiors)
        return new_polygon

    def get_map_boundary_geom(self, patch_box, patch_angle, avm, ego_SE3_city):
        map_boundary_geom = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        # import pdb;pdb.set_trace()
        polygon_list = []
        for da in avm:
            exterior_coords = da
            # import pdb;pdb.set_trace()
            interiors = []
            # import pdb;pdb.set_trace()
            is_polygon = np.array_equal(exterior_coords[0], exterior_coords[-1])
            if is_polygon:
                polygon = Polygon(exterior_coords, interiors)
            else:
                import pdb;
                pdb.set_trace()
                polygon = LineString(exterior_coords)
                raise ValueError(f'WRONG type: line in boundary')
            if is_polygon:
                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        # import pdb;pdb.set_trace()
                        if new_polygon.geom_type is 'Polygon':
                            if not new_polygon.is_valid:
                                continue
                            new_polygon = self.proc_polygon(new_polygon, ego_SE3_city)
                            if not new_polygon.is_valid:
                                continue
                        elif new_polygon.geom_type is 'MultiPolygon':
                            polygons = []
                            for single_polygon in new_polygon.geoms:
                                if not single_polygon.is_valid or single_polygon.is_empty:
                                    continue
                                new_single_polygon = self.proc_polygon(single_polygon, ego_SE3_city)
                                if not new_single_polygon.is_valid:
                                    continue
                                polygons.append(new_single_polygon)
                            if len(polygons) == 0:
                                continue
                            new_polygon = MultiPolygon(polygons)
                            if not new_polygon.is_valid:
                                continue
                        else:
                            raise ValueError('{} is not valid'.format(new_polygon.geom_type))
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
            else:
                raise ValueError(f'WRONG type: line in boundary')
        map_boundary_geom.append(('boundary', polygon_list))
        return map_boundary_geom

    def get_map_ped_geom(self, patch_box, patch_angle, avm, ego_SE3_city):
        map_ped_geom = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        # import pdb;pdb.set_trace()
        polygon_list = []
        for pc in avm:
            exterior_coords = pc
            interiors = []
            polygon = Polygon(exterior_coords, interiors)
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    if new_polygon.geom_type is 'Polygon':
                        if not new_polygon.is_valid:
                            continue
                        new_polygon = self.proc_polygon(new_polygon, ego_SE3_city)
                        if not new_polygon.is_valid:
                            continue
                    elif new_polygon.geom_type is 'MultiPolygon':
                        polygons = []
                        for single_polygon in new_polygon.geoms:
                            if not single_polygon.is_valid or single_polygon.is_empty:
                                continue
                            new_single_polygon = self.proc_polygon(single_polygon, ego_SE3_city)
                            if not new_single_polygon.is_valid:
                                continue
                            polygons.append(new_single_polygon)
                        if len(polygons) == 0:
                            continue
                        new_polygon = MultiPolygon(polygons)
                        if not new_polygon.is_valid:
                            continue
                    else:
                        raise ValueError('{} is not valid'.format(new_polygon.geom_type))

                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
        map_ped_geom.append(('ped_crossing', polygon_list))
        return map_ped_geom

    def proc_line(self, line, ego_SE3_city):
        # import pdb;pdb.set_trace()
        new_line_pts_cityframe = np.array(list(line.coords))
        new_line_pts_egoframe = ego_SE3_city.transform_point_cloud(new_line_pts_cityframe)
        line = LineString(new_line_pts_egoframe[:, :])
        return line

    def get_map_divider_geom(self, patch_box, patch_angle, avm, ego_SE3_city):
        map_divider_geom = []
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        for ls in avm:
            line = LineString(ls)
            if line.is_empty:  # Skip lines without nodes.
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                # import pdb;pdb.set_trace()
                if new_line.geom_type == 'MultiLineString':
                    for single_line in new_line.geoms:
                        if single_line.is_empty:
                            continue

                        single_line = self.proc_line(single_line, ego_SE3_city)
                        line_list.append(single_line)
                else:
                    new_line = self.proc_line(new_line, ego_SE3_city)
                    line_list.append(new_line)
        map_divider_geom.append(('divider', line_list))
        return map_divider_geom

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        # union_segments = ops.unary_union(ped)
        # union_segments = MultiPolygon(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        # local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        # if union_segments.geom_type != 'MultiPolygon':
        #     union_segments = MultiPolygon([union_segments])
        for segments in ped:
            if segments.geom_type != 'MultiPolygon':
                segments = MultiPolygon([segments])
            for poly in segments.geoms:
                exteriors.append(poly.exterior)
                for inter in poly.interiors:
                    interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def bound_poly_geoms_to_instances(self, polygon_geom):
        # roads = polygon_geom[0][1]
        # lanes = polygon_geom[1][1]
        # union_roads = ops.unary_union(roads)
        # union_lanes = ops.unary_union(lanes)
        # union_segments = ops.unary_union([union_roads, union_lanes])
        # import pdb;pdb.set_trace()
        bounds = polygon_geom[0][1]

        union_segments = ops.unary_union(bounds)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_instances(self, line_geom):
        lines = line_geom[0][1]
        multiline = MultiLineString(lines)
        union_lines = ops.unary_union(multiline)
        if union_lines.geom_type == 'LineString':
            return self._one_type_line_geom_to_instances([union_lines])
        before_num = len(union_lines.geoms)
        # import pdb;pdb.set_trace()
        merged_lines = ops.linemerge(union_lines)
        if merged_lines.geom_type == 'LineString':
            return self._one_type_line_geom_to_instances([merged_lines])
        after_num = len(merged_lines.geoms)
        # import pdb;pdb.set_trace()
        while after_num != before_num:
            before_num = len(merged_lines.geoms)
            merged_lines = ops.unary_union(merged_lines)
            if merged_lines.geom_type == 'LineString':
                break
            merged_lines = ops.linemerge(merged_lines)
            if merged_lines.geom_type == 'LineString':
                break
            after_num = len(merged_lines.geoms)

        return self._one_type_line_geom_to_instances([merged_lines])

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1,
                                                                                                                   2)

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        return sampled_points, num_valid

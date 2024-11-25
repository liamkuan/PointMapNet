import os
import warnings
from typing import Dict, Optional, Union, Tuple, List

import mmengine
import numpy
import numpy as np
import torch
from mmcv import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.models.task_modules import AssignResult
from mmdet3d.registry import TRANSFORMS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from scipy.optimize import linear_sum_assignment
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from shapely.strtree import STRtree
from ..map_utils.map_api import CNuScenesMapExplorer

# warnings.filterwarnings("ignore")


class InstanceLines(object):
    def __init__(self,
                 instance_line_list,
                 instance_labels,
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
        self.instance_labels = instance_labels

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
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2)
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
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
            dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
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
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2)
            shift_pts_list.append(sampled_points)
            # if is_poly:
            #     pts_to_shift = poly_pts[:-1,:]
            #     for shift_right_i in range(shift_num):
            #         shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
            #         pts_to_concat = shift_pts[0]
            #         pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
            #         shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
            #         shift_instance = LineString(shift_pts)
            #         shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            #         shift_pts_list.append(shift_sampled_points)
            #     # import pdb;pdb.set_trace()
            # else:
            #     sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            #     flip_sampled_points = np.flip(sampled_points, axis=0)
            #     shift_pts_list.append(sampled_points)
            #     shift_pts_list.append(flip_sampled_points)

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
                padding = torch.full([final_shift_num - multi_shifts_pts_tensor.shape[0], self.fixed_num, 2],
                                     self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
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
                padding = torch.full([shift_num - shift_pts.shape[0], pts_num, 2], self.padding_value)
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
        for idx, instance in enumerate(self.instance_list):
            # import ipdb;ipdb.set_trace()
            instance_label = self.instance_labels[idx]
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
            if instance_label == 3:
                # import ipdb;ipdb.set_trace()
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                shift_pts_list.append(sampled_points)
            else:
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
                                                                                                                   2)
                        shift_pts_list.append(shift_sampled_points)
                    # import pdb;pdb.set_trace()
                else:
                    sampled_points = np.array(
                        [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
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
                padding = torch.full([final_shift_num - multi_shifts_pts_tensor.shape[0], self.fixed_num, 2],
                                     self.padding_value)
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
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape
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
            if multi_shifts_pts_tensor.shape[0] < 2 * final_shift_num:
                padding = torch.full([final_shift_num * 2 - multi_shifts_pts_tensor.shape[0], self.fixed_num, 2],
                                     self.padding_value)
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
        for fixed_num_pts in fixed_num_sampled_points:
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
                padding = torch.full([shift_num * 2 - shift_pts.shape[0], pts_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
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

        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num - shift_pts.shape[0], fixed_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
            dtype=torch.float32)
        return instances_tensor


def radian2position(radian):
    if radian < 0:
        radian += + 2 * torch.pi
    radian += torch.pi / 8
    index = int(4 * radian / torch.pi)
    if index == 1:
        return 0, 2
    elif index == 2:
        return 0, 1
    elif index == 3:
        return 0, 0
    elif index == 4:
        return 1, 0
    elif index == 5:
        return 2, 0
    elif index == 6:
        return 2, 1
    elif index == 7:
        return 2, 2
    else:
        return 1, 2


def convert_three_point_to_mask(point1, point2, point3):
    mask = torch.zeros((3, 3), dtype=torch.long)
    mask[1, 1] = 1

    if point1 is not None:
        direction1 = torch.atan2(point1[1] - point2[1], point1[0] - point2[0])
        row1, col1 = radian2position(direction1)
        mask[row1, col1] = 1
    if point3 is not None:
        direction2 = torch.atan2(point3[1] - point2[1], point3[0] - point2[0])
        row2, col2 = radian2position(direction2)
        mask[row2, col2] = 1
    return mask.flatten(0, 1)


def points2masks(points):
    points = torch.from_numpy(points)
    masks = []
    for i, point in enumerate(points):
        if i == 0:
            masks.append(convert_three_point_to_mask(None, point, points[i + 1]))
        elif i == len(points) - 1:
            masks.append(convert_three_point_to_mask(points[i - 1], point, None))
        else:
            masks.append(convert_three_point_to_mask(points[i - 1], point, points[i + 1]))
    return masks


@TRANSFORMS.register_module(force=True)
class VectorizeLocalMapV1(BaseTransform):

    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }

    def __init__(self,
                 data_root="./data/nuscenes",
                 patch_size=(60., 30.),
                 map_classes=['divider', 'ped_crossing', 'boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 cache_dir=None,
                 use_cache=False,
                 use_depth=False,
                 use_focus_point=False,
                 bev_size=[20, 40],
                 padding_value=-10000, ):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = data_root
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.use_depth = use_depth
        self.use_focus_point = use_focus_point
        self.bev_size = bev_size
        self.coord_dim = 2

    def add_focus_point(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if len(results['gt_labels_3d']) != 0:
            instance_list = results["gt_bboxes_3d"].instance_list
            label_list = results["gt_labels_3d"]
            raw_points = []
            raw_points_label = []
            point_masks = []
            for instance, label in zip(instance_list, label_list):
                line_length = int(instance.length) * 2
                if label == 0:
                        line_length *= 2
                # else:
                #     if label == 0 or label == 1:
                #         line_length = line_length * 2
                num_point = max(line_length, len(instance.coords))
                distances = np.linspace(0, instance.length, num_point)
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances])[...,
                                 :self.coord_dim].reshape(-1, self.coord_dim)
                point_masks.extend(points2masks(sampled_points))
                raw_points.extend(sampled_points)
                raw_points_label.extend([label for _ in range(len(sampled_points))])

            results["point_masks"] = torch.stack(point_masks)
            results["raw_points"] = torch.from_numpy(np.asarray(raw_points)).float()
            results["raw_points_label"] = torch.stack(raw_points_label)

            x_coords = torch.linspace(0.05, 0.95, self.bev_size[0], dtype=torch.float32)
            y_coords = torch.linspace(0.05, 0.95, self.bev_size[1], dtype=torch.float32)
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
            coords = torch.stack((y_grid.flatten(), x_grid.flatten()), dim=-1)
            coords[:, 0:1] = coords[:, 0:1] * 30 - 15
            coords[:, 1:2] = coords[:, 1:2] * 60 - 30

            assigned_gt_inds = results["raw_points"].new_full((self.bev_size[0] * self.bev_size[1],),
                                                 -1,
                                                 dtype=torch.long)
            assigned_labels = results["raw_points"].new_full((self.bev_size[0] * self.bev_size[1],),
                                                -1,
                                                dtype=torch.long)
            cost = torch.cdist(coords, results["raw_points"], p=1)
            # mask_1 = (results["raw_points_label"] == 1).unsqueeze(0).expand(cost.shape)
            # mask_2 = (results["raw_points_label"] == 2).unsqueeze(0).expand(cost.shape)
            # cost[mask_1] *= 2
            # cost[mask_2] *= 4
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds)
            matched_col_inds = torch.from_numpy(matched_col_inds)
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = results["raw_points_label"][matched_col_inds]

            assign_result = AssignResult(len(raw_points), assigned_gt_inds, None, labels=assigned_labels)
            results["assign_result"] = assign_result
        else:
            results["raw_points"] = torch.zeros((0, self.coord_dim)).float()
            results["raw_points_label"] = torch.zeros((0, 1), dtype=torch.int64)
            results["point_masks"] = torch.zeros((0, 9), dtype=torch.int64)
            results["assign_result"] = None
        return results

    def get_image_depths(self, results):
        feat_shape = [60, 100]
        points = results["points"].tensor[:, :3]
        lidar2ego = torch.as_tensor(
            numpy.array(results["lidar2img"]), dtype=torch.float32
        )
        num_cam = lidar2ego.shape[0]
        points = torch.cat([points, torch.ones((points.shape[0], 1))], dim=-1).unsqueeze(-1)
        points = points.unsqueeze(0).repeat(num_cam, 1, 1, 1)

        lidar2ego = lidar2ego.view(num_cam, 1, 4, 4).repeat(1, points.shape[1], 1, 1)
        # 从LIDAR坐标到相机坐标的转换
        cam_points = torch.matmul(lidar2ego, points).squeeze(-1)
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
        image_depths = []
        for mask, cam_point in zip(masks, cam_points):
            image_depth = torch.zeros([*feat_shape, 1], dtype=torch.float32)
            index = torch.nonzero(mask).squeeze(-1)
            cam_point = torch.index_select(cam_point, dim=0, index=index)
            for c_point in cam_point:
                x = int(
                    c_point[1].item() / results["img_shape"][0][0] * feat_shape[0]
                )
                y = int(
                    c_point[0].item() / results["img_shape"][0][1] * feat_shape[1]
                )
                image_depth[x, y] = min(image_depth[x, y], c_point[2]) if image_depth[x, y] != 0 else c_point[2]
            image_depths.append(image_depth)
        results["image_depths"] = torch.stack(image_depths, dim=0)

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        cache_path = os.path.join(
            self.cache_dir, f"{results['sample_idx']}.pkl")
        if self.use_cache and os.path.exists(cache_path):
            cache = mmengine.load(cache_path)
            results['gt_labels_3d'] = cache['gt_labels_3d']
            results['gt_bboxes_3d'] = cache['gt_bboxes_3d']
            if self.use_depth:
                results['image_depths'] = cache['image_depths']
            if self.use_focus_point:
                results['point_masks'] = cache['point_masks']
                results['raw_points'] = cache['raw_points']
                results['raw_points_label'] = cache['raw_points_label']
                results['assign_result'] = cache['assign_result']
        else:
            if self.use_depth:
                self.get_image_depths(results)

            lidar2global = results['lidar2global']
            lidar2global_translation = list(lidar2global[:3, 3])
            lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

            location = results['map_location']

            anns_results = self.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)

            '''
            anns_results, type: dict
                'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
                'gt_vecs_pts_num': list[num_vecs], vec with num_points
                'gt_vecs_label': list[num_vecs], vec with cls index
            '''
            gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
            if isinstance(anns_results['gt_vecs_pts_loc'], InstanceLines):
                gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
            else:
                gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
                try:
                    gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
                except:
                    # empty tensor, will be passed in train,
                    # but we preserve it for test
                    gt_vecs_pts_loc = gt_vecs_pts_loc
            results['gt_labels_3d'] = gt_vecs_label
            results['gt_bboxes_3d'] = gt_vecs_pts_loc
            if self.use_focus_point:
                self.add_focus_point(results)
            if self.use_cache:
                cache = {
                    'gt_labels_3d': gt_vecs_label,
                    'gt_bboxes_3d': gt_vecs_pts_loc,
                }
                if self.use_depth:
                    cache['image_depths'] = results['image_depths']
                if self.use_focus_point:
                    cache['point_masks'] = results['point_masks']
                    cache['raw_points'] = results['raw_points']
                    cache['raw_points_label'] = results['raw_points_label']
                    cache['assign_result'] = results['assign_result']
                mmengine.dump(cache, cache_path)
        return results
    

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        '''
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        filtered_vectors = []
        gt_pts_loc_3d = []
        gt_pts_num_3d = []
        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)

        gt_instance = InstanceLines(gt_instance, gt_labels, self.sample_dist,
                                         self.num_samples, self.padding, self.fixed_num, self.padding_value,
                                         patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,

        )
        return anns_results

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

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

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
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

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
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

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
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

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer[location].map_api.extract_polygon(polygon_token) for polygon_token in
                            record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer[location].map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

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


@TRANSFORMS.register_module(force=True)
class VectorizeLocalMap(BaseTransform):

    def __init__(self,
                 data_root="./data/nuScenes/",
                 patch_size=(30, 60),
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 centerline_class=['lane_connector', 'lane'],
                 sample_dist=10,
                 num_samples=250,
                 padding=True,
                 max_len=30,
                 normalize=True,
                 fixed_num=50,
                 sample_pts=True,
                 class2label={
                     'ped_crossing': 0,
                     'divider': 1,
                     'contours': 2,
                     'others': -1,
                 },
                 num_point=30,
                 map_range=[-30, -15, 30, 15],
                 cache_dir='./data/nuscenes/temp',
                 use_cache=False,
                 ** kwargs):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = data_root
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.contour_classes = contour_classes
        self.centerline_class = centerline_class
        self.use_cache = use_cache
        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = CNuScenesMapExplorer(self.nusc_maps[loc])

        self.layer2class = {
            'ped_crossing': 'ped_crossing',
            'lane_divider': 'divider',
            'road_divider': 'divider',
            'road_segment': 'contours',
            'lane': 'contours',
        }

        self.process_func = {
            'ped_crossing': self.ped_geoms_to_vectors,
            'divider': self.line_geoms_to_vectors,
            'contours': self.poly_geoms_to_vectors,
            'centerline': self.line_geoms_to_vectors,
        }

        self.colors = {
            # 'ped_crossing': 'blue',
            'ped_crossing': 'royalblue',
            'divider': 'orange',
            'contours': 'green',
            # origin type
            'lane_divider': 'orange',
            'road_divider': 'orange',
            'road_segment': 'green',
            'lane': 'green',
        }

        self.sample_pts = sample_pts

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.max_len = max_len
        self.normalize = normalize
        self.fixed_num = fixed_num
        self.size = np.array([self.patch_size[1], self.patch_size[0]]) + 2

        self.num_point = num_point
        self.map_range = map_range
        self.cache_dir = cache_dir

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        cache_path = os.path.join(
            self.cache_dir, f"{results['sample_idx']}.pkl")
        if self.use_cache and os.path.exists(cache_path):
            results['target'] = mmengine.load(cache_path)
        else:
            results = self.vectorization(results)
            vectors = results['vectors']
            labels = []
            lines = []
            key_points = []
            line_weights = []
            for line, length, label in vectors:
                labels.append(label)
                line = torch.as_tensor(line, dtype=torch.float32)

                # line[..., 0:1] = (
                #         line[..., 0:1] * (self.map_range[2] - self.map_range[0]) + self.map_range[0])
                # line[..., 1:2] = (
                #         line[..., 1:2] * (self.map_range[3] - self.map_range[1]) + self.map_range[1])

                key_points.append(line)
                line, line_weight = padding_line_with_interpolation(
                    line, self.num_point)

                lines.append(line)
                line_weights.append(line_weight)

            target = {'lines': torch.stack(lines),
                        'labels': torch.tensor(labels, dtype=torch.int32),
                        'key_points': torch.cat(key_points, dim=0),
                        'line_weights': torch.stack(line_weights)}
            if self.use_cache:
                mmengine.dump(target, cache_path)
            results['target'] = target
        return results

    def retrive_geom(self, patch_params):
        '''
            Get the geometric data.
            Returns: dict
        '''
        patch_box, patch_angle, location = patch_params
        geoms_dict = {}

        layers = self.line_classes + self.ped_crossing_classes + self.contour_classes

        layers = set(layers)
        for layer_name in layers:

            return_token = False
            # retrive the geo
            if layer_name in self.nusc_maps[location].non_geometric_line_layers:
                geoms = self.map_explorer[location]._get_layer_line(
                    patch_box, patch_angle, layer_name)
            elif layer_name in self.nusc_maps[location].lookup_polygon_layers:
                geoms = self.map_explorer[location]._get_layer_polygon(
                    patch_box, patch_angle, layer_name, return_token=return_token)
            else:
                raise ValueError('{} is not a valid layer'.format(layer_name))

            if geoms is None:
                continue

            # change every geoms set to list
            if not isinstance(geoms, list):
                geoms = [geoms, ]

            geoms_dict[layer_name] = geoms

        return geoms_dict

    def union_geoms(self, geoms_dict):

        customized_geoms_dict = {}

        # contour
        roads = geoms_dict['road_segment']
        lanes = geoms_dict['lane']
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])

        customized_geoms_dict['contours'] = ('contours', [union_segments, ])

        # ped
        geoms_dict['ped_crossing'] = self.union_ped(geoms_dict['ped_crossing'])

        for layer_name, custom_class in self.layer2class.items():

            if custom_class == 'contours':
                continue

            customized_geoms_dict[layer_name] = (
                custom_class, geoms_dict[layer_name])

        return customized_geoms_dict

    def union_ped(self, ped_geoms):

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        for i in range(len(final_pgeom)):
            if final_pgeom[i].geom_type != 'MultiPolygon':
                final_pgeom[i] = MultiPolygon([final_pgeom[i]])

        return final_pgeom

    def convert2vec(self, geoms_dict: dict, sample_pts=False, override_veclen: int = None):

        vector_dict = {}
        for layer_name, (customized_class, geoms) in geoms_dict.items():

            line_strings = self.process_func[customized_class](geoms)

            vector_len = self.fixed_num[customized_class]
            if override_veclen is not None:
                vector_len = override_veclen

            vectors = self._geom_to_vectors(
                line_strings, customized_class, vector_len, sample_pts)
            vector_dict.update({layer_name: (customized_class, vectors)})

        return vector_dict

    def _geom_to_vectors(self, line_geom, label, vector_len, sample_pts=False):
        '''
            transfrom the geo type 2 line vectors
        '''
        line_vectors = {'vectors': [], 'length': []}
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line:
                        if sample_pts:
                            v, nl = self._sample_pts_from_line(
                                l, label, vector_len)
                        else:
                            v, nl = self._geoms2pts(l, label, vector_len)
                        line_vectors['vectors'].append(v.astype(np.float64))
                        line_vectors['length'].append(nl)
                elif line.geom_type == 'LineString':
                    if sample_pts:
                        v, nl = self._sample_pts_from_line(
                            line, label, vector_len)
                    else:
                        v, nl = self._geoms2pts(line, label, vector_len)
                    line_vectors['vectors'].append(v.astype(np.float64))
                    line_vectors['length'].append(nl)
                else:
                    raise NotImplementedError

        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geoms: list):

        results = []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []

        for geom in polygon_geoms:
            for poly in geom:
                exteriors.append(poly.exterior)
                for inter in poly.interiors:
                    interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            # since the start and end will disjoint
            # after applying the intersection.
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        return results

    def ped_geoms_to_vectors(self, geoms: list):

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for geom in geoms:
            for ped_poly in geom:
                # rect = ped_poly.minimum_rotated_rectangle
                ext = ped_poly.exterior
                if not ext.is_ccw:
                    ext.coords = list(ext.coords)[::-1]
                lines = ext.intersection(local_patch)

                if lines.type != 'LineString':
                    lines = ops.linemerge(lines)

                # same instance but not connected.
                if lines.type != 'LineString':
                    ls = []
                    for l in lines.geoms:
                        ls.append(np.array(l.coords))

                    lines = np.concatenate(ls, axis=0)
                    lines = LineString(lines)

                results.append(lines)

        return results

    def line_geoms_to_vectors(self, geom):
        # XXX
        return geom

    def _geoms2pts(self, line, label, fixed_point_num):

        # if we still use the fix point
        if fixed_point_num > 0:
            remain_points = fixed_point_num - np.asarray(line.coords).shape[0]
            if remain_points < 0:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > fixed_point_num:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

                remain_points = fixed_point_num - \
                    np.asarray(line.coords).shape[0]
                if remain_points > 0:
                    line = self.pad_line_with_interpolated_line(
                        line, remain_points)

            elif remain_points > 0:

                line = self.pad_line_with_interpolated_line(
                    line, remain_points)

            v = line
            if not isinstance(v, np.ndarray):
                v = np.asarray(line.coords)

            valid_len = v.shape[0]

        elif self.padding:  # dynamic points

            if self.max_len < np.asarray(line.coords).shape[0]:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > self.max_len:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

            v = np.asarray(line.coords)
            valid_len = v.shape[0]

            pad_len = self.max_len - valid_len
            v = np.pad(v, ((0, pad_len), (0, 0)), 'constant')

        else:
            # dynamic points without padding
            line = line.simplify(0.2, preserve_topology=True)
            v = np.array(line.coords)
            valid_len = len(v)

        if self.normalize:
            v = self.normalize_line(v)

        return v, valid_len

    def pad_line_with_interpolated_line(self, line: LineString, remain_points):
        ''' pad variable line with the interploated points'''

        origin_line = line
        line_length = line.length
        v = np.array(origin_line.coords)
        line_size = v.shape[0]

        interval = np.linalg.norm(v[1:]-v[:-1], axis=-1).cumsum()
        edges = np.hstack((np.array([0]), interval))/line_length

        # padding points
        interpolated_distances = np.linspace(
            0, 1, remain_points+2)[1:-1]  # get rid of start and end
        sampled_points = np.array([list(origin_line.interpolate(distance, normalized=True).coords)
                                   for distance in interpolated_distances]).reshape(-1, 2)

        # merge two line
        insert_idx = np.searchsorted(edges, interpolated_distances) - 1

        last_idx = 0
        new_line = []
        inserted_pos = np.unique(insert_idx)

        for i, idx in enumerate(inserted_pos):
            new_line += [v[last_idx:idx+1], sampled_points[insert_idx == idx]]
            last_idx = idx+1
        # for the remain points
        if last_idx <= line_size-1:
            new_line += [v[last_idx:], ]

        merged_line = np.concatenate(new_line, 0)

        return merged_line

    def _sample_pts_from_line(self, line, label, fixed_point_num):

        if fixed_point_num < 0:
            distances = list(np.arange(self.sample_dist,
                             line.length, self.sample_dist))
            distances = [0, ] + distances + [line.length, ]
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num

            distances = np.linspace(0, line.length, fixed_point_num)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)

        num_valid = len(sampled_points)

        # padding
        if fixed_point_num < 0 and self.padding:

            # fixed distance sampling need padding!
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        if self.normalize:
            sampled_points = self.normalize_line(sampled_points)

        return sampled_points, num_valid

    def normalize_line(self, line):
        '''
            prevent extrime pts such as 0 or 1.
        '''

        origin = -np.array([self.patch_size[1]/2, self.patch_size[0]/2])
        # for better learning
        line = line - origin
        line = line / self.size

        return line

    def get_global_patch(self, input_dict: dict):
        # transform to global coordination
        location = input_dict['location']
        ego2global_translation = input_dict['ego2global_translation']
        ego2global_rotation = input_dict['ego2global_rotation']
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)
        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        patch_params = (patch_box, patch_angle, location)
        return patch_params

    def vectorization(self, input_dict: dict):

        patch_params = self.get_global_patch(input_dict)

        # Retrive geo
        geoms_dict = self.retrive_geom(patch_params)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, orgin=False)

        # Optional union the data and convert customized labels
        geoms_dict = self.union_geoms(geoms_dict)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, origin=False, token=input_dict['token'])

        # Convert Geo 2 vec
        vectors_dict = self.convert2vec(geoms_dict, self.sample_pts)
        # self.debug_vis(patch_params, vectors_dict=vectors_dict,
        #                origin=False, token=input_dict['token'])

        # format the outputs list
        vectors = []
        for k, (custom_class, v) in vectors_dict.items():

            label = self.class2label.get(custom_class, -1)
            # filter out -1
            if label == -1:
                continue

            for vec, l in zip(v['vectors'], v['length']):

                vectors.append((vec, l, label))

        input_dict['vectors'] = vectors

        return input_dict


def get_start_name(i):
    return str(i)+'_start'


def get_end_name(i):
    return str(i)+'_end'

def interpolation(p1, p2, n):
    point_list = [p1]
    d = p2 - p1
    for i in range(n):
        point = torch.clone(p1)
        point = point + (d / (n + 1)) * (i + 1)
        point_list.append(point)
    return point_list


def padding_line_with_interpolation(line, num_point):
    if len(line) > num_point:
        indices = torch.sort(torch.randperm(len(line))[:num_point]).values
        return line[indices], torch.ones(num_point) * (1 / num_point)
    else:
        new_line = []
        weight = []
        n = num_point - len(line)
        l = int(n / (len(line) - 1))
        for i in range(len(line) - 1):
            if i < len(line) - 2:
                new_line += interpolation(line[i], line[i + 1], l)
                part_weight = [1] * (l + 1)
                part_weight[0] = 3
                weight += part_weight
            else:
                l = n - (len(line) - 2) * l
                new_line += interpolation(line[i], line[i + 1], l)
                new_line.append(line[-1])
                part_weight = [1] * (l + 2)
                part_weight[0] = 3
                part_weight[-1] = 3
                weight += part_weight
        return torch.stack(new_line), torch.tensor(weight) / sum(weight)
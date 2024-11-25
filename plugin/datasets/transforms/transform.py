import random
from typing import Dict, Optional, Union, Tuple, List

import math

import numpy
import numpy as np
import mmcv
import mmengine
import torch
import os
import cv2
from mmcv.transforms.base import BaseTransform
from mmdet.models.task_modules import AssignResult
from mmdet3d.registry import TRANSFORMS
from scipy.optimize import linear_sum_assignment

from plugin.models.utils.utils import inverse_sigmoid


@TRANSFORMS.register_module(force=True)
class Normalize3D(BaseTransform):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@TRANSFORMS.register_module(force=True)
class PadMultiViewImage(BaseTransform):
    """Pad multi-view images and change intrinsics
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Changed keys is 'cam_intrinsics',

    Args:
        size (tuple, optional): Fixed padding size, (h, w).
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]

        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
            self,
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@TRANSFORMS.register_module()
class RandomScaleImageMultiViewImage(BaseTransform):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results["img"]]
        x_size = [int(img.shape[1] * rand_scale) for img in results["img"]]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        # results["img"] = [img.astype(np.uint8) for img in results['img']]
        results["img"] = [
            mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False, backend='cv2')
            for idx, img in enumerate(results["img"])
        ]
        lidar2img = [scale_factor @ l2i for l2i in results["lidar2img"]]
        img_aug_matrix = [scale_factor for _ in results["lidar2img"]]
        results["lidar2img"] = lidar2img
        results["img_aug_matrix"] = img_aug_matrix
        results["img_shape"] = [img.shape for img in results["img"]]
        results["ori_shape"] = [img.shape for img in results["img"]]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.scales}, "
        return repr_str


@TRANSFORMS.register_module(force=True)
class ResizeMultiViewImages(BaseTransform):
    """Resize mulit-view images and change intrinsics
    Changed keys is 'cam_intrinsics'

    Args:
        size (tuple, optional): Fixed padding size, (h, w).
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size, change_intrinsics=False):
        self.size = size
        self.change_intrinsics = change_intrinsics

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        new_imgs, post_intrinsics, post_ego2imgs = [], [], []

        for img, ego2img in zip(results["img"], results["ego2img"]):
            tmp, scaleW, scaleH = mmcv.imresize(
                img,
                # mmcv.imresize expect (w, h) shape
                (self.size[1], self.size[0]),
                return_scale=True,
            )
            new_imgs.append(tmp)

            rot_resize_matrix = np.array(
                [[scaleW, 0, 0, 0], [0, scaleH, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            post_ego2img = rot_resize_matrix @ ego2img
            post_ego2imgs.append(post_ego2img)

        results["img"] = new_imgs
        results["img_shape"] = [img.shape for img in new_imgs]
        if self.change_intrinsics:
            results.update(
                {
                    "ego2img": post_ego2imgs,
                }
            )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


def random_sample(points, num):
    raw_size = points.size(0)
    if raw_size >= num:
        index = torch.LongTensor(random.sample(range(points.size(0)), num))
        return torch.index_select(points, 0, index)

    gap = num - raw_size
    gap_points = torch.zeros([gap, 3], dtype=torch.float32)
    return torch.cat([points, gap_points], dim=0)


@TRANSFORMS.register_module(force=True)
class Points2PositionEmbeddings(BaseTransform):
    def __init__(self, embed_size=(80, 200)):
        self.embed_size = embed_size

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        points = results["points"].tensor
        lidar2ego = torch.as_tensor(
            results["lidar_points"]["lidar2ego"], dtype=torch.float32
        )
        raw_ego_points = lidar2ego @ torch.cat(
            (points, torch.ones((points.shape[0], 1))), dim=-1
        ).unsqueeze(-1)

        ego2img = torch.as_tensor(results["ego2img"], dtype=torch.float32)
        num_cam = ego2img.shape[0]
        ego_points = raw_ego_points.unsqueeze(0).repeat(num_cam, 1, 1, 1)
        raw_ego_points = raw_ego_points.squeeze(-1)[:, :3]
        ego2img = ego2img.view(num_cam, 1, 4, 4).repeat(1, ego_points.shape[1], 1, 1)
        # 从BEV坐标到相机坐标的转换
        cam_points = torch.matmul(ego2img, ego_points).squeeze(-1)
        eps = 1e-5
        # 第三维代表缩放系数，除以缩放系数得到真实的uv
        masks = cam_points[..., 2:3] > eps
        cam_points = cam_points[..., 0:2] / torch.maximum(
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
        pos_embeds = []
        for mask, cam_point in zip(masks, cam_points):
            pos_embed = torch.zeros([*self.embed_size, 3], dtype=torch.float32)
            index = torch.nonzero(mask).squeeze(-1)
            ego_point = torch.index_select(raw_ego_points, dim=0, index=index)
            cam_point = torch.index_select(cam_point, dim=0, index=index)
            for e_point, c_point in zip(ego_point, cam_point):
                x = int(
                    c_point[1].item() / results["img_shape"][0][0] * self.embed_size[0]
                )
                y = int(
                    c_point[0].item() / results["img_shape"][0][1] * self.embed_size[1]
                )
                pos_embed[x, y] = e_point
            pos_embeds.append(pos_embed)
        results["pos_embeds"] = torch.stack(pos_embeds)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module(force=True)
class GetImageDepth(BaseTransform):
    def __init__(self, feat_shapes=None):
        self.feat_shapes = feat_shapes

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        points = results["points"].tensor[:, :3]
        lidar2ego = torch.as_tensor(
            numpy.array(results["lidar2img"]), dtype=torch.float32
        )
        # raw_ego_points = lidar2ego @ torch.cat(
        #     (points, torch.ones((points.shape[0], 1))), dim=-1
        # ).unsqueeze(-1)

        # ego2img = torch.as_tensor(results["ego2img"], dtype=torch.float32)
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
        image_depths_list = []
        for feat_shape in self.feat_shapes:
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
            image_depths_list.append(image_depths)
        results["image_depths"] = image_depths_list
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module(force=True)
class PolarPositionEmbeddings(BaseTransform):
    def __init__(self, radius=30, feat_shapes=None, num_cam=6):
        self.radius = radius
        self.feat_shapes = feat_shapes
        self.num_cam = num_cam

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        pad_h, pad_w, _ = results['pad_shape'][0]
        pos_points = []
        for feat_shape in self.feat_shapes:
            H, W = feat_shape
            coords_h = torch.arange(H, dtype=torch.float32) * pad_h / H
            coords_w = torch.arange(W, dtype=torch.float32) * pad_w / W

            index = torch.arange(start=0, end=64, step=1, dtype=torch.float32)
            bin_size = (self.radius - 1) / 64
            coords_d = 1 + bin_size * index

            D = coords_d.shape[0]
            coords = (torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij'))
                      .permute(1, 2, 3, 0).contiguous())  # W, H, D, 3
            coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
            coords[..., :2] = coords[..., :2] * coords[..., 2:3]

            img2lidar = []
            for mat in results['lidar2img']:
                img2lidar.append(torch.inverse(torch.from_numpy(mat).float()))
            img2lidar = torch.stack(img2lidar)

            coords = coords[None, :, :, :, :, None].repeat(self.num_cam, 1, 1, 1, 1, 1)
            img2lidar = img2lidar[:, None, None, None, :, :].repeat(1, W, H, D, 1, 1)
            coords3d = torch.matmul(img2lidar, coords).squeeze(-1)[..., :3]

            coords_radius = torch.sqrt(coords3d[..., 0:1] ** 2 + coords3d[..., 1:2] ** 2 + coords3d[..., 2:3] ** 2)
            coords_theta = torch.acos(coords3d[..., 2:3] / coords_radius)
            coords_phi = torch.atan2(coords3d[..., 1:2], coords3d[..., 0:1])

            coords_radius = coords_radius / self.radius
            coords_theta = coords_theta / torch.pi
            coords_phi = coords_phi + torch.pi / (2 * torch.pi)

            coords_polar = torch.cat([coords_radius, coords_theta, coords_phi], dim=-1)

            coords_polar = coords_polar.permute(0, 3, 4, 2, 1).contiguous().flatten(1, 2)
            coords_polar = inverse_sigmoid(coords_polar)
            pos_points.append(coords_polar)

        results["pos_points"] = pos_points

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module(force=True)
class PositionEmbeddings(BaseTransform):
    def __init__(self, pc_range=[-20.0, -40.0, -3.5, 20.0, 40, 0.5], feat_shapes=None, num_cam=6):
        self.pc_range = pc_range
        self.feat_shapes = feat_shapes
        self.num_cam = num_cam

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        pad_h, pad_w, _ = results['pad_shape'][0]
        pos_points = []
        for feat_shape in self.feat_shapes:
            H, W = feat_shape
            coords_h = torch.arange(H, dtype=torch.float32) * pad_h / H
            coords_w = torch.arange(W, dtype=torch.float32) * pad_w / W

            # index = torch.arange(start=0, end=64, step=1, device=device).float()
            # index_1 = index + 1
            # bin_size = (self.pc_range[4] - 1) / (64 * (1 + 64))
            # coords_d = self.pc_range[4] - 0.1 - bin_size * index * index_1

            index = torch.arange(start=0, end=64, step=1, dtype=torch.float32)
            bin_size = (self.pc_range[4] - 1) / 64
            coords_d = 1 + bin_size * index

            D = coords_d.shape[0]
            coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3,
                                                                                                        0).contiguous()  # W, H, D, 3
            coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
            coords[..., :2] = coords[..., :2] * coords[..., 2:3]

            img2lidar = []
            for mat in results['lidar2img']:
                img2lidar.append(torch.inverse(torch.from_numpy(mat).float()))
            img2lidar = torch.stack(img2lidar)

            coords = coords[None, :, :, :, :, None].repeat(self.num_cam, 1, 1, 1, 1, 1)
            img2lidar = img2lidar[:, None, None, None, :, :].repeat(1, W, H, D, 1, 1)
            coords3d = torch.matmul(img2lidar, coords).squeeze(-1)[..., :3]
            coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (
                    self.pc_range[3] - self.pc_range[0])
            coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[0]) / (
                    self.pc_range[3] - self.pc_range[0])
            coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (
                    self.pc_range[5] - self.pc_range[2])

            coords3d = coords3d.permute(0, 3, 4, 2, 1).contiguous().flatten(1, 2)
            coords3d = inverse_sigmoid(coords3d)
            pos_points.append(coords3d)

        results["pos_points"] = pos_points

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


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


@TRANSFORMS.register_module(force=True)
class Vectors2Target(BaseTransform):
    def __init__(
            self,
            num_point=30,
            map_range=[-30, -15, 30, 15],
            cache_dir="./data/nuscenes/temp",
    ):
        self.num_point = num_point
        self.map_range = map_range
        self.cache_dir = cache_dir

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        cache_path = os.path.join(self.cache_dir, f"{results['sample_idx']}.pkl")
        if os.path.exists(cache_path):
            results["target"] = mmengine.load(cache_path)
        else:
            vectors = results["vectors"]
            # if len(vectors) == 0:
            #     results['target'] = {'lines': torch.tensor([], dtype=torch.float32),
            #                         'labels': torch.tensor([], dtype=torch.int32),
            #                         'line_weights': torch.tensor([], dtype=torch.float32)}
            #     mmengine.dump(results['target'], os.path.join(self.cache_dir, f"{results['sample_idx']}.pkl"))
            #     return results
            labels = []
            lines = []
            line_weights = []
            for line, length, label in vectors:
                labels.append(label)
                line = torch.as_tensor(line, dtype=torch.float32)
                line, line_weight = padding_line_with_interpolation(
                    line, self.num_point
                )

                line[..., 0:1] = (
                        line[..., 0:1] * (self.map_range[2] - self.map_range[0])
                        + self.map_range[0]
                )
                line[..., 1:2] = (
                        line[..., 1:2] * (self.map_range[3] - self.map_range[1])
                        + self.map_range[1]
                )

                lines.append(line)
                line_weights.append(line_weight)

            target = {
                "lines": torch.stack(lines),
                "labels": torch.tensor(labels, dtype=torch.int32),
                "line_weights": torch.stack(line_weights),
            }
            mmengine.dump(
                target, os.path.join(self.cache_dir, f"{results['sample_idx']}.pkl")
            )
            results["target"] = target
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module(force=True)
class PaddingTarget(BaseTransform):
    def __init__(self, num_point=20, dim=2):
        self.num_point = num_point
        self.dim = dim

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        lines = results["target"]["lines"]
        results["target"]["raw_lines"] = lines
        # if len(lines) == 0:
        #     results['target'] = {'lines': torch.tensor([], dtype=torch.float32),
        #                          'labels': torch.tensor([], dtype=torch.int32)}
        #     return results
        new_lines = []
        line_weights = []
        for line in lines:
            line, line_weight = padding_line_with_interpolation(line, self.num_point)
            new_lines.append(line[:, : self.dim])
            line_weights.append(line_weight)
        results["target"]["lines"] = torch.stack(new_lines)
        results["target"]["line_weights"] = torch.stack(line_weights)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module(force=True)
class AddPointDirection(BaseTransform):
    def __init__(self):
        pass

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        def get_direction(point1, point2):
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            return dx / distance, dy / distance

        lines = results["target"]["lines"]
        line_directions = []
        for line in lines:
            directions = []
            for i, point in enumerate(line):
                if i == 0:
                    directions.append(0.0)
                    directions.append(0.0)
                else:
                    direction = get_direction(point, line[i - 1])
                    directions.append(direction[0])
                    directions.append(direction[1])

                if i == len(line) - 1:
                    directions.append(0.0)
                    directions.append(0.0)
                else:
                    direction = get_direction(point, line[i + 1])
                    directions.append(direction[0])
                    directions.append(direction[1])
            line_directions.append(directions)

        results["target"]["point_directions"] = torch.tensor(line_directions)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


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
class AddRawPoint(BaseTransform):
    def __init__(self, coord_dim=2, is_large_range=False, ):
        self.coord_dim = coord_dim
        self.is_large_range = is_large_range

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if len(results["gt_bboxes_3d"].instance_list) != 0:
            instance_list = results["gt_bboxes_3d"].instance_list
            label_list = results["gt_labels_3d"]
            raw_points = []
            raw_points_label = []
            point_masks = []
            for instance, label in zip(instance_list, label_list):
                line_length = int(instance.length)
                if self.is_large_range:
                    line_length = int(line_length / 2)
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

            # line_list = results["gt_bboxes_3d"].fixed_num_sampled_points
            # label_list = results["gt_labels_3d"]
            # raw_points = []
            # raw_points_label = []
            # for line, label in zip(line_list, label_list):
            #     for point in line:
            #         raw_points.append(point)
            #         raw_points_label.append(label)
            results["point_masks"] = torch.stack(point_masks)
            results["raw_points"] = torch.from_numpy(np.asarray(raw_points)).float()
            # assert len(results['raw_points']) < 1000
            results["raw_points_label"] = torch.stack(raw_points_label)
        else:
            results["raw_points"] = torch.zeros((0, self.coord_dim)).float()
        return results


@TRANSFORMS.register_module(force=True)
class AddRawPointV1(BaseTransform):
    def __init__(self, coord_dim=2, is_large_range=False, bev_size=(20, 40)):
        self.coord_dim = coord_dim
        self.is_large_range = is_large_range
        self.bev_size = bev_size

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if len(results['gt_labels_3d']) != 0:
            instance_list = results["gt_bboxes_3d"].instance_list
            label_list = results["gt_labels_3d"]
            raw_points = []
            raw_points_label = []
            point_masks = []
            for instance, label in zip(instance_list, label_list):
                line_length = int(instance.length)
                if self.is_large_range:
                    line_length = int(line_length / 2)
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
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds)
            matched_col_inds = torch.from_numpy(matched_col_inds)
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = results["raw_points_label"][matched_col_inds]

            assign_result = AssignResult(len(raw_points), assigned_gt_inds, None, labels=assigned_labels)
            results["assign_result"] = assign_result
        else:
            results["raw_points"] = torch.zeros((0, self.coord_dim)).float()
        return results


@TRANSFORMS.register_module(force=True)
class AddBEVPoint(BaseTransform):
    def __init__(self, bev_shape=[100, 200], pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]):
        self.bev_shape = bev_shape
        self.pc_range = pc_range

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        bev_image = np.zeros(self.bev_shape, dtype=np.uint8)
        pc_width = self.pc_range[3] - self.pc_range[0]
        pc_height = self.pc_range[4] - self.pc_range[1]

        for instance in results["gt_bboxes_3d"].instance_list:
            line = [[int(item[0] * self.bev_shape[0] / pc_width) + self.bev_shape[0] // 2, int(item[1] * self.bev_shape[1] / pc_height) + self.bev_shape[1] // 2] for item in instance.coords]
            for i in range(len(line) - 1):
                cv2.line(bev_image, line[i], line[i + 1], (1, 1, 1), 2)
        bev_points = torch.from_numpy(bev_image)
        results["bev_points"] = bev_points
        return results

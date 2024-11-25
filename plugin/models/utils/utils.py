import math
import torch
from torch import Tensor
from torch import nn
import copy
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
import numpy as np
from mmdet.registry import MODELS
from typing import Tuple, Union, List
from mmcv.cnn.bricks import Swish, build_norm_layer
from mmengine.model import BaseModule
from mmdet.utils import MultiConfig, OptConfigType


class MaxPool2dSamePadding(nn.Module):

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 2,
                 **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) -
                   1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) -
                   1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])
        x = self.pool(x)

        return x


class DepthWiseConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_norm: bool = True,
        conv_bn_act_pattern: bool = False,
        norm_cfg: OptConfigType = dict(type='BN', momentum=1e-2, eps=1e-3)
    ) -> None:
        super(DepthWiseConvBlock, self).__init__()
        self.depthwise_conv = Conv2dSamePadding(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            bias=False)
        self.pointwise_conv = Conv2dSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1)

        self.apply_norm = apply_norm
        if self.apply_norm:
            self.bn = build_norm_layer(norm_cfg, num_features=out_channels)[1]

        self.apply_activation = conv_bn_act_pattern
        if self.apply_activation:
            self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.apply_norm:
            x = self.bn(x)
        if self.apply_activation:
            x = self.swish(x)

        return x


class DownChannelBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        apply_norm: bool = True,
        conv_bn_act_pattern: bool = False,
        norm_cfg: OptConfigType = dict(type='BN', momentum=1e-2, eps=1e-3)
    ) -> None:
        super(DownChannelBlock, self).__init__()
        self.down_conv = Conv2dSamePadding(in_channels, out_channels, 1)
        self.apply_norm = apply_norm
        if self.apply_norm:
            self.bn = build_norm_layer(norm_cfg, num_features=out_channels)[1]
        self.apply_activation = conv_bn_act_pattern
        if self.apply_activation:
            self.swish = Swish()

    def forward(self, x):
        x = self.down_conv(x)
        if self.apply_norm:
            x = self.bn(x)
        if self.apply_activation:
            x = self.swish(x)

        return x



class BiFPNStage(nn.Module):
    """
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        first_time: int, whether is the first bifpnstage
        conv_bn_act_pattern: bool, whether use conv_bn_act_pattern
        norm_cfg: (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer.
        epsilon: float, hyperparameter in fusion features
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 first_time: bool = False,
                 apply_bn_for_resampling: bool = True,
                 conv_bn_act_pattern: bool = False,
                 norm_cfg: OptConfigType = dict(
                     type='BN', momentum=1e-2, eps=1e-3),
                 epsilon: float = 1e-4) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_time = first_time
        self.apply_bn_for_resampling = apply_bn_for_resampling
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.norm_cfg = norm_cfg
        self.epsilon = epsilon

        if self.first_time:
            self.p5_down_channel = DownChannelBlock(
                self.in_channels[-1],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p4_down_channel = DownChannelBlock(
                self.in_channels[-2],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p3_down_channel = DownChannelBlock(
                self.in_channels[-3],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p5_to_p6 = nn.Sequential(
                DownChannelBlock(
                    self.in_channels[-1],
                    self.out_channels,
                    apply_norm=self.apply_bn_for_resampling,
                    conv_bn_act_pattern=self.conv_bn_act_pattern,
                    norm_cfg=norm_cfg), MaxPool2dSamePadding(3, 2))
            self.p6_to_p7 = MaxPool2dSamePadding(3, 2)
            self.p4_level_connection = DownChannelBlock(
                self.in_channels[-2],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p5_level_connection = DownChannelBlock(
                self.in_channels[-1],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # bottom to up: feature map down_sample module
        self.p4_down_sample = MaxPool2dSamePadding(3, 2)
        self.p5_down_sample = MaxPool2dSamePadding(3, 2)
        self.p6_down_sample = MaxPool2dSamePadding(3, 2)
        self.p7_down_sample = MaxPool2dSamePadding(3, 2)

        # Fuse Conv Layers
        self.conv6_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv5_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv4_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv3_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv4_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv5_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv6_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv7_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        # weights
        self.p6_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.swish = Swish()

    def combine(self, x):
        if not self.conv_bn_act_pattern:
            x = self.swish(x)

        return x

    def forward(self, x):
        if self.first_time:
            p3, p4, p5 = x
            # build feature map P6
            p6_in = self.p5_to_p6(p5)
            # build feature map P7
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = x

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(
            self.combine(weight[0] * p6_in +
                         weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(
            self.combine(weight[0] * p5_in +
                         weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(
            self.combine(weight[0] * p4_in +
                         weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(
            self.combine(weight[0] * p3_in +
                         weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_level_connection(p4)
            p5_in = self.p5_level_connection(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.combine(weight[0] * p4_in + weight[1] * p4_up +
                         weight[2] * self.p4_down_sample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.combine(weight[0] * p5_in + weight[1] * p5_up +
                         weight[2] * self.p5_down_sample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.combine(weight[0] * p6_in + weight[1] * p6_up +
                         weight[2] * self.p6_down_sample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(
            self.combine(weight[0] * p7_in +
                         weight[1] * self.p7_down_sample(p6_out)))
        return p3_out, p4_out, p5_out, p6_out, p7_out


@MODELS.register_module()
class BiFPN(BaseModule):
    """
        num_stages: int, bifpn number of repeats
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        start_level: int, Index of input features in backbone
        epsilon: float, hyperparameter in fusion features
        apply_bn_for_resampling: bool, whether use bn after resampling
        conv_bn_act_pattern: bool, whether use conv_bn_act_pattern
        norm_cfg: (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer.
        init_cfg: MultiConfig: init method
    """

    def __init__(self,
                 num_stages: int,
                 in_channels: List[int],
                 out_channels: int,
                 start_level: int = 0,
                 epsilon: float = 1e-4,
                 apply_bn_for_resampling: bool = True,
                 conv_bn_act_pattern: bool = False,
                 norm_cfg: OptConfigType = dict(
                     type='BN', momentum=1e-2, eps=1e-3),
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.start_level = start_level
        self.bifpn = nn.Sequential(*[
            BiFPNStage(
                in_channels=in_channels,
                out_channels=out_channels,
                first_time=True if _ == 0 else False,
                apply_bn_for_resampling=apply_bn_for_resampling,
                conv_bn_act_pattern=conv_bn_act_pattern,
                norm_cfg=norm_cfg,
                epsilon=epsilon) for _ in range(num_stages)
        ])

    def forward(self, x):
        x = x[self.start_level:]
        x = self.bifpn(x)

        return x

@MODELS.register_module()
class Conv2dSamePadding(nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
                         dilation, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        extra_w = (math.ceil(img_w / self.stride[1]) -
                   1) * self.stride[1] - img_w + kernel_w
        extra_h = (math.ceil(img_h / self.stride[0]) -
                   1) * self.stride[0] - img_h + kernel_h

        left = extra_w // 2
        right = extra_w - left
        top = extra_h // 2
        bottom = extra_h - top
        x = F.pad(x, [left, right, top, bottom])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    # pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)

    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def normalize_2d_pts(pts, pc_range):
    if pts.size(-1) == 2:
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        new_pts = pts.clone()
        new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
        new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
        factor = pts.new_tensor([patch_w, patch_h])
        normalized_pts = new_pts / factor
    elif pts.size(-1) == 3:
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        patch_z = pc_range[5] - pc_range[2]
        new_pts = pts.clone()
        new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
        new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
        new_pts[..., 2:3] = pts[..., 2:3] - pc_range[2]
        factor = pts.new_tensor([patch_w, patch_h, patch_z])
        normalized_pts = new_pts / factor
    else:
        return None
    return normalized_pts


# def polar_normalize_2d_pts(pts, radius=30):
#     if pts.size(-1) == 2:
#         new_pts = pts.clone()
#         coords_radius = torch.sqrt(new_pts[..., 0:1] ** 2 + new_pts[..., 1:2] ** 2)
#         coords_theta = torch.atan2(new_pts[..., 1:2], new_pts[..., 0:1])
#         coords_radius = coords_radius / radius
#         coords_theta = (coords_theta + torch.pi) / (2 * torch.pi)
#         normalized_pts = torch.cat([coords_radius, coords_theta], dim=-1)
#     elif pts.size(-1) == 3:
#         return None
#     else:
#         return None
#     return normalized_pts


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] -
                                          pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] -
                                          pc_range[1]) + pc_range[1])
    return new_pts


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def get_arc_length(dataset):
    r"""
    Obtain arc length distances between every point in 2-D space

    Obtains the total arc length of a curve in 2-D space (a curve of x and y)
    as well as the arc lengths between each two consecutive data points of the
    curve.

    Parameters
    ----------
    dataset : ndarray (2-D)
        The dataset of the curve in 2-D space.

    Returns
    -------
    arcLength : float
        The sum of all consecutive arc lengths
    arcLengths : array_like
        A list of the arc lengths between data points

    Notes
    -----
    Your x locations of data points should be dataset[:, 0], and the y
    locations of the data points should be dataset[:, 1]
    """
    #   split the dataset into two discrete datasets, each of length m-1
    m = len(dataset)
    a = dataset[0:m-1, :]
    b = dataset[1:m, :]
    #   use scipy.spatial to compute the euclidean distance
    dataDistance = torch.cdist(a, b, p=2)
    #   this returns a matrix of the euclidean distance between all points
    #   the arc length is simply the sum of the diagonal of this matrix
    arcLengths = torch.diagonal(dataDistance)
    arcLength = sum(arcLengths)
    return arcLength, arcLengths


def area_between_two_curves(exp_data: Tensor, num_data: Tensor):
    r"""
    Calculates the area between two curves.

    This calculates the area according to the algorithm in [1]_. Each curve is
    constructed from discretized data points in 2-D space, e.g. each curve
    consists of x and y data points.

    Parameters
    ----------
    exp_data : ndarray (2-D)
        Curve from your experimental data.
    num_data : ndarray (2-D)
        Curve from your numerical data.

    Returns
    -------
    area : float
        The area between exp_data and num_data curves.

    References
    ----------
    .. [1] Jekel, C. F., Venter, G., Venter, M. P., Stander, N., & Haftka, R.
        T. (2018). Similarity measures for identifying material parameters from
        hysteresis loops using inverse analysis. International Journal of
        Material Forming. https://doi.org/10.1007/s12289-018-1421-8

    Notes
    -----
    Your x locations of data points should be exp_data[:, 0], and the y
    locations of the data points should be exp_data[:, 1]. Same for num_data.
    """

    n_exp = len(exp_data)
    n_num = len(num_data)

    # get the arc length data of the curves
    # arcexp_data, _ = get_arc_length(exp_data)
    _, arcsnum_data = get_arc_length(num_data)

    # let's find the largest gap between point the num_data, and then
    # linearally interpolate between these points such that the num_data
    # becomes the same length as the exp_data
    for i in range(0, n_exp-n_num):
        a = num_data[0:n_num-1, 0]
        b = num_data[1:n_num, 0]
        nIndex = torch.argmax(arcsnum_data)
        newX = (b[nIndex] + a[nIndex])/2.0
        #   the interpolation model messes up if x2 < x1 so we do a quick check
        if a[nIndex] < b[nIndex]:

            newY = np.interp(newX, [a[nIndex], b[nIndex]],
                             [num_data[nIndex, 1], num_data[nIndex+1, 1]])
        else:
            newY = np.interp(newX, [b[nIndex], a[nIndex]],
                             [num_data[nIndex+1, 1], num_data[nIndex, 1]])
        num_data = np.insert(num_data, nIndex+1, newX, axis=0)
        num_data[nIndex+1, 1] = newY

        _, arcsnum_data = get_arc_length(num_data)
        n_num = len(num_data)

    # Calculate the quadrilateral area, by looping through all of the quads
    area = []
    for i in range(1, n_exp):
        tempX = [exp_data[i-1, 0], exp_data[i, 0], num_data[i, 0],
                 num_data[i-1, 0]]
        tempY = [exp_data[i-1, 1], exp_data[i, 1], num_data[i, 1],
                 num_data[i-1, 1]]
        area.append(makeQuad(tempX, tempY))
    return np.sum(area)

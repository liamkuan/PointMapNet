# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import random
from xml.etree.ElementPath import prepare_descendant
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models import NormedLinear, inverse_sigmoid
from mmdet3d.registry import MODELS
import numpy as np
import math
import copy

from mmengine.model import bias_init_with_prob


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
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
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class RegLayer(nn.Module):
    def __init__(self, embed_dims=256,
                 shared_reg_fcs=2,
                 group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()

        reg_branch = []
        for _ in range(shared_reg_fcs):
            reg_branch.append(Linear(embed_dims, embed_dims))
            reg_branch.append(act_layer())
            reg_branch.append(nn.Dropout(drop))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.task_heads = nn.ModuleList()
        for reg_dim in group_reg_dims:
            task_head = nn.Sequential(
                Linear(embed_dims, embed_dims),
                act_layer(),
                Linear(embed_dims, reg_dim)
            )
            self.task_heads.append(task_head)

    def forward(self, x):
        reg_feat = self.reg_branch(x)
        outs = []
        for task_head in self.task_heads:
            out = task_head(reg_feat.clone())
            outs.append(out)
        outs = torch.cat(outs, -1)
        return outs


@MODELS.register_module()
class PETRv2Head(nn.Module):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 criterion=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 train_cfg=None,
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_level=0,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 normedlinear=False,
                 with_fpe=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(PETRv2Head, self).__init__()
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = position_level
        self.with_position = with_position
        self.with_multiview = with_multiview
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        self.with_fpe = with_fpe

        self.cls_out_channels = num_classes
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.criterion = MODELS.build(criterion)
        self.transformer = MODELS.build(transformer)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        nx = 4
        ny = 60
        assert nx * ny == self.num_query # 调整num_query的时候也要调整这里的关键点分布
        x = (torch.arange(nx) + 0.5) / (nx * 2) + 0.5
        y = (torch.arange(ny) + 0.5) / ny
        xy = torch.meshgrid(x, y)
        self.front_reference_points = torch.cat(
            [xy[0].reshape(-1)[..., None], xy[1].reshape(-1)[..., None]], -1).cuda()
        self.front_query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 2 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        if self.with_fpe:
            self.fpe = SELayer(self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['ego2img'])):
                img2lidar.append(np.linalg.inv(img_meta['ego2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars, dtype=torch.float32)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
                self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
                self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
                self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def get_perv_query_embeds(self, x, masks, pos_embed):
        """
        ref_point 有没有得到训练的问题需要考虑
        :param x:
        :param masks:
        :param pos_embed:
        :return:
        """
        bs = x.shape[0]
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        query_embeds = query_embeds.unsqueeze(1).repeat(1, bs, 1)
        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_class = self.cls_branches[-1](outs_dec[-1]).sigmoid()
        query_masks = outputs_class > 0.3
        query_masks = query_masks[:, :, 0] | query_masks[:, :, 1]

        queries = []
        ref_points = []
        for query, mask in zip(outs_dec[-1], query_masks):
            index = torch.nonzero(mask).squeeze(-1)
            query = torch.index_select(query, dim=0, index=index)
            ref_point = torch.index_select(reference_points, dim=0, index=index)
            
            front_query = self.front_query_embedding(pos2posemb2d(self.front_reference_points))
            index = torch.LongTensor(random.sample(range(front_query.shape[0]), front_query.shape[0] - len(query))).cuda()
            front_query = torch.index_select(front_query, 0, index)
            front_ref_point = torch.index_select(self.front_reference_points, 0, index)
            queries.append(torch.cat((query, front_query), dim=0))
            front_ref_point = torch.cat((front_ref_point, torch.zeros_like(front_ref_point[:, 0:1])), dim=1)
            ref_points.append(torch.cat((ref_point, front_ref_point), dim=0))


        return torch.stack(queries, dim=0).permute(1, 0, 2), torch.stack(ref_points, dim=0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.size(0), x.size(1)

        current_frame = x[:, :6]
        past_frame = x[:, 6:]

        x = torch.cat([current_frame, past_frame.detach()], 1)

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
            if self.with_fpe:
                coords_position_embeding = self.fpe(coords_position_embeding.flatten(0, 1), x.flatten(0, 1)).view(
                    x.size())

            pos_embed = coords_position_embeding

            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        prev_x = x[:, 6:]
        prev_masks = masks[:, 6:]
        prev_pos_embed = pos_embed[:, :6]

        query_embeds, reference_points = self.get_perv_query_embeds(prev_x, prev_masks, prev_pos_embed)
        # reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)  # .sigmoid()
        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp = tmp.view(*tmp.shape[:-1], int(self.code_size / 3), 3)
            tmp += reference.view(*reference.shape[:2], 1, 3).repeat(1, 1, tmp.shape[-2], 1)
            tmp = tmp.sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:1] = (
                    all_bbox_preds[..., 0:1] * (self.position_range[3] - self.position_range[0])
                    + self.position_range[0])
        all_bbox_preds[..., 1:2] = (
                    all_bbox_preds[..., 1:2] * (self.position_range[4] - self.position_range[1])
                    + self.position_range[1])
        all_bbox_preds[..., 2:3] = (
                    all_bbox_preds[..., 2:3] * (self.position_range[5] - self.position_range[2])
                    + self.position_range[2])

        new_outputs = {}
        new_outputs['pred_logits'] = all_cls_scores[-1]
        new_outputs['pred_lines'] = all_bbox_preds[-1]
        new_outputs['aux_outputs'] = [{'pred_logits': a, 'pred_lines': b}
                                      for a, b in zip(all_cls_scores[:-1], all_bbox_preds[:-1])]

        return new_outputs

    def loss_single(self,
                    lane_preds,
                    gt_lane_list):

        # loss_lane_mask=self.loss_lane_mask(lane_preds,gt_lane_list[0])
        # import ipdb;ipdb.set_trace()
        lane_preds = lane_preds.unsqueeze(-1).flatten(0, 1)
        gt_lane_list = gt_lane_list.flatten(1).to(lane_preds.device).flatten(0, 1).long()
        # labels = torch.ones_like(gt_lane_list)

        loss_dri = self.loss_dri(lane_preds, gt_lane_list)
        loss_dri = torch.nan_to_num(loss_dri)

        return loss_dri, None

    def loss(self,
             preds_dicts,
            #  gt_lanes,
             targets):
        loss_dict = dict()
        # all_lane_preds = preds_dicts['all_lane_preds']
        # num_dec_layers = len(all_lane_preds)
        # all_gt_lanes_list = [torch.stack(gt_lanes) for _ in range(num_dec_layers)]
        # losses_dri, _ = multi_apply(
        #     self.loss_single, all_lane_preds, torch.stack(all_gt_lanes_list))

        # loss_dict['loss_dri'] = losses_dri[-1]
        # loss from other decoder layers
        # num_dec_layer = 0
        # for loss_dri_i in losses_dri[:-1]:
        #     loss_dict[f'd{num_dec_layer}.loss_dri'] = loss_dri_i
        #     num_dec_layer += 1

        losses_line = self.criterion(preds_dicts, targets, self.num_classes)
        loss_dict.update(losses_line)
        return loss_dict

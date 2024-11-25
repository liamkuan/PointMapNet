# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models import inverse_sigmoid
import numpy as np
import math
from mmdet.models.utils import multi_apply
from mmdet3d.registry import MODELS
import mmengine
from mmdet.models import NormedLinear
import os


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
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


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            # nn.BatchNorm2d(dim),
            # nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.LayerNorm(out_channels),
        )

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, out_dim, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels
        self.to_logits = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels,
                      3, padding=1, bias=False),
            # nn.BatchNorm2d(self.out_channels),
            # nn.LayerNorm(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, out_dim, 1))

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)
        y = self.to_logits(y)
        return y


@MODELS.register_module()
class PETRHead_seg(nn.Module):

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_line_query=100,
                 num_reg_fcs=2,
                 with_position=True,
                 with_multiview=False,
                 depth_num=64,
                 blocks=[128, 128],
                 LID=False,
                 depth_start=1,
                 position_level=0,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 normedlinear=False,
                 with_se=False,
                 with_time=False,
                 with_detach=False,
                 with_multi=False,
                 group_reg_dims=(2, 1, 3, 2, 2),
                 transformer=None,
                 criterion=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_dri=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.5,
                     loss_weight=2.0),
                 loss_lan=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.5,
                     loss_weight=2.0),
                 loss_ped=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.5,
                     loss_weight=2.0),
                 train_cfg: Optional[dict] = None,
                 test_cfg=dict(max_per_img=100),):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        super(PETRHead_seg, self).__init__()
        # self.num_line_query = num_line_query
        self.blocks = blocks
        # self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.test_cfg = test_cfg
        self.embed_dims = 256
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = position_level
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.with_multi = with_multi
        self.group_reg_dims = group_reg_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        self.with_se = with_se
        self.with_time = with_time
        self.with_detach = with_detach

        self.loss_dri = MODELS.build(loss_dri)
        self.loss_lan = MODELS.build(loss_lan)
        self.loss_ped = MODELS.build(loss_ped)

        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        self.transformer = MODELS.build(transformer)
        # self.criterion = MODELS.build(criterion)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        # cls_branch = []
        # for _ in range(self.num_reg_fcs):
        #     cls_branch.append(Linear(self.embed_dims, self.embed_dims))
        #     cls_branch.append(nn.LayerNorm(self.embed_dims))
        #     cls_branch.append(nn.ReLU(inplace=True))
        # if self.normedlinear:
        #     cls_branch.append(NormedLinear(self.embed_dims, 2))
        # else:
        #     cls_branch.append(Linear(self.embed_dims, 2))
        # fc_cls = nn.Sequential(*cls_branch)
        #
        # reg_branch = []
        # for _ in range(self.num_reg_fcs):
        #     reg_branch.append(Linear(self.embed_dims, self.embed_dims))
        #     reg_branch.append(nn.ReLU())
        # reg_branch.append(Linear(self.embed_dims, 40))
        # reg_branch = nn.Sequential(*reg_branch)
        #
        # self.cls_branches = nn.ModuleList(
        #     [fc_cls for _ in range(self.num_pred)])
        # self.reg_branches = nn.ModuleList(
        #     [reg_branch for _ in range(self.num_pred)])

        lane_branch_dri = Decoder(self.embed_dims, self.blocks, 1)
        lane_branch_lan = Decoder(self.embed_dims, self.blocks, 1)
        lane_branch_ped = Decoder(self.embed_dims, self.blocks, 1)

        self.lane_branches_dri = nn.ModuleList(
            [lane_branch_dri for _ in range(self.num_pred)])
        self.lane_branches_lan = nn.ModuleList(
            [lane_branch_lan for _ in range(self.num_pred)])
        self.lane_branches_ped = nn.ModuleList(
            [lane_branch_ped for _ in range(self.num_pred)])
        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims *
                          4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims *
                          4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims,
                          kernel_size=1, stride=1, padding=0),
            )

        if self.with_se:
            self.se = SELayer(self.embed_dims)

        nx = 30
        ny = 15
        x = (torch.arange(nx) + 0.5) / nx
        y = (torch.arange(ny) + 0.5) / ny
        xy = torch.meshgrid(x, y)
        self.reference_points_map = torch.cat(
            [xy[0].reshape(-1)[..., None], xy[1].reshape(-1)[..., None]], -1).cuda()

        self.query_embedding_map = nn.Sequential(
            nn.Linear(self.embed_dims * 2 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important

        self.transformer.init_weights()

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(
            H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(
            W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num,
                                 step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (
                self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num,
                                 step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] -
                        self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(
            1, 2, 3, 0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * \
            torch.maximum(coords[..., 2:3], torch.ones_like(
                coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['ego2img'])):
                img2lidar.append(np.linalg.inv(img_meta['ego2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(
            B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(
            img2lidars, coords).squeeze(-1)[..., :3].float()
        # mmengine.dump(coords3d, os.path.join('coords3d.pkl'))
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(
            0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

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

        x = mlvl_feats[self.position_level].float()
        batch_size, num_cams = x.size(0), x.size(1)

        if self.with_detach:
            current_frame = x[:, :6]
            past_frame = x[:, 6:]
            x = torch.cat([current_frame, past_frame.detach()], 1)

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = self.input_proj(x.flatten(0, 1)).float()
        x = x.view(batch_size, num_cams, *x.shape[-3:])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(
                mlvl_feats, img_metas, masks)
            coords_position_embeding = coords_position_embeding.float()
            if self.with_se:
                coords_position_embeding = self.se(coords_position_embeding.flatten(0, 1), x.flatten(0, 1)).view(
                    x.size())

            pos_embed = coords_position_embeding

            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(
                    sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(
                    sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(
                    pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        query_map = self.query_embedding_map(
            pos2posemb2d(self.reference_points_map))

        outs_dec, _ = self.transformer(
            x, masks, query_map, pos_embed, self.lane_branches_dri)
        outs_dec = torch.nan_to_num(outs_dec)
        map_queries = outs_dec

        outputs_classes = []
        outputs_coords = []
        outputs_maps = []
        for lvl in range(map_queries.shape[0]):
            map_queries_lvl = map_queries[lvl].view(
                map_queries.shape[1], 15, 30, -1).permute(0, 3, 1, 2)
            outputs_dri = self.lane_branches_dri[lvl](map_queries_lvl)
            outputs_lan = self.lane_branches_lan[lvl](map_queries_lvl)
            outputs_ped = self.lane_branches_ped[lvl](map_queries_lvl)

            outputs_map = torch.cat(
                [outputs_dri, outputs_lan, outputs_ped], dim=1)
            outputs_map = outputs_map.view(-1, 3, 60*120)
            outputs_maps.append(outputs_map)

        all_map_preds = torch.stack(outputs_maps)

        outs = {
            'all_map_preds': all_map_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def loss_single(self,
                    lane_preds,
                    gt_lane_list):
        
        loss_lan = self.loss_lan(lane_preds[0].flatten(0, 1), gt_lane_list[0].flatten(0, 1).long())
        loss_ped = self.loss_ped(lane_preds[1].flatten(0, 1), gt_lane_list[1].flatten(0, 1).long())
        loss_dri = self.loss_dri(lane_preds[2].flatten(0, 1), gt_lane_list[2].flatten(0, 1).long())

        loss_dri = torch.nan_to_num(loss_dri)
        loss_lan = torch.nan_to_num(loss_lan)
        loss_ped = torch.nan_to_num(loss_ped)

        return loss_dri, loss_lan, loss_ped

    def loss(self,
             preds_dicts,
             maps,):
        all_map_preds = preds_dicts['all_map_preds']
        maps = torch.stack(maps, dim=0)
        num_dec_layers = len(all_map_preds)
        all_map_list = [maps.flatten(2) for _ in range(num_dec_layers)]
        all_map_list = torch.stack(all_map_list, dim=0)

        all_map_preds = all_map_preds.permute(0, 2, 1, 3)
        all_map_list = all_map_list.permute(0, 2, 1, 3)

        losses_dri, losses_lan, losses_ped = multi_apply(
            self.loss_single, all_map_preds, all_map_list)
        loss_dict = dict()
        # loss of proposal generated from encode feature map.

        # loss from the last decoder layer

        loss_dict['loss_dri'] = losses_dri[-1]
        loss_dict['loss_lan'] = losses_lan[-1]
        loss_dict['loss_ped'] = losses_ped[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_dri_i, loss_lan_i, loss_ped_i in zip(losses_dri[:-1], losses_lan[:-1], losses_ped[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_dri'] = loss_dri_i
            loss_dict[f'd{num_dec_layer}.loss_lan'] = loss_lan_i
            loss_dict[f'd{num_dec_layer}.loss_ped'] = loss_ped_i
            num_dec_layer += 1
        return loss_dict

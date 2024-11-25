from typing import Optional

import math
import warnings

import numpy as np
import torch
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models import inverse_sigmoid
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, xavier_init
from torch import nn, Tensor
from torch.nn.init import normal_
from plugin.models.utils.utils import _get_clones


@MODELS.register_module()
class PMNTransformer(BaseModule):
    def __init__(
            self,
            encoder=None,
            decoder=None,
            positional_encoding=None,
            pc_range=None,
            num_points=20,
            num_line=50,
            coord_dim=2,
            init_cfg=None,
            num_feature_levels=4,
            use_3d_pos=True,
    ):
        super(PMNTransformer, self).__init__(init_cfg=init_cfg)
        self.encoder = MODELS.build(encoder)
        self.embed_dims = self.encoder.embed_dims
        self.decoder = MODELS.build(decoder)
        self.positional_encoding = MODELS.build(positional_encoding)
        self.num_feature_levels = num_feature_levels
        self.num_points = num_points
        self.num_line = num_line
        self.coord_dim = coord_dim
        self.num_cams = 6
        self.pc_range = pc_range
        self.use_3d_pos = use_3d_pos
        self.init_layers()

    def init_layers(self):
        self.point_ref_points = nn.Linear(self.embed_dims, self.coord_dim)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        # self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        if self.use_3d_pos:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(3 * 64, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.instance_embedding = nn.Embedding(self.num_line, self.embed_dims)
        self.pts_embedding = nn.Embedding(self.num_points, self.embed_dims)

        # self.line_ref_points = nn.Linear(self.embed_dims, 2)
        # self.can_bus_mlp = nn.Sequential(
        #     nn.Linear(7, self.embed_dims // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(),
        # )
        # self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        # follow the official DETR to init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                xavier_init(m, distribution="uniform")
        normal_(self.level_embeds)
        # normal_(self.cams_embeds)
        xavier_init(self.point_ref_points, distribution="uniform", bias=0.0)
        # xavier_init(self.line_ref_points, distribution="uniform", bias=0.0)
        # xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        self._is_init = True

    def position_embeding(self, img_feats, img_metas):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats.shape
        device = img_feats.device
        coords_h = torch.arange(H, device=device).float() * pad_h / H
        coords_w = torch.arange(W, device=device).float() * pad_w / W

        # index = torch.arange(start=0, end=64, step=1, device=device).float()
        # index_1 = index + 1
        # bin_size = (self.pc_range[4] - 1) / (64 * (1 + 64))
        # coords_d = self.pc_range[4] - 0.1 - bin_size * index * index_1

        index = torch.arange(start=0, end=64, step=1, device=device).float()
        bin_size = (self.pc_range[4] - 1) / 64
        coords_d = 1 + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3,
                                                                                                    0).contiguous()  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(torch.inverse(torch.from_numpy(img_meta['lidar2img'][i])))
            img2lidars.append(torch.stack(img2lidar))
        img2lidars = torch.stack(img2lidars).float().to(device)  # (B, N, 4, 4)

        coords = coords[None, None, :, :, :, :, None].repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars[:, :, None, None, None, :, :].repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])

        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().flatten(0, 1).flatten(1, 2)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W)

    def get_key_and_key_pos(self, mlvl_feats, img_metas):
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat_mask = torch.zeros((bs, num_cam, h, w), device=feat.device).to(feat.dtype)

            if self.use_3d_pos:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                    pos_points = [item['pos_points'][lvl] for item in img_metas]
                    pos_points = torch.stack(pos_points, dim=0).flatten(0, 1)
                    pos_embed = self.position_encoder(pos_points).view(bs, num_cam, self.embed_dims, h, w)
            sin_embed = self.positional_encoding(feat_mask)
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(feat.size())
            if self.use_3d_pos:
                key_pos = (pos_embed + sin_embed).to(feat.dtype)
            else:
                key_pos = sin_embed.to(feat.dtype)
            feat = feat + key_pos
            feat = feat.flatten(3).permute(1, 0, 3, 2).contiguous()
            # key_pos = key_pos.flatten(3).permute(1, 0, 3, 2).contiguous().to(feat.dtype)
            # feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl: lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        return feat_flatten, spatial_shapes

    def forward(
            self, mlvl_feats, point_embedding, point_reg_branch, reg_branch, img_metas
    ):
        bs = mlvl_feats[0].size(0)
        point_pos, point_queries = torch.split(point_embedding, self.embed_dims, dim=1)
        point_queries = point_queries.unsqueeze(0).repeat(bs, 1, 1)
        point_pos = point_pos.unsqueeze(0).repeat(bs, 1, 1)
        point_ref_points = self.point_ref_points(point_pos).float().sigmoid()
        # can_bus = point_embedding.new_tensor(
        #     [each["ego2global_translation"] + each["ego2global_rotation"] for each in img_metas]
        # )  # [:, :]
        # can_bus = self.can_bus_mlp(can_bus)[:, None, :]
        # point_embedding = point_embedding + can_bus

        feat_flatten, spatial_shapes = self.get_key_and_key_pos(mlvl_feats, img_metas)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=point_embedding.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3
        ).contiguous()  # (num_cam, H*W, bs, embed_dims)
        point_embed, point_inter_ref_points = self.encoder(
            query=point_queries,
            key=feat_flatten,
            value=feat_flatten,
            ref_points=point_ref_points,
            query_pos=point_pos,
            key_pos=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=point_reg_branch,
            img_metas=img_metas,
        )

        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)
        tgt = object_query_embeds.unsqueeze(0).expand(bs, -1, -1)
        out_dec = self.decoder(tgt, point_embed[-1], point_inter_ref_points[-1].detach(), reg_branch, None)
        return point_embed, out_dec, point_ref_points, point_inter_ref_points


@MODELS.register_module()
class PMNV1Transformer(BaseModule):
    def __init__(
            self,
            encoder=None,
            decoder=None,
            positional_encoding=None,
            pc_range=None,
            num_points=20,
            num_line=50,
            coord_dim=2,
            init_cfg=None,
            num_feature_levels=4,
    ):
        super(PMNV1Transformer, self).__init__(init_cfg=init_cfg)
        self.encoder = MODELS.build(encoder)
        self.embed_dims = self.encoder.embed_dims
        self.decoder = MODELS.build(decoder)
        self.positional_encoding = MODELS.build(positional_encoding)
        self.num_feature_levels = num_feature_levels
        self.num_points = num_points
        self.num_line = num_line
        self.coord_dim = coord_dim
        self.num_cams = 6
        self.pc_range = pc_range
        self.init_layers()

    def init_layers(self):
        self.point_ref_points = nn.Linear(self.embed_dims, self.coord_dim)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        # self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.position_encoder = nn.Sequential(
            nn.Conv2d(3, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        depth_head = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, 1, kernel_size=1, stride=1, padding=0),
        )
        self.depth_heads = _get_clones(depth_head, self.num_feature_levels)
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.instance_embedding = nn.Embedding(self.num_line, self.embed_dims)
        self.pts_embedding = nn.Embedding(self.num_points, self.embed_dims)

    def init_weights(self):
        # follow the official DETR to init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                xavier_init(m, distribution="uniform")
        normal_(self.level_embeds)
        # normal_(self.cams_embeds)
        xavier_init(self.point_ref_points, distribution="uniform", bias=0.0)
        self._is_init = True

    def position_embeding(self, img_feats, img_metas, num_level):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats.shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        coords_d = self.depth_heads[num_level](img_feats.flatten(0, 1).float()).sigmoid() * self.pc_range[4]
        coords_d = coords_d.view(B, N, 1, H, W).permute(0, 1, 4, 3, 2)  # B, N, W, H, 1
        coords_h = coords_h.view(1, 1, 1, H, 1).repeat(B, N, W, 1, 1)
        coords_w = coords_w.view(1, 1, W, 1, 1).repeat(B, N, 1, H, 1)
        coords = torch.cat([coords_w, coords_h, coords_d, torch.ones_like(coords_h)], dim=-1)
        coords = torch.cat((coords, ), -1)
        temp = coords[..., :2] * coords[..., 2:3]
        coords = torch.cat([temp, coords[..., 2:]], dim=-1)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(B, N, W, H, 4, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 4, 4).repeat(1, 1, W, H, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (
                    self.pc_range[3] - self.pc_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[1]) / (
                    self.pc_range[4] - self.pc_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (
                    self.pc_range[5] - self.pc_range[2])

        # coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        # coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        # coords_mask = coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 3, 2).flatten(0, 1)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W).contiguous()

    def get_key_and_key_pos(self, mlvl_feats, img_metas):
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat_mask = torch.zeros((bs, num_cam, h, w), device=feat.device).to(feat.dtype)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                pos_embed = self.position_embeding(feat, img_metas, lvl)

            sin_embed = self.positional_encoding(feat_mask)
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(feat.size())
            key_pos = (pos_embed + sin_embed).to(feat.dtype)
            feat = feat + key_pos
            feat = feat.flatten(3).permute(1, 0, 3, 2).contiguous()
            feat = feat + self.level_embeds[None, None, lvl: lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        return feat_flatten, spatial_shapes

    def forward(
            self, mlvl_feats, point_embedding, point_reg_branch, point_cls_branch, reg_branch, img_metas
    ):
        bs = mlvl_feats[0].size(0)
        point_pos, point_queries = torch.split(point_embedding, self.embed_dims, dim=1)
        point_queries = point_queries.unsqueeze(0).repeat(bs, 1, 1)
        point_pos = point_pos.unsqueeze(0).repeat(bs, 1, 1)

        point_ref_points = self.point_ref_points(point_pos).float().sigmoid()

        feat_flatten, spatial_shapes = self.get_key_and_key_pos(mlvl_feats, img_metas)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=point_embedding.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3
        ).contiguous()  # (num_cam, H*W, bs, embed_dims)
        point_embed, point_inter_ref_points = self.encoder(
            query=point_queries,
            key=feat_flatten,
            value=feat_flatten,
            ref_points=point_ref_points,
            query_pos=point_pos,
            key_pos=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=point_reg_branch,
            img_metas=img_metas,
        )
        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)
        tgt = object_query_embeds.unsqueeze(0).expand(bs, -1, -1)
        out_dec = self.decoder(tgt, point_embed[-1], point_inter_ref_points[-1].detach(), reg_branch, None)
        return point_embed, out_dec, point_ref_points, point_inter_ref_points


@MODELS.register_module()
class PMNDecoder(BaseModule):
    def __init__(self, embed_dims=256, return_intermediate=True, num_layers=3, num_points=20, distance=0.1, use_mask=True):
        super(PMNDecoder, self).__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_points = num_points
        self.num_head = 8
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dims, nhead=self.num_head)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate
        self.distance = distance
        self.use_mask = use_mask
        # self.line_encoding = nn.Linear(20 * embed_dims, embed_dims)

    def forward(self, tgt, memory, key_points, reg_branch, cls_mask, *args, **kwargs):
        memory = memory.transpose(0, 1)

        intermediate = []
        output = tgt
        # cls_mask = cls_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1).flatten(0, 1)
        for lvl, layer in enumerate(self.layers):
            if lvl > 0:
                ref_points = reg_branch[-1](output).float().sigmoid()
                bs, num_point, _ = ref_points.shape
                bs, num_key_points, _ = key_points.shape
                cdist = torch.cdist(ref_points, key_points).view(bs, 50, self.num_points, num_key_points)
                cdist, _ = torch.min(cdist, dim=2)
                mask = cdist > self.distance
                # temp = torch.count_nonzero(mask, dim=2)
                mask = mask[:, None, :, None, :].repeat(1, self.num_head, 1, self.num_points, 1).flatten(0, 1).flatten(1, 2)
            else:
                mask = None
            if not self.use_mask:
                mask = None
            output = layer(output.transpose(0, 1), memory, memory_mask=mask, *args, **kwargs)
            output = output.transpose(0, 1)
            intermediate.append(output)
        return torch.stack(intermediate)


@MODELS.register_module()
class PMNDecoderV1(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(PMNDecoderV1, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(
            self,
            query,
            *args,
            reference_points=None,
            reg_branches=None,
            **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2
            reference_points_input = reference_points_input.repeat(1, 1, 4, 1)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs,
            )
            # output = output.permute(1, 0, 2)

            if reg_branches is not None:
                # reference_points = reg_branches[lid](output).sigmoid()

                tmp = reg_branches[lid](output)
                # tmp = tmp.view(*tmp.shape[:-1], 20, 2)
                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            # output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


# @MODELS.register_module()
# class PMNAttention(BaseModule):
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  num_levels=1,
#                  kernel_size=(3, 3),
#                  dilation=1,
#                  im2col_step=64,
#                  dropout=0.1,
#                  batch_first=False,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super(PMNAttention, self).__init__()
#         if embed_dims % num_heads != 0:
#             raise ValueError(f'embed_dims must be divisible by num_heads, '
#                              f'but got {embed_dims} and {num_heads}')
#         dim_per_head = embed_dims // num_heads
#         self.norm_cfg = norm_cfg
#         self.batch_first = batch_first
#         self.output_proj = None
#
#         # you'd better set dim_per_head to a power of 2
#         # which is more efficient in the CUDA implementation
#         def _is_power_of_2(n):
#             if (not isinstance(n, int)) or (n < 0):
#                 raise ValueError(
#                     'invalid input for _is_power_of_2: {} (type: {})'.format(
#                         n, type(n)))
#             return (n & (n - 1) == 0) and n != 0
#
#         if not _is_power_of_2(dim_per_head):
#             warnings.warn(
#                 "You'd better set embed_dims in "
#                 'MultiScaleDeformAttention to make '
#                 'the dimension of each attention head a power of 2 '
#                 'which is more efficient in our CUDA implementation.')
#
#         self.im2col_step = im2col_step
#         self.embed_dims = embed_dims
#         # 4
#         self.num_levels = num_levels
#         # 4 num_heads -> num_z_anchors
#         self.num_heads = num_heads
#         self.kernel_size = kernel_size
#         self.num_points = kernel_size[0] * kernel_size[1]
#         # self.sampling_offsets = nn.Linear(
#         #     embed_dims, num_heads * num_levels * self.num_points * 2)
#
#         self.attention_weights = nn.Linear(
#             embed_dims, num_levels * self.num_points * self.num_heads)
#         self.value_proj = nn.Linear(embed_dims, embed_dims)
#
#         grid_h, grid_w = kernel_size
#         y = (torch.arange(grid_h) - grid_h // 2) * dilation
#         x = (torch.arange(grid_w) - grid_w // 2) * dilation
#         offsets = torch.stack(
#             torch.meshgrid(x, y)).permute(1, 2, 0).reshape(grid_h * grid_w, 2)
#         self.register_buffer("grid_offsets", offsets, persistent=False)
#         self.init_weights()
#
#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 identity=None,
#                 query_pos=None,
#                 key_padding_mask=None,
#                 reference_points=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 **kwargs):
#         if value is None:
#             value = query
#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             query = query + query_pos
#
#         if not self.batch_first:
#             # change to (bs, num_query ,embed_dims)
#             query = query.permute(1, 0, 2)
#             value = value.permute(1, 0, 2)
#
#         bs, num_query, _ = query.shape
#         bs, num_value, _ = value.shape
#         assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
#
#         value = self.value_proj(value)
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], 0.0)
#         value = value.view(bs, num_value, self.num_heads, -1)
#         # sampling_offsets = self.sampling_offsets(query).view(
#         #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
#
#         # bs, num_query, num_heads, num_levels, num_points
#         # bs, q, 4, 4, K^2
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query, self.num_heads, self.num_levels * self.num_points)
#
#         attention_weights = attention_weights.softmax(-1)
#
#         attention_weights = attention_weights.view(bs, num_query,
#                                                    self.num_heads,
#                                                    self.num_levels,
#                                                    self.num_points)
#
#         if reference_points.shape[-1] == 2:
#             """
#             For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
#             After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
#             For each referent point, we sample `num_points` sampling points.
#             For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
#             """
#             with torch.no_grad():
#                 offset_normalizer = torch.stack(
#                     [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#
#                 bs, num_query, num_Z_anchors, xy = reference_points.shape
#                 # from IPython import embed; embed()
#                 # (K,2) -> (1, 1, 1, 1, k, 2) -> (bs, q, nz, l, k, 2)
#                 offsets = self.grid_offsets[None, None, None, None]
#                 # (bs, q, nz, 1, xy) -> (bs, q, z, l, 2)
#                 reference_points = reference_points[:,
#                                    :, :, None, :] * offset_normalizer
#
#                 # from IPython import embed;embed()
#                 # (bs, q, nz, l, k, xy)
#                 sampling_locations = (
#                         reference_points[:, :, :, :, None, :] + offsets).round().long()
#
#             # sampling_offsets = sampling_offsets / \
#             #     offset_normalizer[None, None, None, :, None, :]
#             # (bs, q, 4(z), 4, K^2, 2)
#             bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_locations.shape
#             # sampling_offsets = sampling_offsets.view(
#             #     bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
#             # sampling_locations = reference_points + sampling_offsets
#             # bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
#             # assert num_all_points == num_points * num_Z_anchors
#
#             # sampling_locations = sampling_locations.view(
#             #     bs, num_query, num_heads, num_levels, num_all_points, xy)
#
#         elif reference_points.shape[-1] == 4:
#             assert False
#         else:
#             raise ValueError(
#                 f'Last dim of reference_points must be'
#                 f' 2 or 4, but get {reference_points.shape[-1]} instead.')
#
#         #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
#         #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
#         # import pdb;pdb.set_trace()
#         # output = self.forward_kernel_multihead_attention(
#         #     value, spatial_shapes, sampling_locations, attention_weights)
#         # torch.cuda.synchronize()
#         # start = time.perf_counter()
#         output = GeometricKernelAttentionFunc.apply(
#             value, spatial_shapes, level_start_index, sampling_locations.contiguous(), attention_weights,
#             self.im2col_step
#         )
#         # if torch.cuda.is_available() and value.is_cuda:
#         #     if value.dtype == torch.float16:
#         #         MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#         #     else:
#         #         MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#         #     output = MultiScaleDeformableAttnFunction.apply(
#         #         value, spatial_shapes, level_start_index, sampling_locations,
#         #         attention_weights, self.im2col_step)
#         # else:
#         #     output = multi_scale_deformable_attn_pytorch(
#         #         value, spatial_shapes, sampling_locations, attention_weights)
#         if not self.batch_first:
#             output = output.permute(1, 0, 2)
#         # torch.cuda.synchronize()
#         # end = time.perf_counter()
#         # print("geometric kernel attention: {:.3f} ms".format((end-start)*1000))
#         return output


@MODELS.register_module()
class PMNTransformerTest(BaseModule):
    def __init__(
            self,
            encoder=None,
            decoder=None,
            positional_encoding=None,
            pc_range=None,
            num_points=20,
            num_line=50,
            coord_dim=2,
            init_cfg=None,
            num_feature_levels=4,
    ):
        super(PMNTransformerTest, self).__init__(init_cfg=init_cfg)
        self.encoder = MODELS.build(encoder)
        self.embed_dims = self.encoder.embed_dims
        self.decoder = MODELS.build(decoder)
        self.positional_encoding = MODELS.build(positional_encoding)
        self.num_feature_levels = num_feature_levels
        self.num_points = num_points
        self.num_line = num_line
        self.coord_dim = coord_dim
        self.num_cams = 6
        self.pc_range = pc_range
        self.init_layers()

    def init_layers(self):
        self.point_ref_points = nn.Linear(self.embed_dims, self.coord_dim)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        # self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.position_encoder = nn.Sequential(
            nn.Conv2d(3 * 64, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.instance_embedding = nn.Embedding(self.num_line, self.embed_dims)
        self.pts_embedding = nn.Embedding(self.num_points, self.embed_dims)

        # self.line_ref_points = nn.Linear(self.embed_dims, 2)
        # self.can_bus_mlp = nn.Sequential(
        #     nn.Linear(7, self.embed_dims // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(),
        # )
        # self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        # follow the official DETR to init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                xavier_init(m, distribution="uniform")
        normal_(self.level_embeds)
        # normal_(self.cams_embeds)
        xavier_init(self.point_ref_points, distribution="uniform", bias=0.0)
        # xavier_init(self.line_ref_points, distribution="uniform", bias=0.0)
        # xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        self._is_init = True

    def position_embeding(self, img_feats, img_metas):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats.shape
        device = img_feats.device
        coords_h = torch.arange(H, device=device).float() * pad_h / H
        coords_w = torch.arange(W, device=device).float() * pad_w / W

        # index = torch.arange(start=0, end=64, step=1, device=device).float()
        # index_1 = index + 1
        # bin_size = (self.pc_range[4] - 1) / (64 * (1 + 64))
        # coords_d = self.pc_range[4] - 0.1 - bin_size * index * index_1

        index = torch.arange(start=0, end=64, step=1, device=device).float()
        bin_size = (self.pc_range[4] - 1) / 64
        coords_d = 1 + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3,
                                                                                                    0).contiguous()  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(torch.inverse(torch.from_numpy(img_meta['lidar2img'][i])))
            img2lidars.append(torch.stack(img2lidar))
        img2lidars = torch.stack(img2lidars).float().to(device)  # (B, N, 4, 4)

        coords = coords[None, None, :, :, :, :, None].repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars[:, :, None, None, None, :, :].repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])

        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().flatten(0, 1).flatten(1, 2)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W)

    def get_key_and_key_pos(self, mlvl_feats, img_metas):
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat_mask = torch.zeros((bs, num_cam, h, w), device=feat.device).to(feat.dtype)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                pos_points = [item['pos_points'][lvl] for item in img_metas]
                pos_points = torch.stack(pos_points, dim=0).flatten(0, 1)
                pos_embed = self.position_encoder(pos_points).view(bs, num_cam, self.embed_dims, h, w)
            sin_embed = self.positional_encoding(feat_mask)
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(feat.size())
            key_pos = (pos_embed + sin_embed).to(feat.dtype)
            feat = feat + key_pos
            feat = feat.flatten(3).permute(1, 0, 3, 2).contiguous()
            # key_pos = key_pos.flatten(3).permute(1, 0, 3, 2).contiguous().to(feat.dtype)
            # feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl: lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        return feat_flatten, spatial_shapes

    def forward(
            self, mlvl_feats, point_embedding, point_reg_branch, reg_branch, img_metas
    ):
        bs = mlvl_feats[0].size(0)
        point_pos, point_queries = torch.split(point_embedding, self.embed_dims, dim=1)
        point_queries = point_queries.unsqueeze(0).repeat(bs, 1, 1)
        point_pos = point_pos.unsqueeze(0).repeat(bs, 1, 1)
        point_ref_points = self.point_ref_points(point_pos).float().sigmoid()
        # can_bus = point_embedding.new_tensor(
        #     [each["ego2global_translation"] + each["ego2global_rotation"] for each in img_metas]
        # )  # [:, :]
        # can_bus = self.can_bus_mlp(can_bus)[:, None, :]
        # point_embedding = point_embedding + can_bus

        feat_flatten, spatial_shapes = self.get_key_and_key_pos(mlvl_feats, img_metas)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=point_embedding.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3
        ).contiguous()  # (num_cam, H*W, bs, embed_dims)
        point_embed, point_inter_ref_points = self.encoder(
            query=point_queries,
            key=feat_flatten,
            value=feat_flatten,
            ref_points=point_ref_points,
            query_pos=point_pos,
            key_pos=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=point_reg_branch,
            img_metas=img_metas,
        )

        pts_embeds = self.pts_embedding.weight
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        instance_embeds = instance_embeds + pts_embeds.unsqueeze(0)
        instance_embeds = instance_embeds.flatten(0, 1).unsqueeze(0).repeat(bs, 1, 1)
        out_dec = self.decoder(instance_embeds, pts_embeds, point_embed[-1], point_inter_ref_points[-1].detach(), reg_branch, None)
        return point_embed, out_dec, point_ref_points, point_inter_ref_points


@MODELS.register_module()
class PMNDecoderTest(BaseModule):
    def __init__(self, embed_dims=256, return_intermediate=True, num_layers=3, num_points=20, distance=0.1):
        super(PMNDecoderTest, self).__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_points = num_points
        self.num_head = 8
        decoder_layer = PMNDecoderLayer(d_model=self.embed_dims, nhead=self.num_head)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate
        self.distance = distance
        # self.line_encoding = nn.Linear(20 * embed_dims, embed_dims)

    def forward(self, instance_embeds, pts_embeds, memory, key_points, reg_branch, cls_mask, *args, **kwargs):
        output = instance_embeds
        intermediate = []
        for lvl, layer in enumerate(self.layers):
            output = layer(output, pts_embeds, memory, memory_mask=None, *args, **kwargs)
            intermediate.append(output)
        return torch.stack(intermediate)


@MODELS.register_module()
class PMNDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, num_points=20, **kwargs):
        super(PMNDecoderLayer, self).__init__(batch_first=True, **kwargs)
        self.num_points = num_points
        self.embed_dims = 256
        self.linear = nn.Linear(in_features=self.embed_dims * num_points, out_features=self.embed_dims)

    def forward(
            self,
            instance_embeds: Tensor,
            pts_embeds: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> Tensor:
        bs = instance_embeds.shape[0]
        instance_embeds = instance_embeds.reshape(bs, -1, self.num_points, self.embed_dims).flatten(2)
        instance_embeds = self.linear(instance_embeds)
        x = self.norm1(instance_embeds + self._sa_block(instance_embeds, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
        pts_embeds = pts_embeds.view(1, 1, self.num_points, self.embed_dims)
        x = x.unsqueeze(2) + pts_embeds
        x = x.flatten(1, 2)
        x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
        x = self.norm3(x + self._ff_block(x))
        return x
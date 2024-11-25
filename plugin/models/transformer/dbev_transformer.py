import torch
from mmdet.models import inverse_sigmoid
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, xavier_init
from torch import nn
from torch.nn.init import normal_
from plugin.models.utils.utils import _get_clones
import numpy as np


@MODELS.register_module()
class DBEVNetTransformer(BaseModule):
    def __init__(
            self,
            encoder=None,
            decoder=None,
            positional_encoding=None,
            pc_range=None,
            num_points=20,
            num_line=50,
            coord_dim=2,
            bev_size=(20, 40),
            init_cfg=None,
            num_feature_levels=4,
    ):
        super(DBEVNetTransformer, self).__init__(init_cfg=init_cfg)
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
        self.bev_size = bev_size
        self.use_cams_embeds = True
        self.init_layers()

    def init_layers(self):
        x_coords = torch.linspace(0.05, 0.95, self.bev_size[0], dtype=torch.float32)
        y_coords = torch.linspace(0.05, 0.95, self.bev_size[1], dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        coords = torch.stack((y_grid.flatten(), x_grid.flatten()), dim=-1)
        self.reference_points = coords
        # self.point_ref_points = nn.Linear(self.embed_dims, self.coord_dim)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        # self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.position_encoder = nn.Sequential(
            nn.Conv2d(3, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
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

        # self.pts_embedding = nn.Embedding(self.num_points, self.embed_dims)
        self.query_embedding = nn.Sequential(
            nn.Linear(2, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
        )

        self.query_reference_points = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, 2 * self.num_points),
        )

    def init_weights(self):
        # follow the official DETR to init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                xavier_init(m, distribution="uniform")
        normal_(self.level_embeds)
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

    def format_feats(self, mlvl_feats):
        bs = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            # if self.use_cams_embeds:
            #     feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        return feat_flatten, spatial_shapes, level_start_index

    def forward(
            self, mlvl_feats, point_embedding, object_query_embeds, point_reg_branch, point_cls_branch, reg_branch, img_metas, **kwargs
    ):
        bs = mlvl_feats[0].size(0)
        point_embedding = point_embedding.unsqueeze(0).repeat(bs, 1, 1)
        # can_bus = point_embedding.new_tensor(
        #     [each["ego2global_translation"] + each["ego2global_rotation"] for each in img_metas]
        # )  # [:, :]
        # can_bus = self.can_bus_mlp(can_bus)[:, None, :]
        # point_embedding = point_embedding + can_bus

        point_reference_points = self.reference_points.to(point_embedding.device)
        query_pos = self.query_embedding(point_reference_points)
        point_reference_points = point_reference_points.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)

        point_embedding = point_embedding + query_pos

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
            query=point_embedding,
            key=feat_flatten,
            value=feat_flatten,
            ref_points=point_reference_points,
            query_pos=None,
            key_pos=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=point_reg_branch,
            img_metas=img_metas,
        )

        # pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        # instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        # object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)
        # tgt = object_query_embeds.unsqueeze(0).expand(bs, -1, -1)
        # init_reference = self.line_reference_points(tgt).float().sigmoid()
        # inter_states, inter_references = self.decoder(
        #     query=tgt.transpose(1, 0),
        #     key=None,
        #     value=point_embed[-1].transpose(1, 0),
        #     query_pos=None,
        #     reference_points=init_reference,
        #     reg_branches=reg_branch,
        #     cls_branches=None,
        #     spatial_shapes=torch.tensor([[self.bev_size[1], self.bev_size[0]]], device=tgt.device),
        #     level_start_index=torch.tensor([0], device=tgt.device),
        # )

        query_pos, query = torch.split(object_query_embeds, self.embed_dims, dim=1)
        reference_points = self.query_reference_points(query_pos).sigmoid()
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
        init_reference_out = reference_points
        query_pos = None

        query = query.permute(1, 0, 2)
        bev_embed = point_embed[-1].permute(1, 0, 2)

        feat_flatten, feat_spatial_shapes, feat_level_start_index \
            = self.format_feats(mlvl_feats)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branch,
            cls_branches=None,
            spatial_shapes=torch.tensor([[self.bev_size[1], self.bev_size[0]]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            mlvl_feats=mlvl_feats,
            feat_flatten=feat_flatten,
            feat_spatial_shapes=feat_spatial_shapes,
            feat_level_start_index=feat_level_start_index,
            **kwargs)

        # out_dec = self.decoder(tgt, point_embed[-1], point_inter_ref_points[-1].detach(), reg_branch, None)
        return point_embed, inter_states, point_reference_points, point_inter_ref_points, init_reference_out, inter_references

# @MODELS.register_module()
# class DBEVNetTransformer(BaseModule):
#     def __init__(
#             self,
#             encoder=None,
#             decoder=None,
#             positional_encoding=None,
#             pc_range=None,
#             num_points=20,
#             num_line=50,
#             coord_dim=2,
#             bev_size=(20, 40),
#             init_cfg=None,
#             num_feature_levels=4,
#     ):
#         super(DBEVNetTransformer, self).__init__(init_cfg=init_cfg)
#         self.encoder = MODELS.build(encoder)
#         self.embed_dims = self.encoder.embed_dims
#         self.decoder = MODELS.build(decoder)
#         self.positional_encoding = MODELS.build(positional_encoding)
#         self.num_feature_levels = num_feature_levels
#         self.num_points = num_points
#         self.num_line = num_line
#         self.coord_dim = coord_dim
#         self.num_cams = 6
#         self.pc_range = pc_range
#         self.bev_size = bev_size
#         self.init_layers()

#     def init_layers(self):
#         x_coords = torch.linspace(0.05, 0.95, self.bev_size[0], dtype=torch.float32)
#         y_coords = torch.linspace(0.05, 0.95, self.bev_size[1], dtype=torch.float32)
#         x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
#         coords = torch.stack((y_grid.flatten(), x_grid.flatten()), dim=-1)
#         self.reference_points = coords
#         # self.point_ref_points = nn.Linear(self.embed_dims, self.coord_dim)
#         self.level_embeds = nn.Parameter(
#             torch.Tensor(self.num_feature_levels, self.embed_dims)
#         )
#         # self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
#         self.position_encoder = nn.Sequential(
#             nn.Conv2d(3 * 64, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
#         )
#         self.adapt_pos3d = nn.Sequential(
#             nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
#         )
#         self.instance_embedding = nn.Embedding(self.num_line, self.embed_dims)
#         self.pts_embedding = nn.Embedding(self.num_points, self.embed_dims)
#         self.query_embedding = nn.Sequential(
#             nn.Linear(2, self.embed_dims // 2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims // 2, self.embed_dims),
#         )

#     def init_weights(self):
#         # follow the official DETR to init parameters
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if hasattr(m, "weight") and m.weight.dim() > 1:
#                 xavier_init(m, distribution="uniform")
#         normal_(self.level_embeds)
#         self._is_init = True

#     def position_embeding(self, img_feats, img_metas):
#         eps = 1e-5
#         pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
#         B, N, C, H, W = img_feats.shape
#         device = img_feats.device
#         coords_h = torch.arange(H, device=device).float() * pad_h / H
#         coords_w = torch.arange(W, device=device).float() * pad_w / W

#         # index = torch.arange(start=0, end=64, step=1, device=device).float()
#         # index_1 = index + 1
#         # bin_size = (self.pc_range[4] - 1) / (64 * (1 + 64))
#         # coords_d = self.pc_range[4] - 0.1 - bin_size * index * index_1

#         index = torch.arange(start=0, end=64, step=1, device=device).float()
#         bin_size = (self.pc_range[4] - 1) / 64
#         coords_d = 1 + bin_size * index

#         D = coords_d.shape[0]
#         coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3,
#                                                                                                     0).contiguous()  # W, H, D, 3
#         coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
#         coords[..., :2] = coords[..., :2] * coords[..., 2:3]

#         img2lidars = []
#         for img_meta in img_metas:
#             img2lidar = []
#             for i in range(len(img_meta['lidar2img'])):
#                 img2lidar.append(torch.inverse(torch.from_numpy(img_meta['lidar2img'][i])))
#             img2lidars.append(torch.stack(img2lidar))
#         img2lidars = torch.stack(img2lidars).float().to(device)  # (B, N, 4, 4)

#         coords = coords[None, None, :, :, :, :, None].repeat(B, N, 1, 1, 1, 1, 1)
#         img2lidars = img2lidars[:, :, None, None, None, :, :].repeat(1, 1, W, H, D, 1, 1)
#         coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
#         coords3d[..., 0:1] = (coords3d[..., 0:1] - self.pc_range[0]) / (
#                 self.pc_range[3] - self.pc_range[0])
#         coords3d[..., 1:2] = (coords3d[..., 1:2] - self.pc_range[0]) / (
#                 self.pc_range[3] - self.pc_range[0])
#         coords3d[..., 2:3] = (coords3d[..., 2:3] - self.pc_range[2]) / (
#                 self.pc_range[5] - self.pc_range[2])

#         coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().flatten(0, 1).flatten(1, 2)
#         coords3d = inverse_sigmoid(coords3d)
#         coords_position_embeding = self.position_encoder(coords3d)

#         return coords_position_embeding.view(B, N, self.embed_dims, H, W)

#     def get_key_and_key_pos(self, mlvl_feats, img_metas):
#         feat_flatten = []
#         spatial_shapes = []
#         for lvl, feat in enumerate(mlvl_feats):
#             bs, num_cam, c, h, w = feat.shape
#             spatial_shape = (h, w)
#             feat_mask = torch.zeros((bs, num_cam, h, w), device=feat.device).to(feat.dtype)

#             with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
#                 pos_embed = self.position_embeding(feat, img_metas)
#             sin_embed = self.positional_encoding(feat_mask)
#             sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(feat.size())
#             key_pos = (pos_embed + sin_embed).to(feat.dtype)
#             feat = feat + key_pos
#             feat = feat.flatten(3).permute(1, 0, 3, 2).contiguous()
#             # key_pos = key_pos.flatten(3).permute(1, 0, 3, 2).contiguous().to(feat.dtype)
#             # feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
#             feat = feat + self.level_embeds[None, None, lvl: lvl + 1, :].to(feat.dtype)
#             spatial_shapes.append(spatial_shape)
#             feat_flatten.append(feat)

#         feat_flatten = torch.cat(feat_flatten, 2)
#         return feat_flatten, spatial_shapes

#     def forward(
#             self, mlvl_feats, point_embedding, point_reg_branch, point_cls_branch, reg_branch, img_metas
#     ):
#         bs = mlvl_feats[0].size(0)
#         point_embedding = point_embedding.unsqueeze(0).repeat(bs, 1, 1)
#         # can_bus = point_embedding.new_tensor(
#         #     [each["ego2global_translation"] + each["ego2global_rotation"] for each in img_metas]
#         # )  # [:, :]
#         # can_bus = self.can_bus_mlp(can_bus)[:, None, :]
#         # point_embedding = point_embedding + can_bus

#         reference_points = self.reference_points.to(point_embedding.device)
#         query_pos = self.query_embedding(reference_points)
#         reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)
#         query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1)

#         point_embedding = point_embedding + query_pos

#         feat_flatten, spatial_shapes = self.get_key_and_key_pos(mlvl_feats, img_metas)
#         spatial_shapes = torch.as_tensor(
#             spatial_shapes, dtype=torch.long, device=point_embedding.device
#         )
#         level_start_index = torch.cat(
#             (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
#         )

#         feat_flatten = feat_flatten.permute(
#             0, 2, 1, 3
#         ).contiguous()  # (num_cam, H*W, bs, embed_dims)
#         point_embed, point_inter_ref_points = self.encoder(
#             query=point_embedding,
#             key=feat_flatten,
#             value=feat_flatten,
#             ref_points=reference_points,
#             query_pos=None,
#             key_pos=None,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#             reg_branches=point_reg_branch,
#             img_metas=img_metas,
#         )

#         pts_embeds = self.pts_embedding.weight.unsqueeze(0)
#         instance_embeds = self.instance_embedding.weight.unsqueeze(1)
#         object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)
#         tgt = object_query_embeds.unsqueeze(0).expand(bs, -1, -1)
#         # init_reference = self.line_reference_points(tgt).float().sigmoid()
#         # inter_states, inter_references = self.decoder(
#         #     query=tgt.transpose(1, 0),
#         #     key=None,
#         #     value=point_embed[-1].transpose(1, 0),
#         #     query_pos=None,
#         #     reference_points=init_reference,
#         #     reg_branches=reg_branch,
#         #     cls_branches=None,
#         #     spatial_shapes=torch.tensor([[self.bev_size[1], self.bev_size[0]]], device=tgt.device),
#         #     level_start_index=torch.tensor([0], device=tgt.device),
#         # )

#         out_dec = self.decoder(tgt, point_embed[-1], point_inter_ref_points[-1].detach(), reg_branch, None)
#         return point_embed, out_dec, reference_points, point_inter_ref_points

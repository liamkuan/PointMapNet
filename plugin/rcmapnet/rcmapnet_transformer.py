from typing import Optional, Union, Callable

import numpy as np
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models import inverse_sigmoid
from torch import nn, Tensor
from torch.nn.init import normal_

from plugin.models.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from plugin.models.utils.utils import _get_clones
from mmengine.model import BaseModule, constant_init, xavier_init
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import math
from mmdet3d.registry import MODELS
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@MODELS.register_module()
class RCMapNetTransformer(BaseModule):
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
        super(RCMapNetTransformer, self).__init__(init_cfg=init_cfg)
        self.encoder = MODELS.build(encoder)
        self.embed_dims = self.encoder.embed_dims
        self.decoder = MODELS.build(decoder)
        self.positional_encoding = MODELS.build(positional_encoding)

        self.depth_model = DepthModel()

        self.num_feature_levels = num_feature_levels
        self.num_points = num_points
        self.num_line = num_line
        self.coord_dim = coord_dim
        self.num_cams = 6
        self.pc_range = pc_range

        self.bev_w = 60  # 分成多少份
        self.bev_h = 120
        self.init_layers()

    def init_layers(self):
        self.point_ref_points = nn.Linear(self.embed_dims, self.coord_dim)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        # self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.position_encoder = nn.Sequential(
            nn.Conv2d(3, self.embed_dims, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.instance_embedding = nn.Embedding(self.num_line, self.embed_dims * 2)
        self.pts_embedding = nn.Embedding(self.num_points, self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, self.coord_dim)  # TODO, this is a hack

        self.bev_embedding = nn.Embedding(self.bev_w * self.bev_h, self.embed_dims)

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
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        self._is_init = True

    def position_embeding(self, img_depth, img_metas):
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, _, H, W = img_depth.shape
        coords_h = torch.arange(H, device=img_depth.device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_depth.device).float() * pad_w / W

        coords_d = img_depth * self.pc_range[4]
        coords_d = coords_d.permute(0, 1, 4, 3, 2)  # B, N, W, H, 1
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
        coords3d = torch.clamp(coords3d, 0, 1)
        coords3d = coords3d.permute(0, 1, 4, 3, 2).flatten(0, 1)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H // 2, W // 2).contiguous()

    def get_feats(self, mlvl_feats, pos_3d):
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat_mask = torch.zeros((bs, num_cam, h, w), device=feat.device).to(feat.dtype)

            sin_embed = self.positional_encoding(feat_mask)
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(feat.size()).to(feat.dtype)
            feat = feat + sin_embed
            if lvl == 0:
                feat = feat + pos_3d.to(feat.dtype)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
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

        image_depths = self.depth_model(*mlvl_feats)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            pos_3d = self.position_embeding(image_depths, img_metas)
        feat_flatten, spatial_shapes = self.get_feats(mlvl_feats, pos_3d)
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

        bs, num_point, embed_dim = point_embed[-1].shape
        bev_embed = self.bev_embedding.weight.to(point_embed.dtype).unsqueeze(0).repeat(bs, 1, 1)
        bev_points = point_inter_ref_points[-1]
        bev_pos = bev_points[:, :, 0:1] // (1 / self.bev_w) + self.bev_w * (bev_points[:, :, 1:2] // (1 / self.bev_h))
        bev_pos = bev_pos.to(torch.int64)
        bev_embed = torch.scatter(bev_embed, dim=1, index=bev_pos.repeat(1, 1, embed_dim), src=point_embed[-1])

        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)

        query_pos, query = torch.split(object_query_embeds, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branch,
            spatial_shapes=torch.tensor([[self.bev_w, self.bev_h]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )
        inter_states = inter_states.transpose(1, 2)
        return point_embed, inter_states, point_ref_points, point_inter_ref_points, init_reference_out, inter_references, image_depths


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        # 处理第一层特征的卷积层
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # 处理第二层特征的卷积层
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # 融合两层特征的卷积层
        self.conv_fusion = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu_fusion = nn.ReLU()
        # 上采样模块，参考论文中的UpProj模块思想
        self.upsample1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_upsample1 = nn.ReLU()
        self.upsample2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_upsample2 = nn.ReLU()
        self.upsample3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_upsample3 = nn.ReLU()
        self.upsample4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu_upsample4 = nn.ReLU()
        self.bilinear_upsample = nn.Upsample(size=(60, 100), mode='bilinear', align_corners=False)
        # 输出深度估计的卷积层
        self.conv_out = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, feature1, feature2):
        bs, num_cam, _, _, _ = feature1.shape
        feature1 = feature1.flatten(0, 1)
        feature2 = feature2.flatten(0, 1)
        # 处理第一层特征
        out1 = self.relu1(self.conv1(feature1))
        # 处理第二层特征
        out2 = self.relu2(self.conv2(feature2))
        # 上采样第二层特征以匹配第一层特征的尺寸
        out2_up = nn.functional.interpolate(out2, size=(30, 50), mode='bilinear', align_corners=False)
        # 融合两层特征
        fused = torch.cat([out1, out2_up], dim=1)
        fused = self.relu_fusion(self.conv_fusion(fused))
        # 参考论文中的上采样流程
        upsampled = self.relu_upsample1(self.upsample1(fused))
        upsampled = self.relu_upsample2(self.upsample2(upsampled))
        upsampled = self.bilinear_upsample(upsampled)
        upsampled = self.relu_upsample3(self.upsample3(upsampled))
        upsampled = self.relu_upsample4(self.upsample4(upsampled))
        # 输出深度估计
        depth_estimate = self.conv_out(upsampled).float().sigmoid()
        depth_estimate = depth_estimate.reshape(bs, num_cam, 1, 60, 100)
        return depth_estimate


@MODELS.register_module()
class RCMapNetDecoder(TransformerLayerSequence):
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(RCMapNetDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(
            self,
            query,
            *args,
            reference_points=None,
            reg_branches=None,
            key_padding_mask=None,
            **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
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

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@MODELS.register_module()
class SparseSelfAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_levels, num_levels * num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * num_levels, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        # self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',

                **kwargs):

        if value is None:
            assert self.batch_first
            bs, n, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, n, c)

            # value = torch.cat([query, query], 0)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs * self.num_bev_queue,
                              num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5) \
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6) \
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points \
                                 * reference_points[:, :, None, :, None, 2:] \
                                 * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

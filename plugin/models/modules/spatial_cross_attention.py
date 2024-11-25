import warnings

import math
import numpy as np
import mmcv.ops
import torch
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, xavier_init, constant_init
from torch import nn
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

from plugin.models.utils.utils import _get_clones


@MODELS.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MODELS.build(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                            key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len,
                                                                                           D, 2),
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len,
                                                                                      self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@MODELS.register_module()
class SpatialCrossAttentionV1(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 dropout=0.1,
                 init_cfg=None,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttentionV1, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MODELS.build(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        bs, num_query, _ = query.size()

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).contiguous().reshape(
            bs * num_cams, l, embed_dims)
        value = value.permute(2, 0, 1, 3).contiguous().reshape(
            bs * num_cams, l, embed_dims)
        query = query.unsqueeze(1).repeat(1, num_cams, 1, 1).flatten(0, 1)
        # num_cam B num_query D coord -> B*num_cam num_query D coord
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4).contiguous().flatten(0, 1)
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        queries = self.deformable_attention(query=query,
                                            key=key, value=value,
                                            reference_points=reference_points_cam,
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, num_query,
                                                                                      self.embed_dims)
        mask = bev_mask[..., 0:1].permute(1, 0, 2, 3).contiguous()
        data_tensor = queries.masked_fill(~mask, 0.0)
        queries = torch.mean(data_tensor.permute(0, 2, 1, 3).contiguous(), dim=2)
        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual


@MODELS.register_module()
class SpatialCrossAttentionV2(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttentionV2, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MODELS.build(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        num_cam, bs, num_query, D = reference_points_cam.shape[:4]
        bev_mask = bev_mask.flatten(0, 1)
        reference_points_cam = reference_points_cam.flatten(0, 1)
        batch_index = torch.arange(bs)[:, None].repeat(1, num_cam).flatten(0, 1)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bev_mask.size(0), max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bev_mask.size(0), max_len, D, 2])

        for i, reference_points_per_img in enumerate(reference_points_cam):
            index_query_per_img = indexes[i]
            queries_rebatch[i, :len(index_query_per_img)] = query[batch_index[i], index_query_per_img]
            reference_points_rebatch[i, :len(index_query_per_img)] = reference_points_per_img[index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, embed_dims)

        queries = self.deformable_attention(query=queries_rebatch,
                                            key=key, value=value,
                                            reference_points=reference_points_rebatch,
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index)
        for i, index_query_per_img in enumerate(indexes):
            slots[batch_index[i], index_query_per_img] += queries[i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.view(num_cam, bs, num_query).permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@MODELS.register_module()
class MSDeformableAttention3D(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=72,
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
        self.batch_first = batch_first
        self.output_proj = None

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
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # constant_init(self.sampling_offsets, 0.)
        # thetas = torch.arange(
        #     self.num_heads,
        #     dtype=torch.float32,
        #     device=self.sampling_offsets.weight.device,) * (2.0 * math.pi / self.num_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init /
        #              grid_init.abs().max(-1, keepdim=True)[0]).view(
        #     self.num_heads, 1, 1,
        #     2).repeat(1, self.num_levels, self.num_points, 1)
        # for i in range(self.num_points):
        #     grid_init[:, :, i, :] *= i + 1
        #
        # self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        # xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2).contiguous()
            value = value.permute(1, 0, 2).contiguous()

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        assert reference_points.shape[-1] == 2
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

        bs, num_query, num_Z_anchors, xy = reference_points.shape

        sampling_offsets = sampling_offsets / \
                           offset_normalizer[None, None, None, :, None, :]
        bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
        sampling_offsets = sampling_offsets.view(
            bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
        sampling_locations = reference_points[:, :, None, None, None, :, :] + sampling_offsets
        bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
        assert num_all_points == num_points * num_Z_anchors

        sampling_locations = sampling_locations.view(
            bs, num_query, num_heads, num_levels, num_all_points, xy)
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations.contiguous(),
            attention_weights, self.im2col_step)
        if not self.batch_first:
            output = output.permute(1, 0, 2).contiguous()

        return output


@MODELS.register_module()
class MSDeformableAttention3DV1(BaseModule):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=72,
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
        self.batch_first = batch_first
        self.output_proj = None

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
        # self.sampling_offsets = nn.Linear(
        #     embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # constant_init(self.sampling_offsets, 0.)
        # thetas = torch.arange(
        #     self.num_heads,
        #     dtype=torch.float32,
        #     device=self.sampling_offsets.weight.device,) * (2.0 * math.pi / self.num_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init /
        #              grid_init.abs().max(-1, keepdim=True)[0]).view(
        #     self.num_heads, 1, 1,
        #     2).repeat(1, self.num_levels, self.num_points, 1)
        # for i in range(self.num_points):
        #     grid_init[:, :, i, :] *= i + 1
        #
        # self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        # xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2).contiguous()
            value = value.permute(1, 0, 2).contiguous()

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        # sampling_offsets = self.sampling_offsets(query).view(
        #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        assert reference_points.shape[-1] == 2
        # offset_normalizer = torch.stack(
        #     [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

        bs, num_query, num_Z_anchors, xy = reference_points.shape
        reference_points = reference_points.view(bs, num_query, 2, 2, 2)
        point1 = reference_points[:, :, :, 0, :]  # Coordinates of the first point
        point2 = reference_points[:, :, :, 1, :]  # Coordinates of the second point

        # Calculate the middle points
        middle_point1 = point1 + 0.25 * (point2 - point1)
        middle_point2 = point1 + 0.75 * (point2 - point1)

        # Concatenate the points along the third dimension
        output_reference_points = torch.stack([point1, middle_point1, middle_point2, point2], dim=3)
        output_reference_points = output_reference_points.flatten(2, 3)
        output_reference_points = output_reference_points[:, :, None, None, :, :]
        output_reference_points = output_reference_points.expand(-1, -1, self.num_heads, self.num_levels, -1, -1).contiguous()
        # sampling_offsets = sampling_offsets / \
        #                    offset_normalizer[None, None, None, :, None, :]
        # bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
        # sampling_offsets = sampling_offsets.view(
        #     bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
        # sampling_locations = reference_points + sampling_offsets
        # bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
        # assert num_all_points == num_points * num_Z_anchors
        #
        # sampling_locations = sampling_locations.view(
        #     bs, num_query, num_heads, num_levels, num_all_points, xy)
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, output_reference_points,
            attention_weights, self.im2col_step)
        if not self.batch_first:
            output = output.permute(1, 0, 2).contiguous()

        return output

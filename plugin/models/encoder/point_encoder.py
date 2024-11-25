import copy
import warnings

import numpy as np
import torch
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmdet.models import inverse_sigmoid
from mmdet3d.registry import MODELS
from torch import nn


@MODELS.register_module()
class PointEncoder(TransformerLayerSequence):
    def __init__(
            self,
            *args,
            return_intermediate=False,
            num_points=20,
            coord_dim=2,
            pc_range=[-32, -16, -3.0, 32, 16, 2.0],
            **kwargs,
    ):
        super(PointEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.pc_range = pc_range
        self.fusion_point_query = False
        self.embed_dims = 256
        self.init_layers()

    def init_layers(self):
        if self.fusion_point_query:
            self.fusion_model = nn.Linear(self.embed_dims * 3, self.embed_dims)

    def point_sampling(self, reference_points, img_metas):
        pc_range = self.pc_range
        reference_points = reference_points.float()

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
                reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
                reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
                reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3).contiguous()
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = (
            torch.matmul(lidar2img, reference_points).squeeze(-1).float()
        )
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["pad_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["pad_shape"][0][0]

        bev_mask = (
                bev_mask
                & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0)
        )
        # bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4).contiguous()
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).contiguous().squeeze(-1)

        return reference_points_cam, bev_mask

    def add_height(self, ref_points):
        num_point_in_height = 4
        zs = torch.linspace(
            0.1, 0.5, num_point_in_height, dtype=torch.float32, device=ref_points.device
        ).view(1, -1, 1, 1, 1)
        ref_points = ref_points.view(*ref_points.shape[:-1], 3, 2)
        bs, num_query, num_point, dim = ref_points.shape
        zs = zs.repeat(bs, 1, num_query, num_point, 1)
        ref_points = ref_points.unsqueeze(1).repeat(1, num_point_in_height, 1, 1, 1)
        ref_points = torch.cat([ref_points, zs], dim=-1)
        ref_points = ref_points.permute(0, 3, 1, 2, 4).contiguous().flatten(1, 2)
        return ref_points
        # z = 0.2167
        # ref_points = ref_points.view(*ref_points.shape[:-1], 3, 2)
        # ref_points = torch.cat([ref_points, torch.ones_like(ref_points[..., 0:1]) * z], dim=-1)
        # ref_points = ref_points.permute(0, 2, 1, 3)
        # return ref_points

    def forward(
            self,
            query,
            key,
            value,
            query_pos,
            ref_points,
            spatial_shapes,
            level_start_index,
            reg_branches,
            img_metas,
            **kwargs,
    ):
        output = query
        intermediate = []

        ref_3d = self.add_height(ref_points)

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_metas)
        inter_ref_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                # key_pos=key_pos,
                ref_3d=ref_3d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                **kwargs,
            )

            if self.fusion_point_query:
                output = output.view([output.size(0), 50, 20, output.size(-1)])
                temp1 = output.clone()
                temp2 = output.clone()
                temp1[:, :, 1:] = output[:, :, :-1]
                temp2[:, :, :-1] = output[:, :, 1:]
                joint_query = torch.cat([temp1, output, temp2], dim=-1)
                output = self.fusion_model(joint_query)
                output = output.flatten(1, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = inverse_sigmoid(ref_points) + tmp
                new_reference_points = new_reference_points.sigmoid()
                ref_points = new_reference_points.detach()
                ref_3d = self.add_height(ref_points)
                reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_metas)

            if self.return_intermediate:
                intermediate.append(output)
                inter_ref_points.append(ref_points)

        if self.return_intermediate:
            return torch.stack(intermediate), inter_ref_points

        return output, inter_ref_points


@MODELS.register_module()
class PointEncoderV1(TransformerLayerSequence):
    def __init__(
            self,
            *args,
            return_intermediate=False,
            pc_range=[-32, -16, -3.0, 32, 16, 2.0],
            zs=[0.2, 0.3, 0.4, 0.5],
            num_points_in_pillar=4,
            **kwargs,
    ):
        super(PointEncoderV1, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.zs = zs
        self.num_points_in_pillar = num_points_in_pillar

    def point_sampling(self, reference_points, img_metas):
        pc_range = self.pc_range
        reference_points = reference_points.float()

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
                reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
                reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
                reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3).contiguous()
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = (
            torch.matmul(lidar2img, reference_points).squeeze(-1).float()
        )
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["pad_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["pad_shape"][0][0]

        bev_mask = (
                bev_mask
                & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0)
        )
        # bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4).contiguous()
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).contiguous().squeeze(-1)

        return reference_points_cam, bev_mask

    def add_height(self, ref_points, ):
        zs = ref_points.new_tensor(self.zs).view(1, -1, 1, 1)
        bs, num_query, dim = ref_points.shape
        ref_points = ref_points.unsqueeze(1).repeat(1, self.num_points_in_pillar, 1, 1)
        zs = zs.repeat(bs, 1, num_query, 1)
        ref_points = torch.cat([ref_points, zs], dim=-1)
        return ref_points

    def forward(
            self,
            query,
            key,
            value,
            query_pos,
            key_pos,
            ref_points,
            spatial_shapes,
            level_start_index,
            reg_branches,
            img_metas,
            **kwargs,
    ):
        output = query
        intermediate = []

        ref_3d = self.add_height(ref_points)

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_metas)
        inter_ref_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                ref_3d=ref_3d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                img_metas=img_metas,
                **kwargs,
            )

            tmp = reg_branches[lid](output)
            new_reference_points = inverse_sigmoid(ref_points) + tmp
            ref_points = new_reference_points.sigmoid()
            if lid < self.num_layers - 1:
                ref_3d = self.add_height(ref_points.detach())
                reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_metas)

            intermediate.append(output)
            inter_ref_points.append(ref_points)

        return torch.stack(intermediate), inter_ref_points


@MODELS.register_module()
class PointEncoderTest(TransformerLayerSequence):
    def __init__(
            self,
            *args,
            return_intermediate=False,
            pc_range=[-32, -16, -3.0, 32, 16, 2.0],
            bev_size=(20, 40),
            **kwargs,
    ):
        super(PointEncoderTest, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.bev_size = bev_size

    def point_sampling(self, reference_points, img_metas):
        pc_range = self.pc_range
        reference_points = reference_points.float()

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
                reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
                reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
                reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3).contiguous()
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = (
            torch.matmul(lidar2img, reference_points).squeeze(-1).float()
        )
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["pad_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["pad_shape"][0][0]

        bev_mask = (
                bev_mask
                & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0)
        )
        # bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4).contiguous()
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).contiguous().squeeze(-1)

        return reference_points_cam, bev_mask

    def forward(
            self,
            query,
            key,
            value,
            query_pos,
            key_pos,
            ref_points,
            spatial_shapes,
            level_start_index,
            reg_branches,
            img_metas,
            **kwargs,
    ):
        output = query
        intermediate = []
        init_ref_points = ref_points
        if ref_points.size(-1) == 2:
            ref_3d = add_height(ref_points)
        else:
            ref_3d = ref_points.unsqueeze(1)

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_metas)
        inter_ref_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                key,
                value,
                bev_pos=None,
                query_pos=query_pos,
                key_pos=key_pos,
                ref_2d=init_ref_points,
                ref_3d=ref_3d,
                bev_w=self.bev_size[0],
                bev_h=self.bev_size[1],
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                img_metas=img_metas,
                **kwargs,
            )

            tmp = reg_branches[lid](output)
            new_reference_points = inverse_sigmoid(ref_points) + tmp
            ref_points = new_reference_points.sigmoid()
            if lid < self.num_layers - 1:
                if ref_points.size(-1) == 2:
                    ref_3d = add_height(ref_points.detach())
                else:
                    ref_3d = ref_points.unsqueeze(1)
                reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_metas)

            intermediate.append(output)
            inter_ref_points.append(ref_points)

        return torch.stack(intermediate), inter_ref_points


@MODELS.register_module()
class LineLayer(BaseTransformerLayer):
    def __init__(
            self,
            attn_cfgs,
            feedforward_channels,
            ffn_dropout=0.0,
            operation_order=None,
            act_cfg=dict(type="ReLU"),
            norm_cfg=dict(type="LN"),
            ffn_num_fcs=2,
            embed_dims=256,
            **kwargs,
    ):
        super(LineLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=dict(
                type="FFN",
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=ffn_num_fcs,
                ffn_drop=ffn_dropout,
                act_cfg=act_cfg,
            ),
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            batch_first=True,
            **kwargs,
        )
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

    def forward(
            self,
            query,
            key=None,
            value=None,
            bev_pos=None,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            ref_3d=None,
            reference_points_cam=None,
            mask=None,
            spatial_shapes=None,
            level_start_index=None,
            **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
                query = query.to(key.dtype)

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
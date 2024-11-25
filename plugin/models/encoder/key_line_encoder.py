import numpy as np
import torch
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models import inverse_sigmoid
from mmdet3d.registry import MODELS


@MODELS.register_module()
class KeyLineEncoder(TransformerLayerSequence):
    def __init__(self, *args, pc_range=[-32, -16, -3.0, 32, 16, 2.0], **kwargs):
        super(KeyLineEncoder, self).__init__(*args, **kwargs)
        self.pc_range = pc_range

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

        reference_points = reference_points.permute(1, 0, 2, 3)
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

        reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)

        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        bev_mask = (
                bev_mask
                & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0)
        )

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def add_height(self, ref_points):
        num_point_in_height = 8
        zs = torch.linspace(
            0.1, 0.5, num_point_in_height, dtype=torch.float32, device=ref_points.device
        ).view(1, -1, 1, 1, 1)
        bs, num_query, num_point, dim = ref_points.shape
        zs = zs.repeat(bs, 1, num_query, num_point, 1)
        ref_points = ref_points.unsqueeze(1).repeat(1, num_point_in_height, 1, 1, 1)
        ref_points = torch.cat([ref_points, zs], dim=-1)
        ref_points = ref_points.permute(0, 3, 1, 2, 4).flatten(1, 2)
        return ref_points

    def forward(
            self,
            query,
            key,
            value,
            query_pos,
            ref_points,
            spatial_shapes,
            level_start_index,
            key_line_branch,
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
                ref_3d=ref_3d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                **kwargs,
            )

            if key_line_branch is not None:
                tmp = key_line_branch[lid](output)
                tmp = tmp.reshape(*tmp.shape[:-1], 2, 2)
                new_reference_points = inverse_sigmoid(ref_points) + tmp
                ref_points = new_reference_points.sigmoid()

                ref_3d = self.add_height(ref_points)
                reference_points_cam, bev_mask = self.point_sampling(ref_3d, img_metas)

            inter_ref_points.append(ref_points)
            intermediate.append(output)

        return torch.stack(intermediate), inter_ref_points

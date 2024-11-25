# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from torch import Tensor, autocast
from typing import List, Dict, Optional
from plugin.petr.models.utils.grid_mask import GridMask


@MODELS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 data_preprocessor=None):
        super(Petr3D, self).__init__(pts_voxel_encoder=pts_voxel_encoder,
                                     pts_middle_encoder=pts_middle_encoder,
                                     img_backbone=img_backbone,
                                     pts_backbone=pts_backbone,
                                     img_neck=img_neck,
                                     pts_neck=pts_neck,
                                     pts_bbox_head=pts_bbox_head,
                                     img_roi_head=img_roi_head,
                                     img_rpn_head=img_rpn_head,
                                     data_preprocessor=data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.query_dict = {}
        self.ref_point_dict = {}

    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img = img.squeeze()
                else:
                    B, N, C, H, W = img.size()
                    img = img.flatten(0, 1)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)

        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points."""
        imgs = batch_inputs_dict.get('imgs', None)
        if imgs.shape[1] == 12:
            current_frame = imgs[:, :6]
            prev_frame = imgs[:, 6:].detach()

            current_feats = self.extract_img_feat(current_frame, batch_input_metas)
            with torch.no_grad():
                prev_feats = self.extract_img_feat(prev_frame, batch_input_metas)

            img_feats = []
            for a, b in zip(current_feats, prev_feats):
                img_feats.append(torch.cat([a, b], dim=1))

        else:
            img_feats = self.extract_img_feat(imgs, batch_input_metas)
        return img_feats, None

    def loss_imgs(self, x: List[Tensor],
                  batch_data_samples: List[Det3DDataSample], **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            x = [item.float() for item in x]
            outs = self.pts_bbox_head(x, batch_input_metas)
            targets = [item.target for item in batch_data_samples]
            losses = self.pts_bbox_head.loss(outs, targets)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, _ = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        outs = self.pts_bbox_head(img_feats, batch_input_metas)
        data_instance_3d = []
        for i in range(len(batch_data_samples)):
            instance_data = InstanceData()
            instance_data['pred_logits'] = outs['pred_logits'][i]
            instance_data['pred_lines'] = outs['pred_lines'][i]
            data_instance_3d.append(instance_data)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 data_instance_3d,
                                                 None)
        return detsamples
    
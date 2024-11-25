# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from typing import Optional, List
import torch
from torch import Tensor, autocast
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.structures import Det3DDataSample
from plugin.petr.models.utils.grid_mask import GridMask


@MODELS.register_module()
class Petr3D_seg(MVXTwoStageDetector):
    """Detr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,):
        super(Petr3D_seg, self).__init__(img_backbone=img_backbone, 
                                         img_neck=img_neck,
                                         pts_bbox_head=pts_bbox_head,
                                         train_cfg=train_cfg,
                                         test_cfg=test_cfg,
                                         init_cfg=init_cfg,
                                         data_preprocessor=data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.flatten(0, 1)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            # if isinstance(img_feats, dict):
            #     img_feats = list(img_feats.values())
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
            maps = [item.maps for item in batch_data_samples]
            # targets = [item.target for item in batch_data_samples]
            # losses = self.pts_bbox_head.loss(outs, maps, targets)
            losses = self.pts_bbox_head.loss(outs, maps)
        return losses

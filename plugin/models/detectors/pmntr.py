from typing import List, Dict, Optional

import torch
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptSampleList
from mmengine.analysis import get_model_complexity_info
from mmengine.structures import InstanceData
from torch import Tensor, autocast

from plugin.petr.models.utils.grid_mask import GridMask


@MODELS.register_module()
class PMNTR(Base3DDetector):

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 head=None,
                 data_preprocessor=None):
        super(PMNTR, self).__init__(data_preprocessor=data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.head = MODELS.build(head)

    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        B, N, C, H, W = img.size()
        img = img.flatten(0, 1)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
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
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        return img_feats

    def loss(self, batch_inputs: dict, batch_data_samples: List[Det3DDataSample]):
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        x = self.extract_feat(batch_inputs, batch_input_metas)
        outs = self.head(x, batch_input_metas)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            raw_points, raw_points_label, point_masks = (batch_inputs['raw_points'], batch_inputs['raw_points_label'],
                                                         batch_inputs['point_masks'])
            gt_labels_3d, gt_lines_3d = batch_inputs['gt_labels_3d'], batch_inputs['gt_lines_3d']
            losses = self.head.loss(outs, gt_labels_3d, gt_lines_3d, raw_points, raw_points_label, point_masks)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        x = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outs = self.head(x, batch_input_metas)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            preds_list = self.head.get_preds(outs)
        data_instance_3d = []
        data_instance_2d = []
        for preds in preds_list:
            instance_data_3d = InstanceData()
            instance_data_2d = InstanceData()
            instance_data_3d['scores'] = preds['scores']
            instance_data_3d['labels'] = preds['labels']
            instance_data_3d['pts'] = preds['pts']

            instance_data_2d['key_scores'] = preds['key_scores']
            instance_data_2d['key_labels'] = preds['key_labels']
            instance_data_2d['key_pts'] = preds['key_pts']
            data_instance_3d.append(instance_data_3d)
            data_instance_2d.append(instance_data_2d)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 data_instance_3d, data_instance_2d)
        return detsamples


@MODELS.register_module()
class PMNV1TR(Base3DDetector):

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 head=None,
                 data_preprocessor=None):
        super(PMNTR, self).__init__(data_preprocessor=data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.head = MODELS.build(head)

    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        B, N, C, H, W = img.size()
        img = img.flatten(0, 1)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
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
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        return img_feats

    def loss(self, batch_inputs: dict, batch_data_samples: List[Det3DDataSample]):
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        x = self.extract_feat(batch_inputs, batch_input_metas)

        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            outs = self.head([item.float() for item in x], batch_input_metas)
            raw_points, raw_points_label, point_masks = (batch_inputs['raw_points'], batch_inputs['raw_points_label'],
                                                         batch_inputs['point_masks'])
            gt_labels_3d, gt_lines_3d = batch_inputs['gt_labels_3d'], batch_inputs['gt_lines_3d']
            losses = self.head.loss(outs, gt_labels_3d, gt_lines_3d, raw_points, raw_points_label, point_masks, batch_input_metas)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        x = self.extract_feat(batch_inputs_dict, batch_input_metas)
        outs = self.head(x, batch_input_metas)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            preds_list = self.head.get_preds(outs)
        data_instance_3d = []
        data_instance_2d = []
        for preds in preds_list:
            instance_data_3d = InstanceData()
            instance_data_2d = InstanceData()
            instance_data_3d['scores'] = preds['scores']
            instance_data_3d['labels'] = preds['labels']
            instance_data_3d['pts'] = preds['pts']

            data_instance_3d.append(instance_data_3d)
            data_instance_2d.append(instance_data_2d)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 data_instance_3d, data_instance_2d)
        return detsamples
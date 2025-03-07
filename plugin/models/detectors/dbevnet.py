import copy
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
class DBEVNet(Base3DDetector):

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 head=None,
                 data_preprocessor=None):
        super(DBEVNet, self).__init__(data_preprocessor=data_preprocessor)
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
            gt_labels_3d, gt_bboxes_3d = batch_inputs['gt_labels_3d'], batch_inputs['gt_bboxes_3d']
            losses = dict()
            losses_pts = self.head.loss(outs, gt_bboxes_3d, gt_labels_3d, raw_points, raw_points_label, point_masks, batch_input_metas)
            losses.update(losses_pts)
            k_one2many = self.head.k_one2many
            multi_gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
            multi_gt_labels_3d = copy.deepcopy(gt_labels_3d)
            for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(zip(multi_gt_bboxes_3d, multi_gt_labels_3d)):
                each_gt_bboxes_3d.instance_list = each_gt_bboxes_3d.instance_list * k_one2many
                each_gt_bboxes_3d.instance_labels = each_gt_bboxes_3d.instance_labels * k_one2many
                multi_gt_labels_3d[i] = each_gt_labels_3d.repeat(k_one2many)
            # import ipdb;ipdb.set_trace()
            one2many_outs = outs['one2many_outs']
            loss_one2many_inputs = [one2many_outs, multi_gt_bboxes_3d, multi_gt_labels_3d, None, None, None, batch_input_metas ]
            loss_dict_one2many = self.head.loss(*loss_one2many_inputs)

            lambda_one2many = self.head.lambda_one2many
            for key, value in loss_dict_one2many.items():
                if key + "_one2many" in losses.keys():
                    losses[key + "_one2many"] += value * lambda_one2many
                else:
                    losses[key + "_one2many"] = value * lambda_one2many
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

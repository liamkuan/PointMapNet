import torch

from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model.wrappers.distributed import MMDistributedDataParallel


@HOOKS.register_module()
class ChangeWeightHook(Hook):
    def __init__(self, epoch=12):
        self.epoch = epoch

    def before_train_epoch(self, runner):
        head = runner.cfg.model.head
        point_reg_weight = head.loss_point_pts.loss_weight
        point_mask_weight = head.loss_point_mask.loss_weight
        point_cls_weight = head.loss_point_cls.loss_weight
        pts_weight = head.loss_pts.loss_weight
        cls_weight = head.loss_cls.loss_weight

        if isinstance(runner.model, MMDistributedDataParallel):
            head = runner.model.module.head
        else:
            head = runner.model.head
        if runner.epoch < self.epoch:
            head.loss_pts.loss_weight = pts_weight * 0.1
            head.loss_cls.loss_weight = cls_weight * 0.1
        else:
            head.loss_pts.loss_weight = pts_weight
            head.loss_cls.loss_weight = cls_weight
            head.loss_point_pts.loss_weight = point_reg_weight * 0.1
            head.loss_point_mask.loss_weight = point_mask_weight * 0.1
            head.loss_point_cls.loss_weight = point_cls_weight * 0.1

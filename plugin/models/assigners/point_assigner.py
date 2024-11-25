import torch
import torch.nn.functional as F
from mmdet.models.task_modules import BaseAssigner, AssignResult
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from mmdet3d.registry import TASK_UTILS, MODELS

from plugin.models.utils.utils import normalize_2d_pts


@TASK_UTILS.register_module()
class PointAssigner(BaseAssigner):

    def __init__(self,
                 use_norm_coord=True,
                 pts_cost=dict(type='PtsL1Cost', weight=5.),
                 cls_cost=dict(type='FocalLossCost', weight=2.0),
                 pc_range=None):
        self.use_norm_coord = use_norm_coord
        self.pts_cost = TASK_UTILS.build(pts_cost)
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.pc_range = pc_range

    def assign(self,
               pts_pred,
               points_score,
               gt_raw_points,
               gt_points_label,
               gt_bboxes_ignore=None,
               eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_raw_points.size(0), pts_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = pts_pred.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)
        assigned_labels = pts_pred.new_full((num_bboxes,),
                                            -1,
                                            dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None
        pred_instances = InstanceData()
        gt_instances = InstanceData()
        pred_instances['scores'] = points_score
        gt_instances['labels'] = gt_points_label
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        _, num_coords = gt_raw_points.shape

        normalized_gt_pts = gt_raw_points
        if self.use_norm_coord:
            normalized_gt_pts = normalize_2d_pts(gt_raw_points, self.pc_range)

        pts_cost = self.pts_cost(pts_pred, normalized_gt_pts)
        cost = cls_cost + pts_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(pts_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(pts_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_points_label[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

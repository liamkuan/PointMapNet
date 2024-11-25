import torch
import torch.nn.functional as F
from mmdet.models.task_modules import BaseAssigner, AssignResult
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from mmdet3d.registry import TASK_UTILS, MODELS

from plugin.models.utils.utils import normalize_2d_pts


@TASK_UTILS.register_module()
class LineAssigner(BaseAssigner):

    def __init__(self,
                 use_norm_coord=True,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 pts_cost=dict(type='ChamferDistance', loss_src_weight=1.0, loss_dst_weight=1.0),
                 pc_range=None):
        self.use_norm_coord = use_norm_coord
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.pts_cost = TASK_UTILS.build(pts_cost)
        self.pc_range = pc_range

    def assign(self,
               cls_pred,
               pts_pred,
               gt_labels,
               gt_pts,
               gt_bboxes_ignore=None,
               eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_pts.size(0), pts_pred.size(0)

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

        # 2. compute the weighted costs
        # classification and bboxcost.
        pred_instances = InstanceData()
        gt_instances = InstanceData()
        pred_instances['scores'] = cls_pred
        gt_instances['labels'] = gt_labels
        cls_cost = self.cls_cost(pred_instances, gt_instances)

        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape
        # _, num_pts_per_gtline, num_coords = gt_pts.shape

        normalized_gt_pts = gt_pts
        if self.use_norm_coord:
            normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        
        num_pts_per_predline = pts_pred.size(1)
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(pts_pred.permute(0, 2, 1).contiguous(), size=(num_pts_per_gtline),
                                                  mode='linear', align_corners=True)
            pts_pred_interpolated = pts_pred_interpolated.permute(0, 2, 1).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_pts)
        pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        # pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)

        cost = cls_cost + pts_cost
        # cost = cls_cost + pts_cost_ordered
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
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels), order_index
        # return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

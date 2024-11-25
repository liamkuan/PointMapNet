import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from mmdet3d.registry import MODELS
from torch import Tensor
from mmdet.models.losses.iou_loss import eiou_loss


@MODELS.register_module()
class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, losses=['labels', 'lines', 'area', 'direction'], enable_weight=False):
        super().__init__()
        self.matcher = MODELS.build(matcher)
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = 0.25
        self.enable_weight = enable_weight

    def loss_labels(self, outputs, targets, indices, num_boxes, num_classes):
        src_logits = outputs['pred_logits']

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int32, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1).long(), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce * self.weight_dict['loss_ce']}
        return losses

    def loss_lines(self, outputs, targets, indices, num_boxes, num_classes):
        idx = _get_src_permutation_idx(indices)
        out_line = outputs['pred_lines'][idx]
        target_line = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # loss_area = loss_between_two_curves(out_line, target_line) / out_line.shape[1]
        loss_line = F.l1_loss(out_line.flatten(1), target_line.flatten(1), reduction='none')

        if self.enable_weight:
            line_weights = torch.cat([t['line_weights'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            loss_line = loss_line.view([*line_weights.shape, out_line.shape[-1]]).sum(dim=-1) * line_weights
            loss_line = loss_line * out_line.shape[1]

        losses = {'loss_line': loss_line.sum() / num_boxes * self.weight_dict['loss_line']}
        return losses

    def loss_area(self, outputs, targets, indices, num_boxes, num_classes):
        idx = _get_src_permutation_idx(indices)
        out_line = outputs['pred_lines'][idx]
        target_line = torch.cat([t['lines'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_area = loss_between_two_curves(out_line, target_line) / out_line.shape[1]
        losses = {'loss_area': loss_area / num_boxes * self.weight_dict['loss_area']}
        return losses

    def loss_direction(self, outputs, targets, indices, num_boxes, num_classes):
        idx = _get_src_permutation_idx(indices)
        out_direction = outputs['pred_directions'][idx]
        target_direction = torch.cat([t['point_directions'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_direction = F.l1_loss(out_direction, target_direction, reduction='none')

        loss_direction = loss_direction / out_direction.shape[1]
        losses = {'loss_direction': loss_direction.sum() / num_boxes * self.weight_dict['loss_direction']}
        return losses

    def forward(self, outputs, targets, num_classes):
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t['lines']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, num_classes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, num_classes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, num_classes):
        loss_map = {
            'labels': self.loss_labels,
            'lines': self.loss_lines,
            'area': self.loss_area,
            'direction': self.loss_direction,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, num_classes)


@MODELS.register_module()
class HungarianMatcher(nn.Module):

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_labels = outputs['pred_logits'].flatten(0, 1).sigmoid()
            out_lines = outputs['pred_lines'].flatten(0, 1)

            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_lines = torch.cat([v["lines"] for v in targets])

            sizes = [len(v['lines']) for v in targets]

            # Compute the classification cost.
            assert len(tgt_ids) > 0
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_labels ** gamma) * (-(1 - out_labels + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_labels) ** gamma) * (-(out_labels + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # l1 loss 使用展平后的tensor计算
            # Compute the L1 cost between boxes
            cost_line = torch.cdist(out_lines.flatten(1, 2), tgt_lines.flatten(1, 2), p=1)

            # Final cost matrix
            C = cost_line + cost_class
            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


@MODELS.register_module()
class PointCriterion(nn.Module):
    def __init__(self, matcher, weight_dict):
        super().__init__()
        self.matcher = MODELS.build(matcher)
        self.weight_dict = weight_dict

    def loss_points(self, outputs, targets, indices, num_boxes, num_classes):
        idx = _get_src_permutation_idx(indices)
        out_point = outputs['pred_points'][idx]
        target_point = torch.cat([t['key_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_point = F.l1_loss(out_point, target_point, reduction='none')

        losses = {'loss_point': loss_point.sum() / num_boxes * self.weight_dict['loss_point']}
        return losses

    def forward(self, outputs, targets, num_classes):
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t['key_points']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = self.loss_points(outputs, targets, indices, num_boxes, num_classes)

        return losses


@MODELS.register_module()
class PointHungarianMatcher(nn.Module):

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs['pred_points'].shape[:2]

            out_points = outputs['pred_points'].flatten(0, 1)

            tgt_points = torch.cat([v["key_points"] for v in targets])

            sizes = [len(v['key_points']) for v in targets]

            cost_point = torch.cdist(out_points, tgt_points, p=1)

            # Final cost matrix
            C = cost_point
            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def _get_area(line: Tensor):
    count_x = torch.abs(line[:, 1:, 0] - line[:, :-1, 0]) * 0.5 * torch.abs(line[:, 1:, 1] + line[:, :-1, 1])
    count_y = torch.abs(line[:, 1:, 1] - line[:, :-1, 1]) * 0.5 * torch.abs(line[:, 1:, 0] + line[:, :-1, 0])
    return count_x.sum(dim=1), count_y.sum(dim=1)


def xyxy2bbox(xyxy):
    min_x, _ = torch.min(xyxy[:, ::2], dim=-1)
    min_y, _ = torch.min(xyxy[:, 1::2], dim=-1)
    max_x, _ = torch.max(xyxy[:, ::2], dim=-1)
    max_y, _ = torch.max(xyxy[:, 1::2], dim=-1)
    return torch.stack([min_x, min_y, max_x, max_y], dim=-1)

def loss_between_two_curves(pred: Tensor, gt: Tensor):

    # x_temp_x = 0.5 * (torch.abs(pred[:, 1:, 0] - pred[:, :-1, 0]) + torch.abs(gt[:, 1:, 0] - gt[:, :-1, 0]))
    # x_temp_y = torch.abs(0.5 * (pred[:, :-1, 1] + pred[:, 1:, 1]) - 0.5 * (gt[:, :-1, 1] + gt[:, 1:, 1]))

    # y_temp_y = 0.5 * (torch.abs(pred[:, 1:, 1] - pred[:, :-1, 1]) + torch.abs(gt[:, 1:, 1] - gt[:, :-1, 1]))
    # y_temp_x = torch.abs(0.5 * (pred[:, :-1, 0] + pred[:, 1:, 0]) - 0.5 * (gt[:, :-1, 0] + gt[:, 1:, 0]))

    # area = x_temp_x * x_temp_y + y_temp_x * y_temp_y
    # area = area.sum()

    iou = torch.tensor(0., device=pred.device)
    for i in range(pred.size(1) - 1):
        box_pred = torch.cat([pred[:, i, :], pred[:, i + 1, :]], dim=-1)
        box_gt = torch.cat([gt[:, i, :], gt[:, i + 1, :]], dim=-1)
        iou += eiou_loss(xyxy2bbox(box_pred), xyxy2bbox(box_gt))

    return iou

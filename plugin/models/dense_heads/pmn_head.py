import copy

import torch
from mmdet.models import inverse_sigmoid
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmdet3d.registry import TASK_UTILS
from mmengine import MODELS
from mmengine.model import bias_init_with_prob
from torch import nn
from mmcv.cnn import Linear
from plugin.models.utils.utils import (
    _get_clones,
    normalize_2d_pts,
)


@MODELS.register_module()
class PMNHead(nn.Module):
    def __init__(
            self,
            num_classes,
            num_line=50,
            num_raw_point=1000,
            num_reg_fcs=2,
            embed_dims=256,
            transformer=None,
            bbox_coder=None,
            num_points=20,
            coord_dim=2,
            with_box_refine=True,
            use_norm_coord=True,
            line_assigner=None,
            point_assigner=None,
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
            loss_point_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_point_mask=None,
            loss_point_pts=dict(type="PtsL1Loss", loss_weight=5.0),
    ):
        super(PMNHead, self).__init__()
        self.bg_cls_weight = 0
        self.num_classes = num_classes
        self.transformer = MODELS.build(transformer)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.point_assigner = TASK_UTILS.build(point_assigner)
        self.line_assigner = TASK_UTILS.build(line_assigner)
        self.sampler = TASK_UTILS.build(dict(type="PseudoSampler"))
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_pts = MODELS.build(loss_pts)
        self.loss_point_cls = MODELS.build(loss_point_cls)
        self.loss_point_pts = MODELS.build(loss_point_pts)
        self.loss_point_mask = MODELS.build(loss_point_mask)

        self.num_points = num_points
        self.coord_dim = coord_dim
        self.num_line = num_line
        self.num_raw_point = num_raw_point
        self.num_reg_fcs = num_reg_fcs
        self.embed_dims = embed_dims
        self.with_box_refine = with_box_refine
        self.num_enc_layer = self.transformer.encoder.num_layers
        self.num_dec_layer = self.transformer.decoder.num_layers
        self.use_norm_coord = use_norm_coord

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        point_cls_branch = []
        for _ in range(self.num_reg_fcs):
            point_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            point_cls_branch.append(nn.LayerNorm(self.embed_dims))
            point_cls_branch.append(nn.ReLU())
        point_cls_branch.append(Linear(self.embed_dims, self.num_classes))
        point_cls_branch = nn.Sequential(*point_cls_branch)

        point_mask_branch = []
        for _ in range(self.num_reg_fcs):
            point_mask_branch.append(Linear(self.embed_dims, self.embed_dims))
            point_mask_branch.append(nn.LayerNorm(self.embed_dims))
            point_mask_branch.append(nn.ReLU())
        point_mask_branch.append(Linear(self.embed_dims, 9))
        point_mask_branch = nn.Sequential(*point_mask_branch)

        point_reg_branch = []
        for _ in range(self.num_reg_fcs):
            point_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            point_reg_branch.append(nn.ReLU())
        point_reg_branch.append(Linear(self.embed_dims, self.coord_dim))
        point_reg_branch = nn.Sequential(*point_reg_branch)

        self.point_reg_branch = _get_clones(point_reg_branch, self.num_enc_layer)
        self.point_cls_branch = _get_clones(point_cls_branch, self.num_enc_layer)
        self.point_mask_branch = _get_clones(point_mask_branch, self.num_enc_layer)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU())
        cls_branch.append(Linear(self.embed_dims, self.num_classes))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.coord_dim))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branch = nn.ModuleList([cls_branch for _ in range(self.num_dec_layer)])
        self.reg_branch = nn.ModuleList([reg_branch for _ in range(self.num_dec_layer)])

        self.point_embedding = nn.Embedding(self.num_raw_point, self.embed_dims * 2)
        # self.line_embedding = nn.Embedding(self.num_line, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

        bias_init = bias_init_with_prob(0.01)
        for m in self.point_cls_branch:
            nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.cls_branch:
        for m in self.cls_branch:
            nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas):
        point_embedding = self.point_embedding.weight
        # line_embedding = self.line_embedding.weight
        bs = mlvl_feats[0].size(0)
        memory, outs_dec, point_init_ref_points, point_inter_ref_points = self.transformer(
            mlvl_feats, point_embedding, self.point_reg_branch, self.reg_branch, img_metas
        )
        memory = torch.nan_to_num(memory)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_raw_points = []
        outputs_point_masks = []
        outputs_raw_points_score = []
        for lvl in range(memory.shape[0]):
            if lvl == 0:
                reference = point_init_ref_points
            else:
                reference = point_inter_ref_points[lvl - 1]
            point_class = self.point_cls_branch[lvl](memory[lvl]).float()
            point_mask = self.point_mask_branch[lvl](memory[lvl]).float()
            tmp = self.point_reg_branch[lvl](memory[lvl]).float()
            tmp = inverse_sigmoid(reference) + tmp
            outputs_raw_point = tmp.sigmoid()
            outputs_raw_points_score.append(point_class)
            outputs_point_masks.append(point_mask)
            outputs_raw_points.append(outputs_raw_point)

        all_outputs_point_masks = torch.stack(outputs_point_masks)
        all_outputs_raw_points_score = torch.stack(outputs_raw_points_score)
        all_outputs_raw_points = torch.stack(outputs_raw_points)

        outputs_classes = []
        outputs_pts_coords = []
        for lvl in range(outs_dec.shape[0]):
            outputs_class = self.cls_branch[lvl](outs_dec[lvl]
                                                 .reshape(bs, self.num_line, self.num_points, self.embed_dims)
                                                 .mean(2)).float()
            tmp = self.reg_branch[lvl](outs_dec[lvl]).float().sigmoid()
            outputs_pts_coord = tmp.reshape(tmp.shape[0], self.num_line, self.num_points, 2)
            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_pts_coords = torch.stack(outputs_pts_coords)

        outputs = {
            "all_raw_points": all_outputs_raw_points,
            "all_raw_points_score": all_outputs_raw_points_score,
            "all_point_masks": all_outputs_point_masks,
            "all_cls_scores": outputs_classes,
            "all_line_preds": outputs_pts_coords,
            # "init_ref_points": point_init_ref_points,
        }

        return outputs

    @torch.no_grad()
    def get_targets_single(
            self, cls_score, pts_pred, gt_labels, gt_shifts_pts, gt_bboxes_ignore=None
    ):
        num_lines = pts_pred.size(0)
        # assigner and sampler
        assign_result, order_index = self.line_assigner.assign(
            cls_score, pts_pred, gt_labels, gt_shifts_pts, gt_bboxes_ignore
        )

        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        gt_flags = pts_pred.new_zeros(pts_pred.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=pts_pred,
            gt_bboxes=gt_shifts_pts,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False,
        )

        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_shifts_pts.new_full(
            (num_lines,), self.num_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_shifts_pts.new_ones(num_lines)

        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[
                sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds
            ]
        pts_targets = pts_pred.new_zeros(
            (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2))
        )
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        pts_targets[pos_inds] = gt_shifts_pts[
                                sampling_result.pos_assigned_gt_inds, assigned_shift, :, :
                                ]
        # divider_indices = torch.nonzero(labels == 0)
        select_indices = torch.nonzero((labels == 1) | (labels == 0))
        pts_weights[select_indices] = pts_weights[select_indices] * 2
        label_weights[select_indices] = label_weights[select_indices] * 2
        return (labels, label_weights, pts_targets, pts_weights, pos_inds, neg_inds)

    @torch.no_grad()
    def get_point_targets_single(
            self, pts_pred, points_score, point_masks, gt_raw_points, gt_points_label, gt_point_masks,
            gt_bboxes_ignore=None
    ):
        num_point = pts_pred.size(0)
        # start_time = time.perf_counter()
        assign_result = self.point_assigner.assign(
            pts_pred, points_score, gt_raw_points, gt_points_label, gt_bboxes_ignore
        )
        # end_time = time.perf_counter()
        # print(f"point assign : {end_time - start_time}")

        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        labels = pts_pred.new_full((num_point,),
                                   self.num_classes,
                                   dtype=torch.long)
        labels[pos_inds] = gt_points_label[assign_result.gt_inds[pos_inds] - 1]
        label_weights = pts_pred.new_ones(num_point)

        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        pts_targets[pos_inds] = gt_raw_points[assign_result.gt_inds[pos_inds] - 1, :]

        mask_targets = point_masks.new_zeros((point_masks.size(0), point_masks.size(1)), dtype=torch.long)
        mask_targets[pos_inds] = gt_point_masks[assign_result.gt_inds[pos_inds] - 1, :]
        mask_weights = pts_pred.new_ones(num_point)

        # divider_indices = torch.nonzero(labels == 0)
        select_indices = torch.nonzero((labels == 1) | (labels == 0))
        pts_weights[select_indices] = pts_weights[select_indices] * 2
        label_weights[select_indices] = label_weights[select_indices] * 2
        return (labels, label_weights, pts_targets, pts_weights, mask_targets, mask_weights, pos_inds, neg_inds)

    def get_targets(
            self,
            cls_scores_list,
            pts_preds_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list=None,
    ):
        assert (
                gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self.get_targets_single,
            cls_scores_list,
            pts_preds_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_line(
            self,
            cls_scores,
            pts_preds,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            pts_preds_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = pts_targets
        if self.use_norm_coord:
            normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_pts_targets.flatten(1)).all(dim=-1)
        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))

        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=cls_avg_factor,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_pts = torch.nan_to_num(loss_pts)
        return loss_cls, loss_pts

    def loss_raw_points(
            self,
            pts_preds,
            all_raw_points_score,
            all_point_masks,
            gt_raw_points,
            gt_raw_points_label_list,
            gt_point_masks,
            gt_bboxes_ignore_list=None,
    ):
        num_imgs = pts_preds.size(0)
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            mask_targets_list,
            mask_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self.get_point_targets_single,
            pts_preds_list,
            all_raw_points_score,
            all_point_masks,
            gt_raw_points,
            gt_raw_points_label_list,
            gt_point_masks,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.cat(mask_weights_list, 0)

        # classification loss
        cls_scores = all_raw_points_score.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight)
        cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_point_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # mask loss
        point_masks = all_point_masks.reshape(-1, 9)
        loss_mask = self.loss_point_mask(
            point_masks, mask_targets, mask_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = pts_targets
        if self.use_norm_coord:
            normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_pts_targets.flatten(1)).all(dim=-1)

        pts_preds = pts_preds.flatten(0, 1)
        loss_pts = self.loss_point_pts(
            pts_preds[isnotnan, :],
            normalized_pts_targets[isnotnan, :],
            pts_weights[isnotnan, :],
            avg_factor=cls_avg_factor,
        )

        loss_pts = torch.nan_to_num(loss_pts)
        return loss_pts, loss_cls, loss_mask

    def loss(
            self,
            preds_dicts,
            gt_labels,
            gt_lines,
            gt_raw_points_list,
            gt_raw_points_label_list,
            gt_point_masks_list,
            gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )
        all_raw_points = preds_dicts["all_raw_points"]
        all_raw_points_score = preds_dicts["all_raw_points_score"]
        all_point_masks = preds_dicts["all_point_masks"]
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_line_preds = preds_dicts["all_line_preds"]

        gt_shifts_pts_list = copy.deepcopy(gt_lines)
        gt_raw_points_list = copy.deepcopy(gt_raw_points_list)

        num_enc_layers = len(all_raw_points)
        num_dec_layers = len(all_cls_scores)

        gt_raw_points_list = [gt_raw_points_list for _ in range(num_enc_layers)]
        gt_raw_points_label_list = [gt_raw_points_label_list for _ in range(num_enc_layers)]
        gt_point_masks_list = [gt_point_masks_list for _ in range(num_enc_layers)]
        all_gt_labels_list = [gt_labels for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_enc_layers)
        ]

        losses_raw_points, losses_points_cls, losses_point_mask = multi_apply(
            self.loss_raw_points,
            all_raw_points,
            all_raw_points_score,
            all_point_masks,
            gt_raw_points_list,
            gt_raw_points_label_list,
            gt_point_masks_list,
            all_gt_bboxes_ignore_list
        )
        # end_time = time.perf_counter()
        # print(f"point loss : {end_time - start_time}")

        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # start_time = time.perf_counter()
        losses_cls, losses_pts = multi_apply(
            self.loss_line,
            all_cls_scores,
            all_line_preds,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )

        # end_time = time.perf_counter()
        # print(f"point loss : {end_time - start_time}")

        loss_dict = dict()
        loss_dict["loss_pts"] = losses_pts[-1]
        loss_dict["loss_cls"] = losses_cls[-1]

        loss_dict["loss_point"] = losses_raw_points[-1]
        loss_dict["loss_point_cls"] = losses_points_cls[-1]
        loss_dict["loss_point_mask"] = losses_point_mask[-1]
        num_enc_layer = 0
        for loss_pts_i, loss_cls_i, loss_mask_i in zip(losses_raw_points[:-1], losses_points_cls[:-1],
                                                       losses_point_mask[:-1]):
            loss_dict[f"d{num_enc_layer}.loss_raw_point"] = loss_pts_i
            loss_dict[f"d{num_enc_layer}.loss_point_cls"] = loss_cls_i
            loss_dict[f"d{num_enc_layer}.loss_point_mask"] = loss_mask_i
            num_enc_layer += 1
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict

    def get_preds(self, ori_preds_dicts):
        preds_dicts = self.bbox_coder.decode(ori_preds_dicts)
        # level_focus_points = ori_preds_dicts['all_raw_points'].transpose(1, 0)
        # for pred_dict, points in zip(preds_dicts, level_focus_points):
        #     pred_dict['level_focus_points'] = points
        # key_points = preds_dicts["all_raw_points"][-1]
        # points_score = preds_dicts["all_raw_points_score"][-1].sigmoid()
        # key_points[..., 0] = key_points[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        # key_points[..., 1] = key_points[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        return preds_dicts


@MODELS.register_module()
class PMNV1Head(nn.Module):
    def __init__(
            self,
            num_classes,
            num_line=50,
            num_raw_point=1000,
            num_reg_fcs=2,
            embed_dims=256,
            transformer=None,
            bbox_coder=None,
            num_points=20,
            coord_dim=2,
            with_box_refine=True,
            use_norm_coord=True,
            line_assigner=None,
            point_assigner=None,
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
            loss_point_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_point_mask=None,
            loss_point_pts=dict(type="PtsL1Loss", loss_weight=5.0),
    ):
        super(PMNV1Head, self).__init__()
        self.bg_cls_weight = 0
        self.num_classes = num_classes
        self.transformer = MODELS.build(transformer)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.point_assigner = TASK_UTILS.build(point_assigner)
        self.line_assigner = TASK_UTILS.build(line_assigner)
        self.sampler = TASK_UTILS.build(dict(type="PseudoSampler"))
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_pts = MODELS.build(loss_pts)
        self.loss_point_cls = MODELS.build(loss_point_cls)
        self.loss_point_pts = MODELS.build(loss_point_pts)
        self.loss_point_mask = MODELS.build(loss_point_mask)

        self.num_points = num_points
        self.coord_dim = coord_dim
        self.num_line = num_line
        self.num_raw_point = num_raw_point
        self.num_reg_fcs = num_reg_fcs
        self.embed_dims = embed_dims
        self.with_box_refine = with_box_refine
        self.num_enc_layer = self.transformer.encoder.num_layers
        self.num_dec_layer = self.transformer.decoder.num_layers
        self.use_norm_coord = use_norm_coord

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        point_cls_branch = []
        for _ in range(self.num_reg_fcs):
            point_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            point_cls_branch.append(nn.LayerNorm(self.embed_dims))
            point_cls_branch.append(nn.ReLU())
        point_cls_branch.append(Linear(self.embed_dims, self.num_classes))
        point_cls_branch = nn.Sequential(*point_cls_branch)

        point_mask_branch = []
        for _ in range(self.num_reg_fcs):
            point_mask_branch.append(Linear(self.embed_dims, self.embed_dims))
            point_mask_branch.append(nn.LayerNorm(self.embed_dims))
            point_mask_branch.append(nn.ReLU())
        point_mask_branch.append(Linear(self.embed_dims, 9))
        point_mask_branch = nn.Sequential(*point_mask_branch)

        point_reg_branch = []
        for _ in range(self.num_reg_fcs):
            point_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            point_reg_branch.append(nn.ReLU())
        point_reg_branch.append(Linear(self.embed_dims, self.coord_dim))
        point_reg_branch = nn.Sequential(*point_reg_branch)

        self.point_reg_branch = _get_clones(point_reg_branch, self.num_enc_layer)
        self.point_cls_branch = _get_clones(point_cls_branch, self.num_enc_layer)
        self.point_mask_branch = _get_clones(point_mask_branch, self.num_enc_layer)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU())
        cls_branch.append(Linear(self.embed_dims, self.num_classes))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.coord_dim))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branch = nn.ModuleList([cls_branch for _ in range(self.num_dec_layer)])
        self.reg_branch = nn.ModuleList([reg_branch for _ in range(self.num_dec_layer)])

        self.point_embedding = nn.Embedding(self.num_raw_point, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

        bias_init = bias_init_with_prob(0.01)
        for m in self.point_cls_branch:
            nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.cls_branch:
        for m in self.cls_branch:
            nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas):
        point_embedding = self.point_embedding.weight
        # line_embedding = self.line_embedding.weight
        bs = mlvl_feats[0].size(0)
        memory, outs_dec, point_init_ref_points, point_inter_ref_points = self.transformer(
            mlvl_feats, point_embedding, self.point_reg_branch, self.point_cls_branch, self.reg_branch, img_metas
        )
        memory = torch.nan_to_num(memory)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_raw_points = []
        outputs_point_masks = []
        outputs_raw_points_score = []
        for lvl in range(memory.shape[0]):
            # if lvl == 0:
            #     reference = point_init_ref_points
            # else:
            #     reference = point_inter_ref_points[lvl - 1]
            point_class = self.point_cls_branch[lvl](memory[lvl]).float()
            point_mask = self.point_mask_branch[lvl](memory[lvl]).float()
            # tmp = self.point_reg_branch[lvl](memory[lvl]).float()
            # tmp = inverse_sigmoid(reference) + tmp
            # outputs_raw_point = tmp.sigmoid()
            outputs_raw_points_score.append(point_class)
            outputs_point_masks.append(point_mask)
            outputs_raw_points.append(point_inter_ref_points[lvl])

        all_outputs_point_masks = torch.stack(outputs_point_masks)
        all_outputs_raw_points_score = torch.stack(outputs_raw_points_score)
        all_outputs_raw_points = torch.stack(outputs_raw_points)

        outputs_classes = []
        outputs_pts_coords = []

        for lvl in range(outs_dec.shape[0]):
            outputs_class = self.cls_branch[lvl](outs_dec[lvl]
                                                 .reshape(bs, self.num_line, self.num_points, self.embed_dims)
                                                 .mean(2)).float()
            tmp = self.reg_branch[lvl](outs_dec[lvl]).float().sigmoid()
            outputs_pts_coord = tmp.reshape(tmp.shape[0], self.num_line, self.num_points, self.coord_dim)
            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_pts_coords = torch.stack(outputs_pts_coords)

        outputs = {
            "all_raw_points": all_outputs_raw_points,
            "all_raw_points_score": all_outputs_raw_points_score,
            "all_point_masks": all_outputs_point_masks,
            "all_cls_scores": outputs_classes,
            "all_line_preds": outputs_pts_coords
        }

        return outputs

    @torch.no_grad()
    def get_targets_single(
            self, cls_score, pts_pred, gt_labels, gt_shifts_pts, gt_bboxes_ignore=None
    ):
        num_lines = pts_pred.size(0)
        # assigner and sampler
        assign_result, order_index = self.line_assigner.assign(
            cls_score, pts_pred, gt_labels, gt_shifts_pts, gt_bboxes_ignore
        )

        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        gt_flags = pts_pred.new_zeros(pts_pred.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=pts_pred,
            gt_bboxes=gt_shifts_pts,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False,
        )

        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_shifts_pts.new_full(
            (num_lines,), self.num_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_shifts_pts.new_ones(num_lines)

        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[
                sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds
            ]
        pts_targets = pts_pred.new_zeros(
            (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2))
        )
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        pts_targets[pos_inds] = gt_shifts_pts[
                                sampling_result.pos_assigned_gt_inds, assigned_shift, :, :
                                ]
        # divider_indices = torch.nonzero(labels == 0)
        ped_crossing_indices = torch.nonzero(labels == 1)
        pts_weights[ped_crossing_indices] = pts_weights[ped_crossing_indices] * 3
        label_weights[ped_crossing_indices] = label_weights[ped_crossing_indices] * 3
        return (labels, label_weights, pts_targets, pts_weights, pos_inds, neg_inds)

    @torch.no_grad()
    def get_point_targets_single(
            self, pts_pred, points_score, point_masks, gt_raw_points, gt_points_label, gt_point_masks,
            gt_bboxes_ignore=None
    ):
        num_point = pts_pred.size(0)
        # start_time = time.perf_counter()
        assign_result = self.point_assigner.assign(
            pts_pred, points_score, gt_raw_points, gt_points_label, gt_bboxes_ignore
        )
        # end_time = time.perf_counter()
        # print(f"point assign : {end_time - start_time}")

        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        labels = pts_pred.new_full((num_point,),
                                   self.num_classes,
                                   dtype=torch.long)
        labels[pos_inds] = gt_points_label[assign_result.gt_inds[pos_inds] - 1]
        label_weights = pts_pred.new_ones(num_point)

        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        pts_targets[pos_inds] = gt_raw_points[assign_result.gt_inds[pos_inds] - 1, :]

        mask_targets = point_masks.new_zeros((point_masks.size(0), point_masks.size(1)), dtype=torch.long)
        mask_targets[pos_inds] = gt_point_masks[assign_result.gt_inds[pos_inds] - 1, :]
        mask_weights = pts_pred.new_ones(num_point)

        # divider_indices = torch.nonzero(labels == 0)
        ped_crossing_indices = torch.nonzero(labels == 1)
        pts_weights[ped_crossing_indices] = pts_weights[ped_crossing_indices] * 3
        label_weights[ped_crossing_indices] = label_weights[ped_crossing_indices] * 3
        return (labels, label_weights, pts_targets, pts_weights, mask_targets, mask_weights, pos_inds, neg_inds)

    def get_targets(
            self,
            cls_scores_list,
            pts_preds_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list=None,
    ):
        assert (
                gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self.get_targets_single,
            cls_scores_list,
            pts_preds_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_line(
            self,
            cls_scores,
            pts_preds,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            pts_preds_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = pts_targets
        if self.use_norm_coord:
            normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_pts_targets.flatten(1)).all(dim=-1)
        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))

        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_pts = torch.nan_to_num(loss_pts)
        return loss_cls, loss_pts

    def loss_raw_points(
            self,
            pts_preds,
            all_raw_points_score,
            all_point_masks,
            gt_raw_points,
            gt_raw_points_label_list,
            gt_point_masks,
            gt_bboxes_ignore_list=None,
    ):
        num_imgs = pts_preds.size(0)
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            mask_targets_list,
            mask_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self.get_point_targets_single,
            pts_preds_list,
            all_raw_points_score,
            all_point_masks,
            gt_raw_points,
            gt_raw_points_label_list,
            gt_point_masks,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.cat(mask_weights_list, 0)

        # classification loss
        cls_scores = all_raw_points_score.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        cls_avg_factor = reduce_mean(
            cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_point_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # mask loss
        point_masks = all_point_masks.reshape(-1, 9)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        cls_avg_factor = reduce_mean(
            all_point_masks.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_mask = self.loss_point_mask(
            point_masks, mask_targets, mask_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = pts_preds.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = pts_targets
        if self.use_norm_coord:
            normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_pts_targets.flatten(1)).all(dim=-1)

        pts_preds = pts_preds.flatten(0, 1)
        loss_pts = self.loss_point_pts(
            pts_preds[isnotnan, :],
            normalized_pts_targets[isnotnan, :],
            pts_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        loss_pts = torch.nan_to_num(loss_pts)
        return loss_pts, loss_cls, loss_mask

    def loss(
            self,
            preds_dicts,
            gt_labels,
            gt_lines,
            gt_raw_points_list,
            gt_raw_points_label_list,
            gt_point_masks_list,
            gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )
        all_raw_points = preds_dicts["all_raw_points"]
        all_raw_points_score = preds_dicts["all_raw_points_score"]
        all_point_masks = preds_dicts["all_point_masks"]
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_line_preds = preds_dicts["all_line_preds"]

        gt_shifts_pts_list = copy.deepcopy(gt_lines)
        gt_raw_points_list = copy.deepcopy(gt_raw_points_list)

        num_enc_layers = len(all_raw_points)
        num_dec_layers = len(all_cls_scores)

        gt_raw_points_list = [gt_raw_points_list for _ in range(num_enc_layers)]
        gt_raw_points_label_list = [gt_raw_points_label_list for _ in range(num_enc_layers)]
        gt_point_masks_list = [gt_point_masks_list for _ in range(num_enc_layers)]
        all_gt_labels_list = [gt_labels for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_enc_layers)
        ]

        losses_raw_points, losses_points_cls, losses_point_mask = multi_apply(
            self.loss_raw_points,
            all_raw_points,
            all_raw_points_score,
            all_point_masks,
            gt_raw_points_list,
            gt_raw_points_label_list,
            gt_point_masks_list,
            all_gt_bboxes_ignore_list
        )

        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_pts = multi_apply(
            self.loss_line,
            all_cls_scores,
            all_line_preds,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        loss_dict["loss_pts"] = losses_pts[-1]
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_point"] = losses_raw_points[-1]
        loss_dict["loss_point_cls"] = losses_points_cls[-1]
        loss_dict["loss_point_mask"] = losses_point_mask[-1]
        num_enc_layer = 0
        for loss_pts_i, loss_cls_i, loss_mask_i in zip(losses_raw_points[:-1], losses_points_cls[:-1],
                                                       losses_point_mask[:-1]):
            loss_dict[f"d{num_enc_layer}.loss_raw_point"] = loss_pts_i
            loss_dict[f"d{num_enc_layer}.loss_point_cls"] = loss_cls_i
            loss_dict[f"d{num_enc_layer}.loss_point_mask"] = loss_cls_i
            num_enc_layer += 1
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i in zip(losses_cls[:-1], losses_pts[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            num_dec_layer += 1
        return loss_dict

    def get_preds(self, preds_dicts):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        # key_points = preds_dicts["all_raw_points"][-1]
        # points_score = preds_dicts["all_raw_points_score"][-1].sigmoid()
        # key_points[..., 0] = key_points[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        # key_points[..., 1] = key_points[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        # new_preds_dicts[""]
        return preds_dicts

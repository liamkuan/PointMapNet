import copy

import torch
import torch.nn.functional as F
from mmdet.models import inverse_sigmoid
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import multi_apply
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
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


def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] -
                                          pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] -
                                          pc_range[1]) + pc_range[1])
    new_pts[..., 2:3] = (pts[..., 2:3] * (pc_range[5] -
                                          pc_range[2]) + pc_range[2])
    return new_pts


def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    patch_z = pc_range[5] - pc_range[2]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    new_pts[..., 2:3] = pts[..., 2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h, patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2] * (pc_range[3] -
                                              pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2] * (pc_range[4] -
                                              pc_range[1]) + pc_range[1])

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] -
                                          pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] -
                                          pc_range[1]) + pc_range[1])
    return new_pts


@MODELS.register_module()
class DBEVNetHead(nn.Module):
    def __init__(
            self,
            num_classes,
            num_raw_point=1000,
            num_reg_fcs=2,
            embed_dims=256,
            transformer=None,
            bbox_coder=None,
            num_vec_one2one=50,
            num_vec_one2many=0,
            k_one2many=0,
            lambda_one2many=1,
            num_pts_per_vec=2,
            num_pts_per_gt_vec=2,
            coord_dim=2,
            with_box_refine=True,
            use_norm_coord=True,
            assigner=None,
            loss_cls=None,
            loss_pts=None,
            loss_dir=None,
            loss_bbox=None,
            loss_iou=None,
            loss_point_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_point_mask=None,
            loss_point_pts=dict(type="PtsL1Loss", loss_weight=5.0),
    ):
        super(DBEVNetHead, self).__init__()
        self.bg_cls_weight = 0
        self.num_classes = num_classes
        self.transformer = MODELS.build(transformer)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.assigner = TASK_UTILS.build(assigner)
        self.sampler = TASK_UTILS.build(dict(type="PseudoSampler"))
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_pts = MODELS.build(loss_pts)
        self.loss_dir = MODELS.build(loss_dir)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)
        self.loss_point_cls = MODELS.build(loss_point_cls)
        self.loss_point_pts = MODELS.build(loss_point_pts)
        self.loss_point_mask = MODELS.build(loss_point_mask)

        self.coord_dim = coord_dim
        self.code_size = self.coord_dim * num_pts_per_vec
        self.num_raw_point = num_raw_point
        self.num_reg_fcs = num_reg_fcs
        self.embed_dims = embed_dims
        self.with_box_refine = with_box_refine
        self.num_enc_layer = self.transformer.encoder.num_layers
        self.num_dec_layer = self.transformer.decoder.num_layers
        self.use_norm_coord = use_norm_coord

        num_vec = num_vec_one2one + num_vec_one2many
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

        self.code_weights = [1.0, 1.0, 1.0, 1.0]
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
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
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branch = nn.ModuleList([cls_branch for _ in range(self.num_dec_layer)])
        self.reg_branch = nn.ModuleList([reg_branch for _ in range(self.num_dec_layer)])

        self.point_embedding = nn.Embedding(self.num_raw_point, self.embed_dims)
        self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
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
        if self.training:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one

        point_embedding = self.point_embedding.weight
        object_query_embeds = self.instance_embedding.weight
        bs = mlvl_feats[0].size(0)

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros([num_vec, num_vec, ]).bool().to(mlvl_feats[0].device)
        )
        self_attn_mask[self.num_vec_one2one:, 0: self.num_vec_one2one, ] = True
        self_attn_mask[0: self.num_vec_one2one, self.num_vec_one2one:, ] = True

        # memory, outs_dec, point_init_ref_points, point_inter_ref_points = self.transformer(
        #     mlvl_feats, point_embedding, self.point_reg_branch, self.point_cls_branch, self.reg_branch, img_metas
        # )
        memory, outs_dec, point_init_ref_points, point_inter_ref_points, init_reference, inter_references = self.transformer(
            mlvl_feats, point_embedding, object_query_embeds, self.point_reg_branch, self.point_cls_branch, self.reg_branch, img_metas,
            self_attn_mask=self_attn_mask,
        )
        memory = torch.nan_to_num(memory)
        outs_dec = torch.nan_to_num(outs_dec)
        # outs_dec = outs_dec.permute(0, 2, 1, 3)

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

        outs_dec = outs_dec.permute(0, 2, 1, 3)
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []
        # for lvl in range(outs_dec.shape[0]):
        #     outputs_class = self.cls_branch[lvl](outs_dec[lvl]
        #                                          .reshape(bs, self.num_line, self.num_points, self.embed_dims)
        #                                          .mean(2)).float()
        #     tmp = self.reg_branch[lvl](outs_dec[lvl]).float().sigmoid()
        #     outputs_pts_coord = tmp.reshape(tmp.shape[0], self.num_line, self.num_points, 2)
        #     outputs_classes.append(outputs_class)
        #     outputs_pts_coords.append(outputs_pts_coord)

        for lvl in range(outs_dec.shape[0]):

            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branch[lvl](outs_dec[lvl])
            tmp = self.reg_branch[lvl](outs_dec[lvl])
            tmp[..., :] += reference[..., :]
            tmp = tmp.sigmoid()  # cx,cy,w,h
            outputs_coord, outputs_pts_coord = self.transform_box(tmp, num_vec=num_vec)

            outputs_classes_one2one.append(outputs_class[:, 0:self.num_vec_one2one].float())
            outputs_coords_one2one.append(outputs_coord[:, 0:self.num_vec_one2one].float())
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, 0:self.num_vec_one2one].float())

            outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:].float())
            outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:].float())
            outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:].float())

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        outputs = {
            "all_raw_points": all_outputs_raw_points,
            "all_raw_points_score": all_outputs_raw_points_score,
            "all_point_masks": all_outputs_point_masks,
            'all_cls_scores': outputs_classes_one2one,
            'all_bbox_preds': outputs_coords_one2one,
            'all_pts_preds': outputs_pts_coords_one2one,
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_cls_scores=None,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                seg=None,
                pv_seg=None,
            ),
            # "image_depths": image_depths
        }

        return outputs

    def transform_box(self, pts, num_vec=50, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """

        pts_reshape = pts.view(pts.shape[0], num_vec,
                               self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        xmin = pts_x.min(dim=2, keepdim=True)[0]
        xmax = pts_x.max(dim=2, keepdim=True)[0]
        ymin = pts_y.min(dim=2, keepdim=True)[0]
        ymax = pts_y.max(dim=2, keepdim=True)[0]
        bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
        bbox = bbox_xyxy_to_cxcywh(bbox)
        return bbox, pts_reshape

    @torch.no_grad()
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                                          gt_bboxes, gt_labels, gt_shifts_pts,
                                                          gt_bboxes_ignore)

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        gt_flags = pts_pred.new_zeros(pts_pred.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=pts_pred,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                                          pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    @torch.no_grad()
    def get_point_targets_single(
            self, pts_pred, points_score, point_masks, gt_raw_points, gt_points_label, gt_point_masks, img_meta,
            gt_bboxes_ignore=None
    ):
        num_point = pts_pred.size(0)
        # start_time = time.perf_counter()
        assign_result = img_meta['assign_result']
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
        # ped_crossing_indices = torch.nonzero(labels == 1)
        # dri_indices = torch.nonzero(labels == 2)
        # w1, w2, w3 = 0, 0, 0
        # if len(divider_indices) != 0:
        #     w1 = (num_point / len(divider_indices))
        # if len(ped_crossing_indices) != 0:
        #     w2 = (num_point / len(ped_crossing_indices))
        # if len(dri_indices) != 0:
        #     w3 = (num_point / len(dri_indices))
        # W = w1 + w2 + w3
        # if len(divider_indices) != 0:
        #     pts_weights[divider_indices] = pts_weights[divider_indices] * (w1 / W)
        #     label_weights[divider_indices] = label_weights[divider_indices] * (w1 / W)
        # if len(ped_crossing_indices) != 0:
        #     pts_weights[ped_crossing_indices] = pts_weights[ped_crossing_indices] * (w2 / W)
        #     label_weights[ped_crossing_indices] = label_weights[ped_crossing_indices] * (w2 / W)
        # if len(dri_indices) != 0:
        #     pts_weights[dri_indices] = pts_weights[dri_indices] * (w3 / W)
        #     label_weights[dri_indices] = label_weights[dri_indices] * (w3 / W)
        # return (labels, label_weights, pts_targets, pts_weights, mask_targets, mask_weights, pos_inds, neg_inds)

        select_indices = torch.nonzero((labels == 1) | (labels == 0))
        pts_weights[select_indices] = pts_weights[select_indices] * 2
        label_weights[select_indices] = label_weights[select_indices] * 2
        return (labels, label_weights, pts_targets, pts_weights, mask_targets, mask_weights, pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_line(
            self,
            cls_scores,
            bbox_preds,
            pts_preds,
            gt_bboxes_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        cls_avg_factor = reduce_mean(
            cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                      :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                      align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :], normalized_pts_targets[isnotnan,
                                       :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-1, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, 1:, :] - denormed_pts_preds[:,
                                                                                :-1, :]
        pts_targets_dir = pts_targets[:, 1:, :] - pts_targets[:, :-1, :]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :], pts_targets_dir[isnotnan,
                                                    :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_iou = torch.nan_to_num(loss_iou)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    def compute_point_loss_with_match(
            self,
            pts_preds,
            all_raw_points_score,
            all_point_masks,
            labels,
            label_weights,
            mask_targets,
            mask_weights,
            pts_targets,
            pts_weights,
            cls_avg_factor
    ):
        # classification loss
        cls_scores = all_raw_points_score.reshape(-1, self.num_classes)

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_point_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # mask loss
        point_masks = all_point_masks.reshape(-1, 9)
        loss_mask = self.loss_point_mask(
            point_masks, mask_targets, mask_weights, avg_factor=cls_avg_factor)

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

    def loss_raw_points(
            self,
            pts_preds,
            all_raw_points_score,
            all_point_masks,
            gt_raw_points,
            gt_raw_points_label_list,
            gt_point_masks,
            img_metas,
    ):
        num_imgs = pts_preds[-1].size(0)
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
            pts_preds[0],
            all_raw_points_score[0],
            all_point_masks[0],
            gt_raw_points,
            gt_raw_points_label_list,
            gt_point_masks,
            img_metas,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.cat(mask_weights_list, 0)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = reduce_mean(pts_targets.new_tensor([cls_avg_factor]))

        (
            loss_pts,
            loss_cls,
            loss_mask
        ) = multi_apply(
            self.compute_point_loss_with_match,
            pts_preds,
            all_raw_points_score,
            all_point_masks,
            [labels for _ in range(len(pts_preds))],
            [label_weights for _ in range(len(pts_preds))],
            [mask_targets for _ in range(len(pts_preds))],
            [mask_weights for _ in range(len(pts_preds))],
            [pts_targets for _ in range(len(pts_preds))],
            [pts_weights for _ in range(len(pts_preds))],
            [cls_avg_factor for _ in range(len(pts_preds))]
        )

        return loss_pts, loss_cls, loss_mask

    def loss(
            self,
            preds_dicts,
            gt_bboxes_list,
            gt_labels_list,
            gt_raw_points_list,
            gt_raw_points_label_list,
            gt_point_masks_list,
            img_metas,
            # gt_image_depths,
            gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds = preds_dicts['all_pts_preds']
        # all_image_depths = preds_dicts["image_depths"]
        loss_dict = dict()
        if gt_raw_points_list is not None:
            all_raw_points = preds_dicts["all_raw_points"]
            all_raw_points_score = preds_dicts["all_raw_points_score"]
            all_point_masks = preds_dicts["all_point_masks"]
            gt_raw_points_list = copy.deepcopy(gt_raw_points_list)
            num_enc_layers = len(all_raw_points)
            losses_raw_points, losses_points_cls, losses_point_mask = self.loss_raw_points(
                all_raw_points,
                all_raw_points_score,
                all_point_masks,
                gt_raw_points_list,
                gt_raw_points_label_list,
                gt_point_masks_list,
                img_metas,
            )
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

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        gt_shifts_pts_list = [
            gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 计算深度值loss
        # level_depth_loss = []
        # mae_loss = nn.L1Loss()
        # for i in range(2):
        #     level_image_depths = torch.cat([item[i] for item in gt_image_depths], dim=0)
        #     level_image_depths = torch.clamp(level_image_depths, min=1e-5, max=30 - 1e-5)
        #     level_image_depths = level_image_depths.permute(0, 3, 1, 2) / self.pc_range[4]
        #     level_depth_loss.append(mae_loss(all_image_depths[i], level_image_depths))
        # loss_dict["loss_depth"] = (level_depth_loss[0] + level_depth_loss[1]) * 20

        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_line, all_cls_scores, all_bbox_preds, all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1],
                                                                               losses_bbox[:-1],
                                                                               losses_iou[:-1],
                                                                               losses_pts[:-1],
                                                                               losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        return loss_dict

    def get_preds(self, preds_dicts):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        # key_points = preds_dicts["all_raw_points"][-1]
        # points_score = preds_dicts["all_raw_points_score"][-1].sigmoid()
        # key_points[..., 0] = key_points[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        # key_points[..., 1] = key_points[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        return preds_dicts

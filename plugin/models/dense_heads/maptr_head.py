import copy

import math
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.models import inverse_sigmoid
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.model import bias_init_with_prob
from mmengine.registry import MODELS, TASK_UTILS

from plugin.models.utils.utils import _get_clones, normalize_2d_pts, denormalize_2d_pts


@MODELS.register_module()
class MapTRHead(nn.Module):

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 assigner=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_vec=20,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_pts=dict(type='ChamferDistance',
                               loss_src_weight=1.0,
                               loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 **kwargs):
        super(MapTRHead, self).__init__()
        self.bg_cls_weight = 0
        self.num_reg_fcs = 2
        self.embed_dims = 256
        self.cls_out_channels = 3
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        self.assigner = TASK_UTILS.build(assigner)
        self.transformer = MODELS.build(transformer)
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.positional_encoding = MODELS.build(positional_encoding)
        self.loss_pts = MODELS.build(loss_pts)
        self.loss_dir = MODELS.build(loss_dir)
        self.loss_cls = MODELS.build(loss_cls)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.bev_encoder_type == 'BEVFormerEncoder':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas):
        mlvl_feats = [x.float() for x in mlvl_feats]
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # import pdb;pdb.set_trace()
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                   device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        outputs = self.transformer(
            mlvl_feats,
            None,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl]
                                                   .view(bs, self.num_vec, self.num_pts_per_vec, -1)
                                                   .mean(2)).float()
            tmp = self.reg_branches[lvl](hs[lvl]).float()

            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp = tmp.sigmoid()  # cx,cy,w,h
            # import pdb;pdb.set_trace()
            # tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
            #                  self.pc_range[0]) + self.pc_range[0])
            # tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
            #                  self.pc_range[1]) + self.pc_range[1])
            # tmp = tmp.reshape(bs, self.num_vec,-1)
            # TODO: check if using sigmoid
            outputs_pts_coord = tmp.view(tmp.shape[0], self.num_vec, self.num_pts_per_vec, 2)
            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_line_preds': outputs_pts_coords,
        }

        return outs

    def get_targets_single(self,
                           cls_score,
                           pts_pred,
                           gt_labels,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        num_lines = pts_pred.size(0)
        # assigner and sampler
        assign_result, order_index = self.assigner.assign(cls_score, pts_pred,
                                                          gt_labels, gt_shifts_pts,
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
            gt_bboxes=gt_shifts_pts,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)

        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_shifts_pts.new_full((num_lines,),
                                        self.cls_out_channels,
                                        dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_shifts_pts.new_ones(num_lines)

        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                                          pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    pts_preds_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self.get_targets_single, cls_scores_list, pts_preds_list,
            gt_labels_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    pts_preds,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, pts_preds_list,
                                           gt_labels_list, gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
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

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_pts_targets.flatten(1)).all(dim=-1)
        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))

        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :], normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval:, :] - denormed_pts_preds[:,
                                                                                :-self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval:, :] - pts_targets[:, :-self.dir_interval, :]
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :], pts_targets_dir[isnotnan,
                                                    :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_pts, loss_dir

    def loss(self,
             preds_dicts,
             gt_labels_list,
             gt_lines_list,
             gt_bboxes_ignore=None, ):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_shifts_pts_list = copy.deepcopy(gt_lines_list)
        all_cls_scores = preds_dicts['all_cls_scores']
        all_line_preds = preds_dicts['all_line_preds']

        num_dec_layers = len(all_cls_scores)

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_line_preds, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1], losses_pts[:-1], losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        return loss_dict

    def get_preds(self, preds_dicts):
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        return preds_dicts


@MODELS.register_module()
class MapTRHeadV1(nn.Module):

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 assigner=None,
                 num_cls_fcs=2,
                 bev_h=30,
                 bev_w=30,
                 num_vec=20,
                 num_pts_per_vec=2,
                 dir_interval=1,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_pts=dict(type='ChamferDistance',
                               loss_src_weight=1.0,
                               loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 **kwargs):
        super(MapTRHeadV1, self).__init__()
        self.bg_cls_weight = 0
        self.num_reg_fcs = 2
        self.embed_dims = 256
        self.cls_out_channels = 3
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        self.assigner = TASK_UTILS.build(assigner)
        self.transformer = MODELS.build(transformer)

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.dir_interval = dir_interval

        self.positional_encoding = MODELS.build(positional_encoding)
        self.loss_pts = MODELS.build(loss_pts)
        self.loss_dir = MODELS.build(loss_dir)
        self.loss_cls = MODELS.build(loss_cls)
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 2 * 2))
        reg_branch = nn.Sequential(*reg_branch)

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        self.query_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas):
        mlvl_feats = [x.float() for x in mlvl_feats]
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # import pdb;pdb.set_trace()
        object_query_embeds = self.query_embedding.weight.to(dtype)

        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                   device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        outputs = self.transformer(
            mlvl_feats,
            None,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl]).float()
            tmp = self.reg_branches[lvl](hs[lvl]).float()
            tmp = tmp.view(*tmp.shape[:-1], 2, 2)
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp = tmp.sigmoid()  # cx,cy,w,h
            # import pdb;pdb.set_trace()
            # tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
            #                  self.pc_range[0]) + self.pc_range[0])
            # tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
            #                  self.pc_range[1]) + self.pc_range[1])
            # tmp = tmp.reshape(bs, self.num_vec,-1)
            # TODO: check if using sigmoid
            # outputs_pts_coord = tmp.view(tmp.shape[0], self.num_vec, self.num_pts_per_vec, 2)
            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(tmp)

        outputs_classes = torch.stack(outputs_classes)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_line_preds': outputs_pts_coords,
        }

        return outs

    def get_targets_single(self,
                           cls_score,
                           pts_pred,
                           gt_labels,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        num_lines = pts_pred.size(0)
        # assigner and sampler
        assign_result, order_index = self.assigner.assign(cls_score, pts_pred,
                                                          gt_labels, gt_shifts_pts,
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
            gt_bboxes=gt_shifts_pts,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)

        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_shifts_pts.new_full((num_lines,),
                                        self.cls_out_channels,
                                        dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_shifts_pts.new_ones(num_lines)

        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                                          pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    pts_preds_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self.get_targets_single, cls_scores_list, pts_preds_list,
            gt_labels_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    pts_preds,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, pts_preds_list,
                                           gt_labels_list, gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
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

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_pts_targets.flatten(1)).all(dim=-1)
        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))

        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :], normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval:, :] - denormed_pts_preds[:,
                                                                                :-self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval:, :] - pts_targets[:, :-self.dir_interval, :]
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :], pts_targets_dir[isnotnan,
                                                    :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_pts, loss_dir

    def loss(self,
             preds_dicts,
             gt_labels_list,
             gt_lines_list,
             gt_bboxes_ignore=None, ):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_shifts_pts_list = copy.deepcopy(gt_lines_list)
        all_cls_scores = preds_dicts['all_cls_scores']
        all_line_preds = preds_dicts['all_line_preds']

        num_dec_layers = len(all_cls_scores)

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_line_preds, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1], losses_pts[:-1], losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        return loss_dict

    def get_preds(self, preds_dicts):
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        return preds_dicts


@MODELS.register_module()
class MapHead(nn.Module):
    def __init__(
        self,
        num_classes,
        transformer=None,
        bbox_coder=None,
        assigner=None,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
    ):
        super(MapHead, self).__init__()
        self.bg_cls_weight = 0
        self.dir_interval = 1
        self.num_classes = num_classes
        self.transformer = MODELS.build(transformer)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.assigner = TASK_UTILS.build(assigner)
        self.sampler = TASK_UTILS.build(dict(type="PseudoSampler"))
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_pts = MODELS.build(loss_pts)
        self.loss_dir = MODELS.build(loss_dir)

    def forward(self, mlvl_feats, img_metas):
        pass

    def get_targets_single(
        self, cls_score, pts_pred, gt_labels, gt_shifts_pts, gt_bboxes_ignore=None
    ):
        num_lines = pts_pred.size(0)
        # assigner and sampler
        assign_result, order_index = self.assigner.assign(
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
        return (labels, label_weights, pts_targets, pts_weights, pos_inds, neg_inds)

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

    def loss_single(
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
        dir_weights = pts_weights[:, : -self.dir_interval, 0]

        denormed_pts_preds = pts_preds
        if self.use_norm_coord:
            denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = (
            denormed_pts_preds[:, self.dir_interval :, :]
            - denormed_pts_preds[:, : -self.dir_interval, :]
        )
        pts_targets_dir = (
            pts_targets[:, self.dir_interval :, :]
            - pts_targets[:, : -self.dir_interval, :]
        )
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_pts, loss_dir

    def loss(
        self,
        preds_dicts,
        gt_labels_list,
        gt_lines_list,
        gt_bboxes_ignore=None,
    ):
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )
        gt_shifts_pts_list = copy.deepcopy(gt_lines_list)
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_line_preds = preds_dicts["all_line_preds"]

        num_dec_layers = len(all_cls_scores)

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_pts, losses_dir = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_line_preds,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_pts"] = losses_pts[-1]
        loss_dict["loss_dir"] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(
            losses_cls[:-1], losses_pts[:-1], losses_dir[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_pts"] = loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_dir"] = loss_dir_i
            num_dec_layer += 1
        return loss_dict

    def get_preds(self, preds_dicts):
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        return preds_dicts


@MODELS.register_module()
class MapTRv2Head(MapHead):
    def __init__(
            self,
            with_box_refine=False,
            as_two_stage=False,
            num_cls_fcs=2,
            code_weights=None,
            bev_h=30,
            bev_w=30,
            #  num_vec=20,
            num_vec_one2one=50,
            num_vec_one2many=0,
            k_one2many=0,
            lambda_one2many=1,
            num_pts_per_vec=2,
            num_pts_per_gt_vec=2,
            query_embed_type='all_pts',
            gt_shift_pts_pattern='v0',
            dir_interval=1,
            use_norm_coord=True,
            num_classes=3,
            transformer=None,
            bbox_coder=None,
            assigner=None,
            loss_cls=dict(
                type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            ),
            loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
            loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
    ):
        super(MapTRv2Head, self).__init__(
            num_classes=num_classes,
            transformer=transformer,
            bbox_coder=bbox_coder,
            assigner=assigner,
            loss_cls=loss_cls,
            loss_dir=loss_dir,
            loss_pts=loss_pts,
        )
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.code_size = 2
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        self.gt_shift_pts_pattern = gt_shift_pts_pattern

        num_vec = num_vec_one2one + num_vec_one2many
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if 'BEVFormerEncoder' in self.bev_encoder_type:
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas):
        mlvl_feats = [mlvl_feats[0]]

        key_points = self.key_points.weight.reshape(256, 2, 2)
        query_pos = pos2posemb2d(key_points).flatten(1)
        qyery_pos = self.key_points_query_embedding(query_pos)

        outs_dec, key_line_list = self.transformer(
            mlvl_feats, None, qyery_pos, None, self.key_line_branch, img_metas
        )
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            # reference = inverse_sigmoid(reference_points.clone())
            outputs_class = self.cls_branches[lvl](outs_dec[lvl]).float()
            tmp = self.reg_branches[lvl](outs_dec[lvl]).float()
            tmp = tmp.view(*tmp.shape[:-1], self.num_points, self.coord_dim)
            # tmp += reference[..., 0:2].view(*reference.shape[:2], 1, 2).repeat(1, 1, self.num_points, 1)
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_line_preds = torch.stack(outputs_coords)

        if not self.use_norm_coord:
            all_line_preds[..., 0:1] = (
                    all_line_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                    + self.pc_range[0]
            )
            all_line_preds[..., 1:2] = (
                    all_line_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                    + self.pc_range[1]
            )

        outputs = {"all_cls_scores": all_cls_scores, "all_line_preds": all_line_preds}

        return outputs

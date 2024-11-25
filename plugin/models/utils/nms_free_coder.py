from mmdet.models.task_modules import BaseBBoxCoder
from mmdet3d.registry import TASK_UTILS

from plugin.models.utils.utils import denormalize_2d_pts


@TASK_UTILS.register_module()
class MapTRNMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self, pc_range, max_num=100, score_threshold=None,
                 num_classes=10, use_norm_coord=True, **kwargs):
        super().__init__(**kwargs)
        self.pc_range = pc_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.use_norm_coord = use_norm_coord

    def encode(self):
        pass

    def decode_single(self, cls_scores, pts_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        pts_preds = pts_preds[bbox_index]

        final_pts_preds = pts_preds
        if self.use_norm_coord:
            final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)  # num_q,num_p,2
        final_scores = scores
        final_preds = labels

        predictions_dict = {
            'scores': final_scores,
            'labels': final_preds,
            'pts': final_pts_preds,
        }
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_pts_preds[i]))
        return predictions_list


@TASK_UTILS.register_module()
class LineTRNMSFreeCoder(BaseBBoxCoder):
    def __init__(self, pc_range, max_num=100, score_threshold=None,
                 num_classes=10, use_norm_coord=True, **kwargs):
        super().__init__(**kwargs)
        self.pc_range = pc_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.use_norm_coord = use_norm_coord

    def encode(self):
        pass

    def decode_single(self, cls_scores, pts_preds, key_cls_scores, key_pts_preds):
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        pts_preds = pts_preds[bbox_index]

        final_pts_preds = pts_preds
        if self.use_norm_coord:
            final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)  # num_q,num_p,2
        final_scores = scores
        final_preds = labels

        key_cls_scores = key_cls_scores.sigmoid()
        num_key_line = len(key_cls_scores)
        key_scores, indexs = key_cls_scores.view(-1).topk(num_key_line)
        key_labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        key_pts_preds = key_pts_preds[bbox_index]

        final_key_pts_preds = key_pts_preds
        if self.use_norm_coord:
            final_key_pts_preds = denormalize_2d_pts(key_pts_preds, self.pc_range)  # num_q,num_p,2
        final_key_scores = key_scores
        final_key_preds = key_labels

        predictions_dict = {
            'scores': final_scores,
            'labels': final_preds,
            'pts': final_pts_preds,
            'key_scores': final_key_scores,
            'key_labels': final_key_preds,
            'key_pts': final_key_pts_preds
        }
        return predictions_dict

    def decode(self, preds_dicts):
        all_raw_points_score = preds_dicts['all_raw_points_score'][-1]
        all_raw_points = preds_dicts['all_raw_points'][-1]
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_pts_preds[i], all_raw_points_score[i], all_raw_points[i]))
        return predictions_list

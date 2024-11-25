# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union
import torch
from mmengine.model.base_model import BaseDataPreprocessor
from mmdet3d.registry import MODELS


@MODELS.register_module()
class MapDataPreprocessor(BaseDataPreprocessor):

    def __init__(self,
                 non_blocking: Optional[bool] = False,
                 ):
        super(MapDataPreprocessor, self).__init__(non_blocking=non_blocking)

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        return self.simple_process(data, training)

    def simple_process(self, data: dict, training: bool = False) -> dict:
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        return {'inputs': inputs, 'data_samples': data_samples}

    def collate_data(self, data: dict) -> dict:
        data['inputs']['imgs'] = torch.stack(data['inputs']['img']).to(self.device)

        # if 'points' in data['inputs']:
        #     num_point = max([len(item) for item in data['inputs']['points']])
        #     for i, item in enumerate(data['inputs']['points']):
        #         padding_point = torch.zeros([num_point - len(item), item.shape[-1]], dtype=torch.float32)
        #         data['inputs']['points'][i] = torch.cat((item, padding_point), dim=0)
        #     data['inputs']['points'] = torch.stack(data['inputs']['points']).to(self.device)

        del data['inputs']['img']

        if "image_depths" in data["inputs"]:
            data["inputs"]["image_depths"] = torch.stack(data["inputs"]["image_depths"]).to(self.device)

        if 'gt_labels_3d' in data['inputs']:
            data['inputs']['gt_labels_3d'] = [item.to(self.device) for item in data['inputs']['gt_labels_3d']]
        if 'gt_bboxes_3d' in data['inputs']:
            gt_shifts_pts_list = [gt_lines.shift_fixed_num_sampled_points_v2.to(self.device)[..., :2] for gt_lines in
                                  data['inputs']['gt_bboxes_3d']]
            # gt_pts_list = []
            # for gt_lines in data['inputs']['gt_bboxes_3d']:
            #     new_lines = []
            #     lines = gt_lines.fixed_num_sampled_points.to(self.device)
            #     for line in lines:
            #         if not torch.equal(line[0], line[-1]):
            #             max_distance_x = torch.abs(torch.max(line[:, 0]) - torch.min(line[:, 0]))
            #             max_distance_y = torch.abs(torch.max(line[:, 1] - torch.min(line[:, 1])))
            #             if max_distance_y > max_distance_x:
            #                 if line[0, 1] < line[-1, 1]:
            #                     line = torch.flip(line, dims=(0,))
            #             else:
            #                 if line[0, 0] > line[-1, 0]:
            #                     line = torch.flip(line, dims=(0,))
            #         new_lines.append(line)
            #     gt_pts_list.append(torch.stack(new_lines))
            # gt_pts_list = [gt_lines.fixed_num_sampled_points.to(self.device) for gt_lines in
            #                data['inputs']['gt_bboxes_3d']]
            data['inputs']['gt_lines_3d'] = gt_shifts_pts_list
            # data['inputs']['gt_raw_lines'] = gt_pts_list

        if 'raw_points' in data['inputs']:
            data['inputs']['raw_points'] = [item.to(self.device) for item in data['inputs']['raw_points']]
            data['inputs']['raw_points_label'] = [item.to(self.device) for item in data['inputs']['raw_points_label']]
            if "point_masks" in data["inputs"]:
                data['inputs']['point_masks'] = [item.to(self.device) for item in data['inputs']['point_masks']]

        if "bev_points" in data['inputs']:
            data['inputs']['bev_points'] = torch.stack([item for item in data['inputs']['bev_points']]).to(self.device)

        for sample in data['data_samples']:
            metainfo = sample.metainfo
            if 'target' in sample.metainfo.keys():
                # if 'bboxes' in sample.metainfo['target'].keys():
                #     sample.target['bboxes'] = sample.target['bboxes'].to(self.device)
                for key in sample.target.keys():
                    sample.target[key] = sample.target[key].to(self.device)
            if 'maps' in sample.metainfo.keys():
                # sample
                metainfo['maps'] = metainfo['maps'].to(self.device)
            if 'pos_points' in sample.metainfo.keys():
                metainfo['pos_points'] = [item.to(self.device) for item in metainfo['pos_points']]
            if 'image_depths' in sample.metainfo.keys():
                metainfo['image_depths'] = [torch.stack(item, 0).to(self.device) for item in metainfo['image_depths']]
            sample.set_metainfo(metainfo)
        return data

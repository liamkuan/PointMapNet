import copy
from typing import Optional, Union, Sequence

from mmengine.registry import DATASETS
from .base_dataset import BaseMapDataset
from datetime import datetime


@DATASETS.register_module()
class AV2Dataset(BaseMapDataset):
    def __init__(self,
                 ann_file,
                 data_root=None,
                 cat2id=None,
                 modality=dict(
                     use_camera=True,
                     use_lidar=False,
                     use_radar=False,
                     use_map=True,
                     use_external=False,
                 ),
                 pipeline=None,
                 prev_frame: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 ):
        super().__init__(
            ann_file=ann_file,
            modality=modality,
            data_root=data_root,
            pipeline=pipeline,
            cat2id=cat2id,
            indices=indices,
        )
        self.prev_frame = prev_frame

    def prepare_data(self, index: int):
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = copy.deepcopy(self.get_data_info(index))
        input_dict = {
            'log_id': ori_input_dict['log_id'],
            'img_filenames': [c['img_fpath'] for c in ori_input_dict['cams'].values()],
            'ego2img': ori_input_dict['ego2cam_se3'],
            'ego2global': ori_input_dict['ego2world'],
            'map_path': ori_input_dict['map_path'],
            'timestamp': ori_input_dict['timestamp'],
            'map_data_path': ori_input_dict['map_data_path'],
            'sample_idx': str(ori_input_dict['log_id']) + str(ori_input_dict['timestamp'])
            # 'target': ori_input_dict['target']
        }

        if self.modality['use_lidar']:
            input_dict['lidar_points'] = ori_input_dict['lidar_points']

        if self.prev_frame and 'prev_frame' in ori_input_dict.keys():
            input_dict['img_filenames'].extend(
                [item['img_fpath'] for item in ori_input_dict['prev_frame']['cams'].values()])
            cam_timestamp_current = [item['timestamp'] for item in ori_input_dict['cams'].values()]
            cam_timestamp_prev = [item['timestamp'] for item in ori_input_dict['prev_frame']['cams'].values()]
            input_dict['cam_timestamp'] = cam_timestamp_current + cam_timestamp_prev
            input_dict['ego2img'].extend(ori_input_dict['prev_frame']['ego2cam_se3'])
        # now = datetime.now()
        example = self.pipeline(input_dict)
        # print(f'pipeline time: {datetime.now() - now}')
        return example

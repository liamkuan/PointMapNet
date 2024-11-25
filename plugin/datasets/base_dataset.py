import copy
from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import tempfile
from os import path as osp
from mmengine.dataset.base_dataset import BaseDataset
from mmengine.registry import DATASETS


@DATASETS.register_module()
class BaseMapDataset(BaseDataset):
    """BaseClass for Map Dataset

    This is the base dataset of nuScenes and argoverse2 dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 ann_file='',
                 data_root='',
                 modality=dict(
                     use_camera=True,
                     use_lidar=False,
                     use_radar=False,
                     use_map=True,
                     use_external=False,
                 ),
                 pipeline=None,
                 indices: Optional[Union[int, Sequence[int]]] = None,):
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            data_prefix=dict(),
            indices=indices
        )
        self.ann_file = ann_file
        self.modality = modality
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_data(self, index: int):
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        input_dict = copy.deepcopy(self.get_data_info(index))
        example = self.pipeline(input_dict)
        return example

# PointMapNet
Enhancing Real-Time Geospatial Analytics with PointMapNet: A Scalable Approach for High-Definition Map Construction

## Models

#### nuScenes dataset

| Method  | Backbone | Epoch | mAP |                         Config                          |                                                                   Download                                                                   |
|:-------:|:--------:|:-----:|:----:|:-------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
|  PointMapNet  |   R50    |  24   | 55.3 | [config](config/pointmapnet.py) | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155168294_link_cuhk_edu_hk/EXrwWu0yvz5Ap_aU9FFb4x8BahsKfdFgYW7TgnpsIKho2Q?e=s2CnGT) |
|  PointMapNet  |   R50    |  110  | 62.2 | [config](config/pointmapnet.py) | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155168294_link_cuhk_edu_hk/EV-zT_ZOIaNEvOCMNuEqjIAB2GnV8HzR-cfskRmdOJcBPQ?e=uOefF4) |

- mAP is measured under the thresholds { 0.5, 1.0, 1.5 }

## Getting Started
### Installation
Follow by mmdet3d v1.3.0
- [Installation](https://mmdetection3d.readthedocs.io/en/v1.3.0/get_started.html)
### Prepare Dataset
```python tools/data_converter/nuscenes_converter.py --data-root data/nuscenes --canbus```
### Train and Eval
```
python -m tools.train config/pointmapnet.py
python -m tools.train config/pointmapnet.py work_dirs/pointmapnet/epoch_24.pth
```

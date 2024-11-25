import argparse
import os
from functools import partial

import mmcv
from os import path as osp
from pathlib import Path
import mmengine
import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import numpy as np
import av2.utils.raster as raster_utils
from mmengine.utils import track_parallel_progress, track_progress
from plugin.datasets.map_utils.av2_map_api import get_vector_map
from av2.map.map_api import ArgoverseStaticMap
from av2.map.drivable_area import DrivableArea
from av2.map.lane_segment import LaneSegment
from av2.map.pedestrian_crossing import PedestrianCrossing

CAMERAS = ['ring_front_left', 'ring_front_right', 'ring_rear_left',
           'ring_rear_right', 'ring_side_left', 'ring_side_right']

MAP_CHANGE_LOG = {'75e8adad-50a6-3245-8726-5e612db3d165',
                  '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
                  'af170aac-8465-3d7b-82c5-64147e94af7d',
                  '6e106cf8-f6dd-38f6-89c8-9be7a71e7275'}


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    # parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--map_range', default=[-30, -15, 30, 15])
    parser.add_argument('--class2label', default={
        'ped_crossing': 0,
        'divider': 1,
        'drivable_boundary': 2,
        'others': -1,
        # 'centerline': 3,
    })
    parser.add_argument('--densities', default=2)
    args = parser.parse_args()
    return args


def get_samples(log, root_path, dir_path, av2, class2label, map_range, densities):
    map_data_dir_path = os.path.join(root_path, 'map_data', log)
    if not osp.exists(map_data_dir_path):
        os.mkdir(map_data_dir_path)
    map_path = os.path.join(dir_path, log, 'map')
    av_map = ArgoverseStaticMap.from_map_dir(Path(map_path).absolute(), True)
    ego2cam_list = []
    samples = []

    try:
        for camera in CAMERAS:
            cam_log = av2.get_log_pinhole_camera(log, camera)
            intrinsic = cam_log.intrinsics.K
            ego2cam = cam_log.ego_SE3_cam.inverse().transform_matrix
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam = viewpad @ ego2cam
            ego2cam_list.append(ego2cam)
    except KeyError:
        return []
    for timestamp in av2.get_ordered_log_lidar_timestamps(log):
        ego2world = av2.get_city_SE3_ego(log, timestamp)
        ego2world = torch.tensor(ego2world.transform_matrix, dtype=torch.float32)
        lidar_path = av2.get_lidar_fpath(log, timestamp)
        sample = {'log_id': log,
                  'lidar_points': {
                      'lidar_path': str(lidar_path.absolute()),
                  },
                  "timestamp": timestamp,
                  'ego2cam_se3': ego2cam_list,
                  'ego2world': ego2world,
                  'map_path': map_path,
                  'map_data_path': osp.join(root_path, 'map_data', log, f"{timestamp}_map_data.pkl"),
                  'cams': {}}
        flag = True
        for camera in CAMERAS:
            fpath = av2.get_closest_img_fpath(log, camera, timestamp)
            if fpath is None:
                flag = False
                break
            cam_timestamp = int(fpath.name.split('.')[0])
            cam_info = dict(
                img_fpath=str(fpath.absolute()),
                timestamp=cam_timestamp
            )
            sample['cams'][camera] = cam_info
        if flag:
            polylines, labels = get_vector_map(av_map, ego2world, class2label, map_range)

            lane_segments = av_map.get_scenario_lane_segments()
            ped_crossings = av_map.get_scenario_ped_crossings()
            drivable_areas = av_map.get_scenario_vector_drivable_areas()
            layers = [_get_mask(lane_segments, ego2world, map_range, densities),
                      _get_mask(ped_crossings, ego2world, map_range, densities),
                      _get_mask(drivable_areas, ego2world, map_range, densities)]
            maps = torch.stack(layers, dim=0)
            if len(polylines) > 0:
                map_data = {'lines': polylines, 'labels': labels, 'maps': maps}
                mmengine.dump(map_data, osp.join(map_data_dir_path, f"{timestamp}_map_data.pkl"))
                samples.append(sample)

    # new_samples = []
    # for i, sample in enumerate(samples[1:]):
    #     sample['prev_frame'] = {}
    #     sample['prev_frame']['timestamp'] = samples[i]['timestamp']
    #     prev_ego2world = samples[i]['ego2world']
    #     ego2cam_se3 = []
    #     for ego2cam in samples[i]['ego2cam_se3']:
    #         world2ego_current = np.linalg.inv(sample['ego2world'].numpy())
    #         cam2ego = world2ego_current @ prev_ego2world.numpy() @ np.linalg.inv(ego2cam)
    #         ego2cam_se3.append(np.linalg.inv(cam2ego))
    #     sample['prev_frame']['ego2cam_se3'] = ego2cam_se3
    #     sample['prev_frame']['cams'] = samples[i]['cams']
    #     new_samples.append(sample)
    # _resave_imgs(new_samples, root_path)
    return samples


def _resave_imgs(samples, root_path):
    for sample in samples:
        filenames = [c['img_fpath'] for c in sample['cams'].values()]
        filenames = filenames + [c['img_fpath'] for c in sample['prev_frame']['cams'].values()]
        img = [mmcv.imread(name, 'unchanged') for name in filenames]
        dir_path = osp.join(root_path, 'map_data', sample['log_id'])
        if not osp.exists(dir_path):
            os.mkdir(dir_path)
        img_path = osp.join(dir_path, f"{sample['timestamp']}_img.pkl")
        mmengine.dump(img, img_path)
        sample['img_path'] = img_path


def generate_vector_map(log, av2, root_path, dir_path, class2label, map_range, densities):
    map_path = os.path.join(dir_path, log, 'map')
    av_map = ArgoverseStaticMap.from_map_dir(
        Path(map_path).absolute(), True)
    empty_sample_set = set()
    for timestamp in av2.get_ordered_log_lidar_timestamps(log):
        world_se3 = av2.get_city_SE3_ego(log, timestamp)
        world_se3 = torch.tensor(
            world_se3.transform_matrix, dtype=torch.float32)

        polylines, labels = get_vector_map(
            av_map, world_se3, class2label, map_range)

        lane_segments = av_map.get_scenario_lane_segments()
        ped_crossings = av_map.get_scenario_ped_crossings()
        drivable_areas = av_map.get_scenario_vector_drivable_areas()
        layers = [_get_mask(lane_segments, world_se3, map_range, densities),
                  _get_mask(ped_crossings, world_se3, map_range, densities),
                  _get_mask(drivable_areas, world_se3, map_range, densities)]
        maps = torch.stack(layers, dim=0)

        if len(polylines) > 0:
            dir_path = os.path.join(
                root_path, 'map_data', log)
            map_data = {'lines': polylines, 'labels': labels, 'maps': maps}
            mmengine.dump(map_data, osp.join(
                dir_path, f"{timestamp}_map_data.pkl"))
        else:
            empty_sample_set.add(log + str(timestamp))
    return empty_sample_set


def _get_mask(areas, ego2global, map_range, densities):
    polygons = []
    for da in areas:
        if type(da) is LaneSegment:
            points = torch.cat(
                [torch.from_numpy(da.polygon_boundary), torch.ones([da.polygon_boundary.shape[0], 1])],
                dim=-1).unsqueeze(-1)
        elif type(da) is PedestrianCrossing:
            points = torch.cat(
                [torch.from_numpy(da.polygon), torch.ones([da.polygon.shape[0], 1])],
                dim=-1).unsqueeze(-1)
        elif type(da) is DrivableArea:
            points = torch.cat([torch.from_numpy(da.xyz), torch.ones([da.xyz.shape[0], 1])], dim=-1).unsqueeze(-1)
        else:
            return None
        points = points.to(torch.float32)
        polygon = torch.inverse(ego2global) @ points
        polygon = polygon.squeeze(-1)[:, :2]
        polygon = polygon + torch.tensor([map_range[2], map_range[3]])
        polygon = polygon * densities
        polygons.append(polygon.numpy())
    w = map_range[3] - map_range[1]
    h = map_range[2] - map_range[0]
    da_array = raster_utils.get_mask_from_polygons(polygons, w * densities, h * densities)
    return torch.as_tensor(da_array, dtype=torch.float32)


class SampleTask:
    def __init__(self, root_path, dir_path, av2, class2label, map_range, densities=2):
        self.root_path = root_path
        self.dir_path = dir_path
        self.av2 = av2
        self.class2label = class2label
        self.map_range = map_range
        self.densities = densities

    def run(self, log):
        map_data_dir_path = os.path.join(self.root_path, 'map_data', log)
        if not osp.exists(map_data_dir_path):
            os.mkdir(map_data_dir_path)
        dir_path = self.dir_path
        av2 = self.av2
        map_path = os.path.join(dir_path, log, 'map')
        ego2cam_list = []
        samples = []

        try:
            for camera in CAMERAS:
                cam_log = av2.get_log_pinhole_camera(log, camera)
                intrinsic = cam_log.intrinsics.K
                ego2cam = cam_log.ego_SE3_cam.inverse().transform_matrix
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2cam = viewpad @ ego2cam
                ego2cam_list.append(ego2cam)
        except KeyError:
            return []
        for timestamp in av2.get_ordered_log_lidar_timestamps(log):
            ego2world = av2.get_city_SE3_ego(log, timestamp)
            ego2world = torch.tensor(
                ego2world.transform_matrix, dtype=torch.float32)
            lidar_path = av2.get_lidar_fpath(log, timestamp)
            sample = {'log_id': log,
                      'lidar_points': {
                          'lidar_path': str(lidar_path.absolute()),
                      },
                      "timestamp": timestamp,
                      'ego2cam_se3': ego2cam_list,
                      'ego2world': ego2world,
                      'map_path': map_path,
                      'map_data_path': osp.join(self.root_path, 'map_data', log, f"{timestamp}_map_data.pkl"),
                      'cams': {}}
            flag = True
            for camera in CAMERAS:
                fpath = av2.get_closest_img_fpath(log, camera, timestamp)
                if fpath is None:
                    flag = False
                    break
                cam_timestamp = int(fpath.name.split('.')[0])
                cam_info = dict(
                    img_fpath=str(fpath.absolute()),
                    timestamp=cam_timestamp
                )
                sample['cams'][camera] = cam_info
            if flag:
                samples.append(sample)

        new_samples = []
        for i, sample in enumerate(samples[1:]):
            sample['prev_frame'] = {}
            sample['prev_frame']['timestamp'] = samples[i]['timestamp']
            prev_ego2world = samples[i]['ego2world']
            ego2cam_se3 = []
            for ego2cam in samples[i]['ego2cam_se3']:
                world2ego_current = np.linalg.inv(sample['ego2world'].numpy())
                cam2ego = world2ego_current @ prev_ego2world.numpy() @ np.linalg.inv(ego2cam)
                ego2cam_se3.append(np.linalg.inv(cam2ego))
            sample['prev_frame']['ego2cam_se3'] = ego2cam_se3
            sample['prev_frame']['cams'] = samples[i]['cams']
            new_samples.append(sample)
        self._resave_imgs(new_samples)
        return new_samples

    def generate_vector_map(self, log):
        map_path = os.path.join(self.dir_path, log, 'map')
        av_map = ArgoverseStaticMap.from_map_dir(
            Path(map_path).absolute(), True)
        empty_sample_set = set()
        for timestamp in self.av2.get_ordered_log_lidar_timestamps(log):
            world_se3 = self.av2.get_city_SE3_ego(log, timestamp)
            world_se3 = torch.tensor(
                world_se3.transform_matrix, dtype=torch.float32)

            polylines, labels = get_vector_map(
                av_map, world_se3, self.class2label, self.map_range)

            lane_segments = av_map.get_scenario_lane_segments()
            ped_crossings = av_map.get_scenario_ped_crossings()
            drivable_areas = av_map.get_scenario_vector_drivable_areas()
            layers = [self._get_mask(lane_segments, world_se3), self._get_mask(ped_crossings, world_se3),
                      self._get_mask(drivable_areas, world_se3)]
            maps = torch.stack(layers, dim=0)

            if len(polylines) > 0:
                dir_path = os.path.join(
                    self.root_path, 'map_data', log)
                map_data = {'lines': polylines, 'labels': labels, 'maps': maps}
                mmengine.dump(map_data, osp.join(
                    dir_path, f"{timestamp}_map_data.pkl"))
            else:
                empty_sample_set.add(log + str(timestamp))
        return empty_sample_set

    def _resave_imgs(self, samples):
        for sample in samples:
            filenames = [c['img_fpath'] for c in sample['cams'].values()]
            filenames = filenames + [c['img_fpath'] for c in sample['prev_frame']['cams'].values()]
            img = [mmcv.imread(name, 'unchanged') for name in filenames]
            dir_path = osp.join(self.root_path, 'map_data', sample['log_id'])
            if not osp.exists(dir_path):
                os.mkdir(dir_path)
            img_path = osp.join(dir_path, f"{sample['timestamp']}_img.pkl")
            mmengine.dump(img, img_path)
            sample['img_path'] = img_path

    def _get_mask(self, areas, ego2global):
        polygons = []
        for da in areas:
            if type(da) is LaneSegment:
                points = torch.cat(
                    [torch.from_numpy(da.polygon_boundary), torch.ones([da.polygon_boundary.shape[0], 1])],
                    dim=-1).unsqueeze(-1)
            elif type(da) is PedestrianCrossing:
                points = torch.cat(
                    [torch.from_numpy(da.polygon), torch.ones([da.polygon.shape[0], 1])],
                    dim=-1).unsqueeze(-1)
            elif type(da) is DrivableArea:
                points = torch.cat([torch.from_numpy(da.xyz), torch.ones([da.xyz.shape[0], 1])], dim=-1).unsqueeze(-1)
            else:
                return None
            points = points.to(torch.float32)
            polygon = torch.inverse(ego2global) @ points
            polygon = polygon.squeeze(-1)[:, :2]
            polygon = polygon + torch.tensor([self.map_range[2], self.map_range[3]])
            polygon = polygon * self.densities
            polygons.append(polygon.numpy())
        w = self.map_range[3] - self.map_range[1]
        h = self.map_range[2] - self.map_range[0]
        da_array = raster_utils.get_mask_from_polygons(polygons, w * self.densities, h * self.densities)
        return torch.as_tensor(da_array, dtype=torch.float32)


def get_infos_map(root_path, dir_name, args):
    path = Path(os.path.join(root_path, 'sensor', dir_name))
    av2 = AV2SensorDataLoader(data_dir=path, labels_dir=path)
    dir_path = osp.join(root_path, 'map_data')
    if not osp.exists(dir_path):
        os.mkdir(dir_path)

    task_list = []
    for log in os.scandir(os.path.join(root_path, 'sensor', dir_name)):
        if log.is_dir():
            if log.name in MAP_CHANGE_LOG:
                continue
            task_list.append(log.name)
    dir_path = os.path.join(root_path, 'sensor', dir_name)
    samples = []
    # results = track_parallel_progress(partial(get_samples,
    #                                  root_path=root_path,
    #                                  dir_path=dir_path,
    #                                  av2=av2,
    #                                  class2label=args.class2label,
    #                                  map_range=args.map_range,
    #                                  densities=args.densities), task_list, 32)
    results = track_progress(partial(get_samples,
                                     root_path=root_path,
                                     dir_path=dir_path,
                                     av2=av2,
                                     class2label=args.class2label,
                                     map_range=args.map_range,
                                     densities=args.densities), task_list)
    for result in results:
        samples.extend(result)

    # 下面这行只为了生成矢量地图数据，如果是更改其他数据，这行可以注释掉
    # results = track_parallel_progress(partial(generate_vector_map,
    #                                           av2=av2,
    #                                           root_path=root_path,
    #                                           dir_path=dir_path,
    #                                           class2label=args.class2label,
    #                                           map_range=args.map_range,
    #                                           densities=args.densities), task_list, 32)
    # empty_sample = set()
    # for result in results:
    #     empty_sample = empty_sample.union(result)
    # print(len(empty_sample))
    # new_samples = []
    # for sample in samples:
    #     if sample['log_id'] + str(sample['timestamp']) not in empty_sample:
    #         new_samples.append(sample)
    # samples = new_samples

    return samples


def create_argoverse2_infos_map(root_path, args,
                                info_prefix='argoverse2'):
    """Create info file for map learning task on argoverse2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
    """
    train_samples = get_infos_map(root_path, 'train', args)
    val_samples = get_infos_map(root_path, 'val', args)

    # for training set
    train_ann_dict = {'metainfo': {
        "dataset_type": "train_dataset",
        "task_name": "train_task"
    }, 'data_list': train_samples}
    info_path = osp.join(root_path, f'{info_prefix}_map_infos_train.pkl')
    print(f'saving training set to {info_path}')
    mmengine.dump(train_ann_dict, info_path)

    # for val set
    val_ann_dict = {'metainfo': {
        "dataset_type": "val_dataset",
        "task_name": "val_task"
    }, 'data_list': val_samples}
    info_path = osp.join(root_path, f'{info_prefix}_map_infos_val.pkl')
    print(f'saving validation set to {info_path}')
    mmengine.dump(val_ann_dict, info_path)


if __name__ == '__main__':
    args = parse_args()

    create_argoverse2_infos_map(args.data_root, args)

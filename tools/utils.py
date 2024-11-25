import os
from pathlib import Path
import pickle

import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap, DrivableAreaMapLayer
import av2.utils.raster as raster_utils
from matplotlib import pyplot as plt
import numpy as np
import datetime
import mmengine
from mmengine.runner import Runner
from mmengine.config import Config
from torch import Tensor
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

CAMERAS = [
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
]


def show_polygon(polygons):
    ax = plt.subplot()
    ax.set_xlim(-60, 60)
    ax.set_ylim(-30, 30)
    ax.set_aspect(1)
    for polygon in polygons:
        color = "green"
        p = plt.Polygon(xy=polygon[:, :2].numpy(), alpha=0.6, color=color)
        ax.add_patch(p)
    plt.show()


def get_drivable_area():
    torch.set_printoptions(precision=4, sci_mode=False)
    path = Path(r"D:\code\python\kmapnet\data\av2\sensor\train")
    av2 = AV2SensorDataLoader(data_dir=path, labels_dir=path)
    log_id = "0a8a4cfa-4902-3a76-8301-08698d6290a2"
    log_map_dirpath = path / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)
    per = 2
    da_array = avm.raster_drivable_area_layer.array
    for timestamp in av2.get_ordered_log_lidar_timestamps(log_id):
        world_se3 = av2.get_city_SE3_ego(log_id, timestamp)
        world_se3 = torch.from_numpy(world_se3.transform_matrix)
        drivable_areas = avm.get_scenario_vector_drivable_areas()
        polygons = []
        for da in drivable_areas:
            points = torch.cat(
                [torch.from_numpy(da.xyz), torch.ones([da.xyz.shape[0], 1])], dim=-1
            ).unsqueeze(-1)
            polygon = torch.inverse(world_se3) @ points
            polygon = polygon.squeeze(-1)[:, :2]
            polygon = polygon + torch.tensor([60, 30])
            polygon = polygon * per
            # polygons.append(polygon)
            polygons.append(polygon.numpy())
        # show_polygon(polygons)
        da_array = raster_utils.get_mask_from_polygons(polygons, 60 * per, 120 * per)
        ax = plt.subplot()
        plt.imshow(np.flipud(da_array))
        plt.show()
        break


def show_point_cloud():
    coords3d = mmengine.load(os.path.join("coords3d.pkl"))
    coords3d = coords3d[:, 6:]
    coords3d = coords3d.flatten(0, 4).cpu().numpy()
    np.random.shuffle(coords3d)
    coords3d = coords3d[:800]
    plt.style.use("_mpl-gallery")

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(coords3d[:, 0], coords3d[:, 1], coords3d[:, 2])

    # ax.set(xticklabels=[],
    #     yticklabels=[],
    #     zticklabels=[])
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)

    plt.savefig("result.png")
    print(coords3d.shape)


def clear_short_log():
    import re
    import shutil

    delete_list = []
    for log in os.scandir(os.path.join("./out")):
        if log.is_dir() and log.name.startswith("2023"):
            if not log.name.startswith("20230906"):
                with open(os.path.join("./out", log.name, log.name + ".log"), "r") as f:
                    lines = f.readlines()
                    if len(lines) > 2000:
                        # print(lines[-1])
                        flag = True
                        for line in lines[-20:]:
                            index = line.find("Epoch(train)")
                            if index > 0:
                                tmp = re.search(
                                    r"Epoch\(train\) +\[\w*\]", line, flags=0
                                ).span()
                                tmp_str = line[tmp[0] : tmp[1]]
                                if (
                                    int(
                                        tmp_str[
                                            tmp_str.find("[") + 1 : tmp_str.find("]")
                                        ]
                                    )
                                    > 10
                                ):
                                    flag = False
                                    break
                        if flag:
                            delete_list.append(log.name)
                    else:
                        delete_list.append(log.name)
    for delete_log in delete_list:
        shutil.rmtree(os.path.join("./out", delete_log))


def is_simple_quad(ab, bc, cd, da):
    temp0 = np.cross(ab, bc)
    temp1 = np.cross(bc, cd)
    temp2 = np.cross(cd, da)
    temp3 = np.cross(da, ab)
    cross = np.array([temp0, temp1, temp2, temp3])
    #   See that cross products are greater than or equal to zero
    crossTF = cross >= 0
    #   if the cross products are majority false, re compute the cross products
    #   Because they don't necessarily need to lie in the same 'Z' direction
    if sum(crossTF) <= 1:
        crossTF = cross <= 0
    if sum(crossTF) > 2:
        return True
    else:
        return False


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def makeQuad(x, y):
    AB = [x[1] - x[0], y[1] - y[0]]
    BC = [x[2] - x[1], y[2] - y[1]]
    CD = [x[3] - x[2], y[3] - y[2]]
    DA = [x[0] - x[3], y[0] - y[3]]

    isQuad = is_simple_quad(AB, BC, CD, DA)

    if isQuad is False:
        # attempt to rearrange the first two points
        x[1], x[0] = x[0], x[1]
        y[1], y[0] = y[0], y[1]
        AB = [x[1] - x[0], y[1] - y[0]]
        BC = [x[2] - x[1], y[2] - y[1]]
        CD = [x[3] - x[2], y[3] - y[2]]
        DA = [x[0] - x[3], y[0] - y[3]]

        isQuad = is_simple_quad(AB, BC, CD, DA)

        if isQuad is False:
            # place the second and first points back where they were, and
            # swap the second and third points
            x[2], x[0], x[1] = x[0], x[1], x[2]
            y[2], y[0], y[1] = y[0], y[1], y[2]
            AB = [x[1] - x[0], y[1] - y[0]]
            BC = [x[2] - x[1], y[2] - y[1]]
            CD = [x[3] - x[2], y[3] - y[2]]
            DA = [x[0] - x[3], y[0] - y[3]]

            isQuad = is_simple_quad(AB, BC, CD, DA)

    # calculate the area via shoelace formula
    area = poly_area(x, y)
    return area


def area_between_two_curves(exp_data, num_data):
    n_exp = len(exp_data)
    n_num = len(num_data)

    area = []
    for i in range(1, n_exp):
        tempX = [exp_data[i - 1, 0], exp_data[i, 0], num_data[i, 0], num_data[i - 1, 0]]
        tempY = [exp_data[i - 1, 1], exp_data[i, 1], num_data[i, 1], num_data[i - 1, 1]]
        area.append(makeQuad(tempX, tempY))
    return np.sum(area)


def _get_area(line: Tensor):
    count_x = 0
    count_y = 0
    for i, point in enumerate(line[1:]):
        count_x += abs(point[0] - line[i][0]) * 0.5 * abs(point[1] + line[i][1])
        count_y += abs(point[1] - line[i][1]) * 0.5 * abs(point[0] + line[i][0])
    return count_x, count_y


def loss_between_two_curves(pred: Tensor, gt: Tensor):
    pred_x, pred_y = _get_area(pred)
    gt_x, gt_y = _get_area(gt)

    return abs(pred_x - gt_x) + abs(pred_y - gt_y)


def check_empty_map():
    for log in os.scandir(os.path.join("./data/av2/map_data")):
        for data_file in os.scandir(os.path.join(log.path)):
            map_dict = mmengine.load(data_file.path)
            for line, label in zip(map_dict["lines"], map_dict["labels"]):
                if int(label) == 2:
                    print(data_file.path)
                    return


if __name__ == "__main__":
    with open("results/bevtr_15epoch_20231029_125355/epoch_15.pth", "rb") as f:
        params = pickle.load(f)
        print(params)

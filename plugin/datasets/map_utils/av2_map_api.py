from pathlib import Path

import torch



def _global2ego(polylines, ego2global):
    size_list = [len(item) for item in polylines]
    polylines = torch.cat(polylines, dim=0).float()
    polylines = torch.cat([polylines, torch.ones_like(polylines[:, 0:1])], dim=-1).unsqueeze(-1)

    polylines = torch.inverse(ego2global.cuda()) @ polylines.cuda()

    polylines = polylines.cpu().squeeze(-1)[:, 0:3]
    polylines = torch.split(polylines, size_list)
    return polylines


def _remove_redundant(polylines, label_list):
    sum_values = set()
    new_polylines = []
    new_labels = []
    for i, polyline in enumerate(polylines):
        sum_value = torch.sum(polyline).item()
        if sum_value not in sum_values:
            new_polylines.append(polyline)
            sum_values.add(sum_value)
            new_labels.append(label_list[i])
        else:
            flag = True
            for j, line in enumerate(polylines):
                if i != j and len(polyline) == len(line) and torch.max(polyline - line) < 0.01:
                    flag = False
            if flag:
                new_polylines.append(polyline)
                new_labels.append(label_list[i])
    return new_polylines, new_labels


def _cat_line(polylines, label_list):
    """把车道边界线和人行横道线满足条件（该线前后都有唯一的相同类型的线端点相同）的拼接起来"""
    poly_dict = {i: item for i, item in enumerate(polylines)}

    def _cat_single_line(idx):
        count_start = 0
        count_end = 0
        temp1 = -1
        temp2 = -1
        for key, value in poly_dict.items():
            if key != idx and torch.equal(poly_dict[idx][0], value[-1]):
                count_start += 1
                temp1 = key
        for key, value in poly_dict.items():
            if key != idx and torch.equal(poly_dict[idx][-1], value[0]):
                count_end += 1
                temp2 = key
        if count_start == 1 and count_end == 1 and temp1 != temp2:
            if label_list[idx] == label_list[temp1] and label_list[idx] == label_list[temp2]:
                poly_dict[idx] = torch.cat((poly_dict[temp1], poly_dict[idx][1:], poly_dict[temp2][1:]), dim=0)
                poly_dict.pop(temp1)
                poly_dict.pop(temp2)
                return True
        return False

    for i, polyline in enumerate(polylines):
        if i in poly_dict.keys():
            if label_list[i] < 2:
                while _cat_single_line(i):
                    pass
    return [value for _, value in poly_dict.items()], [label_list[idx] for idx, _ in poly_dict.items()]


def _check_point_in_range(point, map_range):
    return (map_range[0] < point[0] < map_range[2]) and (map_range[1] < point[1] < map_range[3])


def _cut_line(polylines, labels, map_range):
    def x_in_range(value):
        if value > map_range[2]:
            value = map_range[2]
        if value < map_range[0]:
            value = map_range[0]
        return value

    def y_in_range(value):
        if value > map_range[3]:
            value = map_range[3]
        if value < map_range[1]:
            value = map_range[1]
        return value

    # point1 在圈内，point2在圈外，返回的point3在圈上
    def adjust_point(point1: torch.Tensor, point2: torch.Tensor):
        k = (point2[1] - point1[1]) / (point2[0] - point1[0])
        point3 = point2.clone()
        if point3[0] > map_range[2] or point3[0] < map_range[0]:
            point3[0] = x_in_range(point3[0])
            point3[1] = (point3[0] - point1[0]) * k + point1[1]
        if point3[1] > map_range[3] or point3[1] < map_range[1]:
            point3[1] = y_in_range(point3[1])
            point3[0] = (point3[1] - point1[1]) / k + point1[0]
        return point3

    new_lines = []
    new_labels = []
    for line, label in zip(polylines, labels):
        temp = []
        for i, point in enumerate(line):
            if _check_point_in_range(point, map_range):
                if i > 0 and not _check_point_in_range(line[i - 1], map_range):
                    temp.append(adjust_point(point, line[i - 1]))
                temp.append(point)
            else:
                if len(temp) > 0:
                    temp.append(adjust_point(temp[-1], point))
                    new_lines.append(torch.stack(temp))
                    new_labels.append(label)
                    temp = []
        if len(temp) > 0:
            new_lines.append(torch.stack(temp))
            new_labels.append(label)
    return new_lines, new_labels


def get_vector_map(av_map, ego2global, class2label, map_range):
    polylines = []
    label_list = []
    if 'divider' in class2label.keys():
        for line_segment in av_map.get_scenario_lane_segments():
            if not line_segment.is_intersection:
                polylines.append(torch.from_numpy(line_segment.left_lane_boundary.xyz))
                label_list.append(class2label['divider'])
                if line_segment.right_neighbor_id is None:
                    polylines.append(torch.from_numpy(line_segment.right_lane_boundary.xyz))
                    label_list.append(class2label['divider'])
    if 'ped_crossing' in class2label.keys():
        for ped_cross in av_map.get_scenario_ped_crossings():
            polylines.append(torch.from_numpy(ped_cross.edge1.xyz))
            label_list.append(class2label['ped_crossing'])
            polylines.append(torch.from_numpy(ped_cross.edge2.xyz))
            label_list.append(class2label['ped_crossing'])
    if 'drivable_boundary' in class2label.keys():
        for dri_area in av_map.get_scenario_vector_drivable_areas():
            polylines.append(torch.from_numpy(dri_area.xyz))
            label_list.append(class2label['drivable_boundary'])

    polylines = _global2ego(polylines, ego2global)

    polylines, label_list = _cut_line(polylines, label_list, map_range)
    polylines, label_list = _remove_redundant(polylines, label_list)
    polylines, label_list = _cat_line(polylines, label_list)

    return polylines, torch.tensor(label_list, dtype=torch.int32)
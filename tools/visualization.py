import copy
import os
import time

import cv2
import mpmath
import torch
from mmengine.analysis import get_model_complexity_info
from mmengine.runner import Runner
from mmengine.config import Config
import mmengine
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mmdet3d.registry import MODELS


def _show_vectors(target: dict):
    target["lines"][..., 0] = target["lines"][..., 0] * 60 - 30
    target["lines"][..., 1] = target["lines"][..., 1] * 30 - 15
    for line, label in zip(target["lines"], target["labels"]):
        line = line.cpu().numpy()
        # line = line * 30 - 15
        color = "green"
        if label == 0:
            color = "red"
        if label == 1:
            color = "teal"
        axes2.plot(line[:, 1], line[:, 0], color=color)
        # for i, point in enumerate(line):
        #     axes1.text(point[0], point[1], str(i))


def show_vectors_v1(lines, labels, key_points=None, key_points_label=None):
    lines = lines.instance_list
    # lines = lines.shift_fixed_num_sampled_points_v2
    for line, label in zip(lines, labels):
        # line = line.cpu().numpy()
        line = np.array(line.coords)
        color = "green"
        if label == 1:
            color = "red"
        if label == 0:
            color = "teal"
        axes1.plot(line[:, 1], line[:, 0], color=color)
        # if label == 1:
        # for i, point in enumerate(line):
        #     axes1.text(point[0], point[1], str(i))

    # for point, label in zip(key_points, key_points_label):
    #     point = point.cpu().numpy()
    #     color = "green"
    #     if label == 1:
    #         color = "red"
    #     if label == 0:
    #         color = "teal"
    #     axes2.plot(point[0], point[1], 'o', color=color)

def show_vectors_v2(axes, lines, labels, key_points=None, key_points_label=None):
    axes.clear()
    axes.grid(False)
    axes.set_xlim(-15, 15)
    axes.set_ylim(-30, 30)
    axes.set_aspect("equal")
    lines = lines.instance_list
    # lines = lines.shift_fixed_num_sampled_points_v2
    for line, label in zip(lines, labels):
        # line = line.cpu().numpy()
        line = np.array(line.coords)
        color = "green"
        if label == 1:
            color = "red"
        if label == 0:
            color = "teal"
        axes.plot(line[:, 0], line[:, 1], color=color)
        # if label == 1:
        # for i, point in enumerate(line):
        #     axes1.text(point[0], point[1], str(i))
    # axes2.clear()
    # axes2.grid(False)
    # axes2.set_xlim(-30, 30)
    # axes2.set_ylim(-60, 60)
    # axes2.set_aspect("equal")
    # for point, label in zip(key_points, key_points_label):
    #     point = point.cpu().numpy()
    #     color = "green"
    #     if label == 1:
    #         color = "red"
    #     if label == 0:
    #         color = "teal"
    #     axes2.plot(point[0], point[1], 'o', color=color)


def _show_outputs(axes, output):
    axes.grid(False)
    axes.axis('off')
    axes.set_xlim(-15, 15)
    axes.set_ylim(-30, 30)
    axes.set_aspect("equal")
    scores = output.pred_instances_3d.scores
    labels = output.pred_instances_3d.labels
    lines = output.pred_instances_3d.pts
    # key_scores = output.pred_instances.key_scores
    # key_labels = output.pred_instances.key_labels
    # key_pts = output.pred_instances.key_pts
    credit = 0.4

    for line, label, score in zip(lines, labels, scores):
        line = line.cpu().numpy()
        if label == 0:
            color = "teal"
        elif label == 1:
            color = "r"
        else:
            color = "green"
        if score > credit:
            axes.plot(line[:, 0], line[:, 1], color=color, linewidth=2)
            axes.plot(line[:, 0], line[:, 1], 'o', color=color, markersize=3)
    
    # for point, label, score in zip(key_pts, key_labels, key_scores):
    #     point = point.cpu().numpy()
    #     if label == 0:
    #         color = "teal"
    #     elif label == 1:
    #         color = "r"
    #     else:
    #         color = "green"
    #     if score > credit:
    #         axes.plot(point[0], point[1], 'o', color=color)


def _show_point_outputs(pred_points, scores):
    pred_points[..., 1] = pred_points[..., 1] * 60 - 30
    pred_points[..., 0] = pred_points[..., 0] * 30 - 15
    credit = 0.1
    scores = scores.sigmoid()
    for point, score in zip(pred_points, scores):
        point = point.cpu().numpy()
        if torch.max(score) > credit:
            axes2.scatter([point[0]], [point[1]])


def _show_map(da_map):
    axes2.set_xlim(0, 240)
    axes2.set_ylim(-60, 180)
    # da_map[:50, :50] = 1
    axes2.imshow(da_map.numpy())


def print_point(bbox, labels):
    points = []
    point_labels = []
    lines = bbox.fixed_num_sampled_points
    for i, line in enumerate(lines):
        for point in line:
            points.append(point)
            point_labels.append(i)
    points = torch.stack(points)
    point_labels = torch.tensor(point_labels)
    values, indices = torch.sort(points[:, 0])
    print(point_labels[indices])


def show_vectorized_local_map():
    cfg = Config.fromfile("config/pointmapnet-large-range.py")
    runner = Runner.from_cfg(cfg)
    dataloader = runner.train_dataloader
    for data in dataloader:
        pass
        # for i, (bbox, label, data_sample, points, points_label) in enumerate(
        #     zip(
        #         data["inputs"]["gt_bboxes_3d"],
        #         data["inputs"]["gt_labels_3d"],
        #         data["data_samples"],
        #         data['inputs']['raw_points'],
        #         data['inputs']['raw_points_label']
        #     )
        # ):
        #     show_vectors_v2(axes1, bbox, label, points, points_label)
        #     print(data_sample.sample_idx)
        #     plt.savefig(f"results/{data_sample.sample_idx}.png")
        # return


def show_mask_map():
    cfg = Config.fromfile("config/petrv2.py")
    runner = Runner.from_cfg(cfg)
    dataloader = runner.train_dataloader
    for data in dataloader:
        target = data["data_samples"][0].target
        da_map = data["data_samples"][0].maps
        print(data["data_samples"][0].sample_idx)
        _show_vectors(target)
        _show_map(da_map[2])
        plt.savefig("result.png")
        # break


def compare_gt():
    import os
    gt1 = dict()
    gt2 = dict()
    g = os.walk("/data/nuscenes/temp1")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            token = file_name.split('.')[0]
            gt1[token] = mmengine.load(os.path.join(path, file_name))
    g = os.walk("/data/nuscenes/temp3")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            token = file_name.split('.')[0]
            gt2[token] = mmengine.load(os.path.join(path, file_name))
    for key in gt1.keys():
        assert torch.equal(gt1[key]['gt_labels_3d'], gt2[key]['gt_labels_3d'])


def show_model_result(model_path):
    cfg = Config.fromfile("config/pointmapnet-av2.py")
    runner = Runner.from_cfg(cfg)
    dataloader = runner.train_dataloader
    model = runner.model
    mmengine.runner.load_checkpoint(model, model_path)
    model.cuda()
    model.eval()

    for data in dataloader:
        with torch.no_grad():
            outputs = model.val_step(data)
            i = 0
            for output, bboxes, labels, key_points, key_points_label in zip(
                outputs, data["inputs"]["gt_bboxes_3d"], data["inputs"]["gt_labels_3d"],
                data['inputs']['raw_points'], data['inputs']['raw_points_label']
            ):
                show_vectors_v2(axes1, bboxes, labels)
                _show_outputs(axes2, output)
                # print(data['data_samples'][i].sample_idx)
                
                plt.savefig(f"{data['data_samples'][i].sample_idx}.png")
                axes1.clear()
                axes2.clear()
                axes1.set_xlim(-35, 35)
                axes1.set_ylim(-35, 35)
                axes2.set_xlim(-35, 35)
                axes2.set_ylim(-35, 35)
                axes1.grid(False)
                axes2.grid(False)
                i += 1
        break

def show_picture_and_result(fig, output1, output2, bboxes, labels, data_sample):
    import matplotlib.image as mpimg
    # fig.cgglf()
    axes1 = fig.add_subplot(2, 6, 1)
    axes2 = fig.add_subplot(2, 6, 2)
    axes3 = fig.add_subplot(2, 6, 3)
    axes4 = fig.add_subplot(2, 6, 7)
    axes5 = fig.add_subplot(2, 6, 8)
    axes6 = fig.add_subplot(2, 6, 9)
    img_axes = [axes1, axes2, axes3, axes4, axes5, axes6]

    axes7 = fig.add_subplot(1, 6, 4)
    axes8 = fig.add_subplot(1, 6, 5)
    axes9 = fig.add_subplot(1, 6, 6)

    file_paths = data_sample.img_filenames
    new_file_paths = [file_paths[2], file_paths[0], file_paths[1], file_paths[4], file_paths[3], file_paths[5]]
    for file_path, axes in zip(new_file_paths, img_axes):
        img = mpimg.imread(file_path)
        axes.imshow(img)
        axes.axis('off')
    _show_outputs(axes7, output1)
    _show_outputs(axes8, output2)
    show_vectors_v2(axes9, bboxes, labels)


def visualization(config, model_path):
    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    dataloader = runner.test_dataloader
    model1 = runner.model
    mmengine.runner.load_checkpoint(model1, model_path)
    model1.cuda()
    model1.eval()
    fig = plt.figure(figsize=(30, 7))
    cfg = Config.fromfile("config/maptr.py")
    runner = Runner.from_cfg(cfg)
    model2 = runner.model
    mmengine.runner.load_checkpoint(model2, "ckpts/maptr.pth")
    model2.cuda()
    model2.eval()
    for data in dataloader:
        with torch.no_grad():
            outputs1 = model1.val_step(copy.deepcopy(data))
            outputs2 = model2.val_step(data)
            i = 0
            for output1, output2, bboxes, labels, key_points, key_points_label, data_sample in zip(
                    outputs1, outputs2, data["inputs"]["gt_bboxes_3d"], data["inputs"]["gt_labels_3d"],
                    data['inputs']['raw_points'], data['inputs']['raw_points_label'], data['data_samples'],
            ):
                show_picture_and_result(fig, output1, output2, bboxes, labels, data_sample)
                # print(data['data_samples'][i].sample_idx)
                plt.tight_layout()
                plt.savefig(f"visualization_result/{data['data_samples'][i].sample_idx}.png")
                i += 1

def show_picture_and_result1(fig, output1, data_sample):
    import matplotlib.image as mpimg
    from matplotlib import gridspec
    fig.clf()
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1])
    file_paths = data_sample.img_filenames
    new_file_paths = [file_paths[2], file_paths[0], file_paths[1], file_paths[4], file_paths[3], file_paths[5]]

    row_1_list = []
    for cam in new_file_paths[:3]:
        cam_img = mpimg.imread(cam)
        row_1_list.append(cam_img)
    row_2_list = []
    for i, cam in enumerate(new_file_paths[3:]):
        cam_img = cv2.imread(cam)
        cam_img = cv2.flip(cam_img, 1)
        row_2_list.append(cam_img)
    row_1_img = cv2.hconcat(row_1_list)
    row_2_img = cv2.hconcat(row_2_list)
    cams_img = cv2.vconcat([row_1_img, row_2_img])
    ax1 = plt.subplot(gs[0])
    ax1.imshow(cams_img)
    ax1.axis('off')
    ax2 = plt.subplot(gs[1])
    _show_outputs(ax2, output1)
    # plt.tight_layout()



def visualization1(config, model_path):
    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    dataloader = runner.test_dataloader
    model1 = runner.model
    mmengine.runner.load_checkpoint(model1, model_path)
    model1.cuda()
    model1.eval()
    fig = plt.figure(figsize=(25, 8))
    j = 0
    for data in dataloader:
        with torch.no_grad():
            j += 1
            outputs1 = model1.val_step(data)
            # i = 0
            # for output1, data_sample in zip(outputs1, data['data_samples']):
            show_picture_and_result1(fig, outputs1[0], data['data_samples'][0])
            # print(data['data_samples'][i].sample_idx)
            # plt.tight_layout()
            # plt.savefig(f"video_frame/{data['data_samples'][i].sample_idx}.png")
            plt.savefig(f"video_frame/{j}.png")
                # i += 1
            if j == 600:
                break


def generate_video():
    image_folder = 'video_frame'  # 图像文件夹路径
    video_name = 'output_video.mp4'  # 视频文件名
    fps = 10  # 输出视频的帧率

    # 获取图像列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda name: int(name.split(".")[0]))  # 如果需要可以对图片进行排序

    # 设置视频编码和尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 创建视频写入对象
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

    # 将图片帧写入视频
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # 释放VideoWriter对象
    video.release()


def show_image_depth(config):
    from PIL import Image
    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    dataloader = runner.train_dataloader
    fig = plt.figure(figsize=(8, 25))
    for data in dataloader:
        with torch.no_grad():
            file_paths = data['data_samples'][0].img_filenames
            image_depths = data['data_samples'][0].image_depths[0]
            
            for i, file_path in enumerate(file_paths):
                # 读取图像
                image = Image.open(file_path)
                
                # 获取深度信息并转换为可视化格式
                depth = image_depths[i].squeeze(-1)
                depth_visual = depth.cpu().numpy()  # 转换到CPU并获取numpy数组
                depth_visual = (depth_visual - depth_visual.min()) / (depth_visual.max() - depth_visual.min())  # 归一化
                depth_visual = plt.cm.viridis(depth_visual)  # 应用色彩映射
                
                # 创建子图
                ax = fig.add_subplot(6, 2, 2*i + 1)  # 左侧为图像
                ax.imshow(image)
                ax.axis('off')  # 关闭坐标轴
                
                ax = fig.add_subplot(6, 2, 2*i + 2)  # 右侧为深度信息
                ax.imshow(depth_visual)
                ax.axis('off')  # 关闭坐标轴

            plt.savefig(f"image_depths.png")
            break


def show_bev_points(config):
    from PIL import Image
    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    dataloader = runner.train_dataloader
    fig = plt.figure(figsize=(30, 30))
    for data in dataloader:
        with torch.no_grad():
            file_paths = data['data_samples'][0].img_filenames
            bev_points = data['data_samples'][0].bev_points[0]

            for i, file_path in enumerate(file_paths):
                # 读取图像
                image = Image.open(file_path)

                # 获取深度信息并转换为可视化格式
                depth = bev_points[i].squeeze(-1)
                depth_visual = depth.cpu().numpy()  # 转换到CPU并获取numpy数组
                depth_visual = (depth_visual - depth_visual.min()) / (depth_visual.max() - depth_visual.min())  # 归一化
                depth_visual = plt.cm.viridis(depth_visual)  # 应用色彩映射

                # 创建子图
                ax = fig.add_subplot(6, 2, 2*i + 1)  # 左侧为图像
                ax.imshow(image)
                ax.axis('off')  # 关闭坐标轴

                ax = fig.add_subplot(6, 2, 2*i + 2)  # 右侧为深度信息
                ax.imshow(depth_visual)
                ax.axis('off')  # 关闭坐标轴

            plt.savefig(f"image_depths.png")
            break


def point_sampling(reference_points, img_metas):
    pc_range = [-15.0, -30.0, -3.0, 15.0, 30.0, 3.0]
    reference_points = reference_points.float()

    lidar2img = np.asarray(img_metas["lidar2img"])
    lidar2img = reference_points.new_tensor(lidar2img)  # (N, 4, 4)
    reference_points = reference_points.clone()

    reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    )

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1
    )

    D, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(0)

    reference_points = (
        reference_points.view(D, 1, num_query, 4)
        .repeat(1, num_cam, 1, 1)
        .unsqueeze(-1)
    )

    lidar2img = lidar2img.view(1, num_cam, 1, 4, 4).repeat(
        D, 1, num_query, 1, 1
    )

    reference_points_cam = (
        torch.matmul(lidar2img, reference_points).squeeze(-1).float()
    )
    eps = 1e-5

    bev_mask = reference_points_cam[..., 2:3] > eps
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3],
        torch.ones_like(reference_points_cam[..., 2:3]) * eps,
    )

    reference_points_cam[..., 0] /= img_metas["pad_shape"][0][1]
    reference_points_cam[..., 1] /= img_metas["pad_shape"][0][0]

    bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
    )
    return reference_points_cam, bev_mask


def add_height(ref_points):
    # zs = ref_points.new_tensor([0.1, 0.15, 0.2, 0.25]).view(-1, 1, 1)
    zs = ref_points.new_tensor([0.2]).view(-1, 1, 1)
    num_query, dim = ref_points.shape
    ref_points = ref_points.unsqueeze(0).repeat(1, 1, 1)
    zs = zs.repeat(1, num_query, 1)
    ref_points = torch.cat([ref_points, zs], dim=-1)
    return ref_points


def show_picture_and_proj_points(fig, output, data_sample, level_focus_point):
    fig.clf()
    file_paths = data_sample.img_filenames
    image = cv2.imread(file_paths[0])

    level_focus_point = torch.stack([level_focus_point[0], level_focus_point[1], level_focus_point[3]], dim=0)

    # 创建一个新的子图，1 行 1 列，占据整个图形区域
    ax = fig.add_subplot(1, 1, 1)

    # 设置子图的大小，使其能容纳三张图像
    ax.set_xlim(0, 1600 * 3)
    ax.set_ylim(0, 900)

    # 依次绘制三张图像并叠加投影点
    for i, points in enumerate(level_focus_point):
        img = cv2.imread(file_paths[0])
        ax.imshow(img, extent=[1600 * i, 1600 * (i + 1), 0, 900])

        points = add_height(points)
        cam_points, masks = point_sampling(points, data_sample.metainfo)
        cam_points = cam_points.transpose(1, 0).flatten(1, 2)[0]
        masks = masks.transpose(1, 0).flatten(1, 2)[0]

        valid_points = cam_points[masks.squeeze() == 1]
        valid_points = valid_points.cpu().numpy()
        valid_points[:, 0] = valid_points[:, 0] * 1600 + 1600 * i
        valid_points[:, 1] = (1 - valid_points[:, 1]) * 900

        ax.scatter(valid_points[:, 0], valid_points[:, 1], c='red', s=10)

    ax.axis('off')
    # ax1.axis('off')


def show_focus_points_on_image(config, model_path):
    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    dataloader = runner.test_dataloader
    model = runner.model
    mmengine.runner.load_checkpoint(model, model_path)
    model.cuda()
    model.eval()
    fig = plt.figure(figsize=(48, 9))
    j = 0
    for data in dataloader:
        with torch.no_grad():
            j += 1
            model_data = model.data_preprocessor(data, False)
            outputs, output_dict = model.predict(model_data['inputs'], model_data['data_samples'])
            k = 0
            level_focus_points = output_dict['all_raw_points'].transpose(1, 0)
            level_focus_points = torch.cat([output_dict['init_ref_points'].unsqueeze(1), level_focus_points], dim=1)
            for output, data_sample, level_focus_point in zip(outputs, data['data_samples'], level_focus_points):
                show_picture_and_proj_points(fig, outputs[0], data_sample, level_focus_point)
                plt.savefig(f"{j}-{k}.png")
                k += 1
            # i = 0
            # for output1, data_sample in zip(outputs1, data['data_samples']):

            # print(data['data_samples'][i].sample_idx)
            # plt.tight_layout()
            # plt.savefig(f"video_frame/{data['data_samples'][i].sample_idx}.png")

                # i += 1
            if j == 20:
                break

def draw_3d_position_embedding():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 设置矩形面的尺寸
    length = 16
    width = 9

    # 生成矩形面上的点
    x = np.linspace(0, length, 16)
    z = np.linspace(0, width, 9)
    x, z = np.meshgrid(x, z)
    y = np.zeros_like(x)  # 假设矩形面在z=0的平面上
    point_in_space = np.array([2, -30, 3])
    # 创建3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(0, 16)
    ax.set_ylim3d(-30, 0)
    ax.set_zlim3d(0, 16)

    # 为每条射线生成点
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # 生成射线上的点
            ray_points = np.linspace(point_in_space, np.array([x[i, j], y[i, j], z[i, j]]), 24)

            # 绘制射线
            ax.plot(ray_points[10:, 0], ray_points[10:, 1], ray_points[10:, 2], 'o', markersize=2, color='#878BCE')

    # 设置图表的标题和坐标轴标签
    ax.set_axis_off()

    # 显示图形
    plt.show()

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    # color_list = list(mcolors.TABLEAU_COLORS.keys())
    # fig = plt.figure(figsize=(20, 10))
    # axes1 = fig.add_subplot(1, 2, 1)
    # axes2 = fig.add_subplot(1, 2, 2)
    # show_model_result("ckpts/pointmapnet-av2.pth")
    # show_vectorized_local_map()
    # # check_vector_length()
    # # show_mask_map()
    # # show_drivable_area()
    # visualization1("config/pointmapnetv1.py", "ckpts/pointmapnet.pth")
    # show_image_depth("config/pointmapnetv2.py")
    # show_bev_points("config/rcmapnet.py")
    # show_focus_points_on_image("config/pointmapnet.py", "work_dirs/pointmapnet/epoch_25.pth")
    config = 'config/pointmapnet.py'
    cfg = Config.fromfile(config)
    runner = Runner.from_cfg(cfg)
    model = runner.model
    mmengine.runner.load_checkpoint(model, 'ckpts/pointmapnet.pth')

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number of trainable parameters:", trainable_params)
    print("Number of non - trainable parameters:", non_trainable_params)

    backbone = runner.model.img_backbone
    neck = runner.model.img_neck
    head = runner.model.head

    model = backbone
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number of trainable parameters:", trainable_params)
    print("Number of non - trainable parameters:", non_trainable_params)

    model = neck
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number of trainable parameters:", trainable_params)
    print("Number of non - trainable parameters:", non_trainable_params)

    model = head
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number of trainable parameters:", trainable_params)
    print("Number of non - trainable parameters:", non_trainable_params)
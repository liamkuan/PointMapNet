import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.get_cachedir())

# 定义五个点的坐标和标注
# points = [(1.9, 23.0, 'HDMapNet[12]'), (6.2, 40.9, 'VectorMapNet[17]'), (24.2, 50.1, 'MapTR[14]'), (19.6, 61.5, 'MapTRv2[16]'), (19.9, 60.6, 'StreamMapNet[40]'), (30.3, 55.3, 'PointMapNet')]
# colors = ['#A9B8C6', '#C497B2', '#F3D266', '#96C37D', '#FF8884',  '#14517C']
# # 提取坐标和标注
# x = [point[0] for point in points]
# y = [point[1] for point in points]
# labels = [point[2] for point in points]
#
# # 生成散点图
# plt.scatter(x[:-1], y[:-1], c=colors[:-1], s=64)
# plt.scatter(x[-1:], y[-1:], marker='*', c='r', s=100)
# font_dict = {
#     'family': 'serif',     # 字体家族
#     'color': 'black',       # 字体颜色
#     'weight': 'bold',    # 字体粗细
#     'size': 12             # 字体大小
# }
# # 将字体家族设置为 Times New Roman
# font_dict['family'] = 'Times New Roman'
# # 应用字体属性到x轴标签
# plt.xlabel('FPS', fontdict=font_dict)
# # 应用字体属性到y轴标签
# plt.ylabel('nuScenes mAP', fontdict=font_dict)
#
# # 添加标注
# for i, label in enumerate(labels):
#     if i in [3, 4]:
#         continue
#     plt.text(x[i] + 0.7, y[i] - 0.7, label, fontdict=font_dict, ha='left')
#
# plt.text(16.6, 63.0, 'MapTRv2[16]', fontdict=font_dict, ha='left')
# plt.text(12.0, 57.6, 'StreamMapNet[40]', fontdict=font_dict, ha='left')
#
# # 设置图表标题和坐标轴标签
# plt.xlim(0, 39)
# plt.ylim(20, 66)
# grid_color = (0.7, 0.7, 0.7, 0.3)
# plt.grid(color=grid_color, linestyle='dashed')
#
# font_dict['color'] = 'red'
# plt.text(33.5, 21.5, 'Faster', fontdict=font_dict, ha='left')
# plt.text(1., 56, 'Stronger', fontdict=font_dict, ha='left', rotation=90)
# # 显示图表
# # plt.show()
# plt.savefig("balance1.png")


points = [(14.37, 40.9, 'VectorMapNet[17]'), (10.04, 50.3, 'MapTR-tiny[14]'), (18.95, 61.5, 'MapTRv2[16]'), (3.12, 32.9, 'MapTR-nano[14]'), (17.24, 60.6, 'StreamMapNet[40]'), (6.2, 55.3, 'PointMapNet')]
colors = ['#C497B2', '#F3D266', '#A9B8C6', '#96C37D', '#FF8884',  '#14517C']
# 提取坐标和标注
x = [point[0] for point in points]
y = [point[1] for point in points]
labels = [point[2] for point in points]

# 生成散点图
plt.scatter(x[:-1], y[:-1], c=colors[:-1], s=64)
plt.scatter(x[-1:], y[-1:], marker='*', c='r', s=100)
font_dict = {
    'family': 'serif',     # 字体家族
    'color': 'black',       # 字体颜色
    'weight': 'bold',    # 字体粗细
    'size': 12             # 字体大小
}
# 将字体家族设置为 Times New Roman
font_dict['family'] = 'Times New Roman'
# 应用字体属性到x轴标签
plt.xlabel('training video memory usage(GB)', fontdict=font_dict)
# 应用字体属性到y轴标签
plt.ylabel('nuScenes mAP', fontdict=font_dict)

# 添加标注
for i, label in enumerate(labels):
    if i in [2, 4]:
        continue
    plt.text(x[i] + 0.7, y[i] - 0.7, label, fontdict=font_dict, ha='left')

plt.text(18.5, 63.0, 'MapTRv2[16]', fontdict=font_dict, ha='left')
plt.text(16.9, 57.6, 'StreamMapNet[40]', fontdict=font_dict, ha='left')

# 设置图表标题和坐标轴标签
plt.xlim(0, 30)
plt.ylim(20, 66)
grid_color = (0.7, 0.7, 0.7, 0.3)
plt.grid(color=grid_color, linestyle='dashed')

font_dict['color'] = 'red'
plt.text(25, 21.5, 'Low Cost', fontdict=font_dict, ha='left')
plt.text(1., 56, 'Stronger', fontdict=font_dict, ha='left', rotation=90)
# 显示图表
# plt.show()
plt.savefig('balance2.png')

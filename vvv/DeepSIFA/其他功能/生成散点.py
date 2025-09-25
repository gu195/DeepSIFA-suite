import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
source_dir = r'D:\DeepSIFA_main\data\MLKL\test\v4\归一化插值后npz_1024_高斯平滑1'
png_dir = os.path.join(source_dir, 'png')
# 如果目录存在，先删除
if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
# 创建新的 png 目录
os.makedirs(png_dir, exist_ok=True)


# 获取原始数据目录中的所有npz文件
npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]


def process_file(filename):
    file_path = os.path.join(source_dir, filename)
    npz_data = np.load(file_path)
    data = npz_data['data']

    # 获取 'start' 和 'end' 数据
    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']

    # 调整 y 轴的最小值为 -200
    min_value = data.min()
    shift_value = 0 - min_value  # 差值
    data = data + shift_value  # 为每个点加上差值

    # 创建 x 轴坐标
    x = range(len(data))
    plt.figure(figsize=(18, 6))  # 设置图形大小

    # 绘制曲线，颜色为蓝色
    plt.plot(x, data, label='Data Line', linewidth=2.5, color='#1F77B4')  # 蓝色

    # 设置 x 轴
    plt.xlabel('Time(s)', fontsize=24, labelpad=10)  # 设置 x 轴标签和字体大小
    plt.xticks([], [])  # 隐藏 x 轴刻度

    # 设置 y 轴
    plt.ylabel('Intensity', fontsize=24, labelpad=10)  # 设置 y 轴标签和字体大小
    plt.yticks(fontsize=20)  # 调整 y 轴刻度字体大小

    # 获取当前轴对象
    ax = plt.gca()

    # 设置边框的粗细
    line_width = 2 # 统一的线条粗细

    # 启用并设置上边框和右边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 自定义边框的颜色和线宽
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(line_width)

    # 在指定范围内标出红色区域
    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        plt.axvspan(start[0], end[0], color='red', alpha=0.3)

    # # 保存图像
    # os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    save_path = os.path.join(source_dir, 'png', filename.replace('.npz', '.png'))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=500)
    plt.clf()  # 清除当前图形，以便下一个文件可以绘制新图



# 定义多进程处理
if __name__ == '__main__':
    num_workers = cpu_count()  # 获取CPU核心数
    print(f'Using {num_workers} workers for parallel processing.')

    # 使用Pool进行多进程处理
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    print("All PNGs have been generated.")







# # 把高斯滤波后和高斯滤波前的对比
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# from tqdm import tqdm

# source_dir1 = '/home/node01/linchen/data/alphaK10/V2/第三批/归一化插值后npz_1024'
# source_dir2 = '/home/node01/linchen/data/alphaK10/V2/第三批/归一化插值后npz_1024_高斯平滑1'
# source_dir3 = '/home/node01/linchen/data/alphaK10/V2/第三批/归一化插值后npz_1024_高斯平滑2'

# # 获取原始数据目录中的所有npz文件
# npz_files1 = [filename for filename in os.listdir(source_dir1) if filename.endswith('.npz')]
# npz_files2 = [filename for filename in os.listdir(source_dir2) if filename.endswith('.npz')]

# # 使用 tqdm 添加进度条
# for filename in tqdm(npz_files1, desc='Processing', unit='file'):
#     # 读取每个npz文件
#     file_path1 = os.path.join(source_dir1, filename)
#     npz_data1 = np.load(file_path1)
#     data1 = npz_data1['data']

#     file_path2 = os.path.join(source_dir2, filename)
#     npz_data2 = np.load(file_path2)
#     data2 = npz_data2['data']

#     file_path3 = os.path.join(source_dir3, filename)
#     npz_data3 = np.load(file_path3)
#     data3 = npz_data3['data']

#     if 'start' in npz_data1.keys() and 'end' in npz_data1.keys():
#         start = npz_data1['start']
#         end = npz_data1['end']
#     # 创建x轴坐标
#     x = range(len(data1))
#     plt.figure(figsize=(24, 6))  # 设置图形的大小为宽10英寸，高5英寸
    
#     # 绘制曲线和散点图
#     plt.plot(x, data1, 'k',     label='original data',      linewidth=0.5)  # 将线条宽度设置为1
#     plt.plot(x, data2, '--',    label='filtered, sigma=1',  linewidth=1)  # 将线条宽度设置为1
#     plt.plot(x, data3, ':',     label='filtered, sigma=2',  linewidth=1)  # 将线条宽度设置为1
    
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.title('{}'.format(filename))
#     plt.legend()  # 显示图例

#     # 设置横坐标刻度和栅格线
#     plt.xticks(np.arange(0, len(data1), step=25))
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)

#     if 'start' in npz_data1.keys() and 'end' in npz_data1.keys():
#         # 在指定范围内标出红色区域
#         plt.axvspan(start[0], end[0], color='red', alpha=0.3)

#     os.makedirs(os.path.join(source_dir1, 'png'), exist_ok=True)
#     save_path = os.path.join(source_dir1, 'png', filename)
#     save_path = save_path.replace('.npz', '.png')
#     plt.savefig(save_path, dpi=400)
#     plt.clf()  # 清除当前图形，以便下一个文件可以绘制新图

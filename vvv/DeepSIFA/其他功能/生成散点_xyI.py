# 可视化x坐标
import matplotlib
matplotlib.use('Agg')  # 切换到无 GUI 的后端
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import shutil
import time
# source_dir = r'D:\DeepSIFA_main\data\wt0117测试\I\0216-v9-circle-g6-200-1.2-高斯滤波\row_data'
source_dir = r'D:\DeepSIFA_main\data\wt0117测试\I\0314-v1-circle-g6-s3-f3-200-1.2-asy\row_data'
png_dir = os.path.join(source_dir, 'png_论文')

if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
    time.sleep(2)  # 等待1秒，确保完全删除
os.makedirs(png_dir, exist_ok=True)

# 获取原始数据目录中的所有npz文件
npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.txt')]


def process_file(filename):
    file_path = os.path.join(source_dir, filename)
    df = pd.read_csv(file_path, delimiter=',', header=1)  

    # 获取第二列和第三列数据
    intensity = df.iloc[:, 8].astype(float)
    column_2 = df.iloc[:, 3].astype(float)  # 强制转换为 float
    column_3 = df.iloc[:, 4].astype(float)  # 强制转换为 float
    data_I = intensity
    data_x = column_2 - 1
    data_y = column_3 - 1

    # 创建 x 轴坐标
    x = range(len(data_x))

    # 创建 3 行 1 列的子图
    fig, axes = plt.subplots(3, 1, figsize=(26, 18))  # (行, 列)，这里是 2 行 1 列
    # fig, axes = plt.subplots(3, 1, figsize=(15, 16))  # (行, 列)，这里是 2 行 1 列
    # fig, axes = plt.subplots(3, 1, figsize=(24, 18))  # (行, 列)，这里是 2 行 1 列

    # 绘制第一个子图 (data_x)
    axes[0].plot(x, data_x, label='Data X', linewidth=2.5, color='#2CA02C')
    axes[0].set_xlabel('Time(s)', fontsize=38, labelpad=10)
    axes[0].set_ylabel('X-coordinate', fontsize=38, labelpad=10)
    axes[0].tick_params(axis='y', labelsize=25)
    axes[0].set_xticks([])  # 隐藏 x 轴刻度

    # 绘制第二个子图 (data_y)
    axes[1].plot(x, data_y, label='Data Y', linewidth=2.5, color='#FF7F0E')
    axes[1].set_xlabel('Time(s)', fontsize=38, labelpad=10)
    axes[1].set_ylabel('Y-coordinate', fontsize=38, labelpad=10)
    axes[1].tick_params(axis='y', labelsize=25)
    axes[1].set_xticks([])  # 隐藏 x 轴刻度

    # 绘制第三个子图 (I)
    axes[2].plot(x, data_I, label='Intensity', linewidth=2.5, color='#1F77B4')
    axes[2].set_xlabel('Time(s)', fontsize=38, labelpad=10)
    axes[2].set_ylabel('Intensity', fontsize=38, labelpad=10)
    axes[2].tick_params(axis='y', labelsize=25)
    axes[2].set_xticks([])  # 隐藏 x 轴刻度

    # 统一设置边框样式
    for ax in axes:
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(2)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(png_dir, filename.replace('.txt', '.png'))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()  # 彻底释放资源


# 定义多进程处理
if __name__ == '__main__':
    # num_workers = cpu_count()  # 获取CPU核心数
    # print(f'Using {num_workers} workers for parallel processing.')
    # # 使用Pool进行多进程处理
    # with Pool(processes=num_workers) as pool:
    #     list(tqdm(pool.imap(process_file, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    # print("All PNGs have been generated.")

    # 逐个处理文件，而不是使用多进程
    for filename in tqdm(npz_files, total=len(npz_files), desc='Processing PNGs', unit='file'):
        process_file(filename)  # 直接调用处理函数
    print("All PNGs have been generated.")






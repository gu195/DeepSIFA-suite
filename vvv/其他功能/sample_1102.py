# # -*- coding: utf-8 -*-
# """
# Created on Thu Oct 17 09:48:08 2024

# @author: zhou-
# 1-安装matlabengine, 如果已经安装，先卸载
# 打开cd "matlabroot\extern\engines\python"
# python -m pip install .

# 2-运行如下代码

# 2024-11-02: 加入u-track，能够直接调用，详见utrackInterface.m
#             https://github.com/DanuserLab/u-track
# """
import os, argparse
import matlab.engine
import shutil
import csv
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用无界面 Agg 后端
import pandas as pd
import math
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import re


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=r'D:\DeepSIFA_main\data\测试\S55C_label_alex55_1.5nM.tif')
    parse_config = parser.parse_args()
    return parse_config


def delete_directories():
    directories_to_delete = [
        os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME),
        os.path.join('D:\\', 'DeepSIFA_main', 'data', 'MLKL','test', f'v{NUM}')

    ]
    
    for dir_path in directories_to_delete:
        if os.path.exists(dir_path):
            # 计算目录深度
            depth = len(os.path.normpath(dir_path).split(os.sep))
            if depth >= 5:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                # print("0 目录删除完毕")
            else:
                print("0 未达到8级目录深度，不执行删除操作。")
        else:
            print(f"0 目录不存在: {dir_path}")


def analyze_image_with_matlab(filename):
    # 启动MATLAB引擎
    eng = matlab.engine.start_matlab()
    # 设置路径
    file_basename = os.path.basename(filename).replace('.tif','')
    workdir = os.path.join(r'D:\DeepSIFA_main\data\测试', file_basename)
    # workdir = r'D:\DeepSIFA_main\data\测试\S55C_label_alex55_1.5nM'  # 数据输出目录
    m_source_dir = r'D:\DeepSIFA_main\CreateTrace'  # MATLAB代码文件夹

    # 定义输入参数字典
    input_parameters = {
        'spotTrackingRadius': 3,  # 亮点最大跳跃距离，默认3px
        'threshold': 2,  # 阈值, 默认2
        'gaussFitWidth': 3,  # 高斯拟合宽度控制, 默认3px
        'frameLength': 30,  # 光斑最小持续的帧数, 默认20
        'frameGap': 2,  # 描述光斑可能不连续的最大帧数, 默认0
        'trackMethod': 'default',  # 轨迹跟踪方法
        'outputIntegralIntensity': 1,  # 是否计算总强度, 默认1
        'frameStart': 1,  # 分析的起始帧
        'frameEnd': 'inf',  # 分析的终止帧
        'utrackMotionType': 0  # u-track运动模式
    }

    # # 检查路径是否存在
    # if not os.path.exists(workdir):
    #     print(f"工作目录不存在: {workdir}")
    # if not os.path.exists(filename):
    #     print(f"图片文件不存在: {filename}")
    # if not os.path.exists(m_source_dir):
    #     print(f"MATLAB代码目录不存在: {m_source_dir}")

    # 执行MATLAB分析
    eng.addpath(m_source_dir)  # 添加MATLAB代码文件夹到路径
    result = eng.sample(filename, workdir, input_parameters)
    # print(result)  # 输出结果 1-正确，0-错误
    eng.quit()


# 1 把 track CSV 文件转为 TXT 文件并计算 x 和 y 的平均值
def zhou_convert_csv_to_txt_and_calculate_avg():
    source_dir =    os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, 'data')
    txt_dir =       os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'txt_{TIF_NAME}')
    os.makedirs(txt_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith('.csv') and filename.startswith('track'):
            csv_path = os.path.join(source_dir, filename)
            txt_filename = filename.replace('.csv', '.txt')
            txt_path = os.path.join(txt_dir, txt_filename)

            total_x = 0.0
            total_y = 0.0
            row_count = 0

            with open(csv_path, 'r') as csv_file, open(txt_path, 'w') as txt_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)

                for row in csv_reader:
                    frame_number = row[0]
                    x = float(row[1])
                    y = float(row[2])
                    relative_intensity = float(row[3]) - float(row[4])

                    total_x += x
                    total_y += y
                    row_count += 1
                    txt_file.write(f"{frame_number},{x},{y},{relative_intensity}\n")

                avg_x = total_x / row_count
                avg_y = total_y / row_count
                txt_file.write(f"Average,{avg_x},{avg_y},\n")

    print("1 zhou_CSV 文件已成功转换为 TXT 文件，并计算 xy 平均值")




# 2 把x y和3种亮度整合到txt文件中
def zhou_process_files():
    source_dir =    os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'data')
    track_dir =     os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'txt_{TIF_NAME}')
    output_dir =    os.path.join('D:\\', 'DeepSIFA_main', 'data', 'MLKL','test', f'v{NUM}', '原始数据')
    os.makedirs(output_dir, exist_ok=True)

    # 找到所有以 'track' 开头的文件
    for filename in os.listdir(track_dir):
        if filename.startswith('track') and filename.endswith('.txt'):
            track_number = re.search(r'track(\d+)', filename)
            if track_number:
                track_num = track_number.group(1)
                track_file_path = os.path.join(track_dir, filename)

                # 读取 track 文件的最后一行
                with open(track_file_path, 'r') as track_file:
                    lines = track_file.readlines()
                    last_line = lines[-1].strip().split(',')
                    if len(last_line) >= 3:  # 确保有足够的列
                        track_value1 = f"{float(last_line[1]):.3f}"  # 保留3位小数
                        track_value2 = f"{float(last_line[2]):.3f}"  # 保留3位小数

                        # 查找对应的 allFramesTrackInten 文件
                        all_frames_file = f'allFramesTrackInten{track_num}.csv'
                        all_frames_path = os.path.join(source_dir, all_frames_file)

                        if os.path.exists(all_frames_path):
                            with open(all_frames_path, 'r') as all_frames:
                                all_lines = all_frames.readlines()
                                
                                # 写入 {数字}.txt 文件
                                output_file_path = os.path.join(output_dir, f'track{track_num}.txt')
                                with open(output_file_path, 'w', newline='') as output_file:
                                    output_writer = csv.writer(output_file)

                                    # 遍历 allFramesTrackInten 文件的每一行
                                    for i, line in enumerate(all_lines):
                                        all_frames_values = line.strip().split(',')
                                        if len(all_frames_values) >= 3:  # 确保有足够的列
                                            all_value1 = all_frames_values[1]
                                            all_value2 = all_frames_values[2]
                                            all_value3 = all_frames_values[3]

                                            # 写入数据
                                            output_writer.writerow([i + 1, track_value1, track_value2, all_value1, all_value2, all_value3])
    print("2 zhou_把x y和3种亮度整合到txt文件中")



# 3 根据指定 TIF 文件的第一帧生成归一化的 PNG 文件
def zhou_save_first_frame_as_png():
    tif_path =              os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', f'{TIF_NAME}.tif')
    png_output_path =       os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'{TIF_NAME}_normalized.png')

    # 读取 TIF 文件的第一帧
    with tifffile.TiffFile(tif_path) as tif:
        first_frame = tif.pages[0].asarray()

    # 统计 first_frame 中的最小值和最大值并进行归一化
    min_pixel_value = np.min(first_frame)
    max_pixel_value = np.max(first_frame)
    normalized_frame = ((first_frame - min_pixel_value) / (max_pixel_value - min_pixel_value)) * 255
    normalized_frame = normalized_frame.astype(np.uint8)  # 转换为 8 位整数类型

    # 保存归一化后的第一帧为 PNG 文件
    tifffile.imwrite(png_output_path, normalized_frame)
    print(f"3 zhou_PNG 文件已保存到 {png_output_path}")



# 4 根据 TXT 文件中的平均 x 和 y 坐标，在 PNG 图片上绘制红点。
def zhou_plot_points_on_image():
    # 定义图片路径和 TXT 文件目录
    image_path =        os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory =     os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'txt_{TIF_NAME}')
    output_image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'{TIF_NAME}_with_points.png')
    # 读取 PNG 图片
    image = tifffile.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')  # 显示图片
    
    # 遍历每个 TXT 文件
    txt_files = [os.path.join(txt_directory, f) for f in os.listdir(txt_directory) if f.endswith('.txt')]
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            lines = file.readlines()  # 读取所有行
            line = lines[-1]  # 获取最后一行
            # 查找包含 'Average' 的行
            if 'Average' in line:
                # 以逗号分割行，并提取 x 和 y 坐标
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    x = float(parts[1])  # 第二个值（x 坐标）
                    y = float(parts[2])  # 第三个值（y 坐标）
                    # 在图片上画出红点
                    ax.plot(x, y, 'ro', markersize=5)  # 'ro'表示红色点

    # 保存标记后的图片
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"4 zhou_带标记的 PNG 图片已保存到 {output_image_path}")



# 5 从指定目录中的 TXT 文件提取坐标，并将结果保存到 CSV 文件。
def zhou_extract_coordinates_to_csv():
    txt_directory =     os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'txt_{TIF_NAME}')
    output_csv_path =   os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'txt_{TIF_NAME}', 'message.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_name', 'x', 'y'])

        for filename in os.listdir(txt_directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(txt_directory, filename)
                
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    last_line = lines[-1].strip()
                    parts = last_line.split(',')
                    if len(parts) >= 3:
                        x = parts[1].strip()  # 获取第一个逗号后的数据
                        y = parts[2].strip()  # 获取第二个逗号后的数据
                        csv_writer.writerow([filename, x, y])

    print(f"5 zhou_坐标数据已成功保存到 {output_csv_path}")
    print('生成荧光曲线 完成！！！')



if __name__ == "__main__":
    parse_config = get_cfg()
    TIF_NAME = os.path.basename(parse_config.filename).replace('.tif','')
    NUM = '2'
    delete_directories()
    analyze_image_with_matlab(parse_config.filename)
    zhou_convert_csv_to_txt_and_calculate_avg()
    zhou_process_files()
    zhou_save_first_frame_as_png()
    zhou_plot_points_on_image()
    zhou_extract_coordinates_to_csv()








# #import trackingSM
# import os
# import matlab.engine

# eng = matlab.engine.start_matlab()  #启动matlab

# workdir = r'D:\DeepSIFA_main\data\MLKL\S55C_label_alex55_15nM'  #数据输出目录
# filename = r'D:\DeepSIFA_main\data\MLKL\S55C_label_alex55_15nM.tif'  #要分析的图片文件
# m_source_dir = r'D:\DeepSIFA_main\CreateTrace'    #mainWithoutUI在内的matlab代码文件夹

# # 创建一个Python字典来模拟MATLAB struct, 用来描述输入参数
# # 数值必须为整数或字符型
# input_parameters = {
#     'spotTrackingRadius': 3, # 亮点最大跳跃距离，默认3px
#     'threshold': 2,  # 阈值, 默认2
#     'gaussFitWidth': 3, # 高斯拟合宽度控制, 2*gaussFitWidth-1, 默认3px
#     'frameLength': 30, # 光斑最小持续的帧数, 默认20
#     'frameGap': 2,  # 描述光斑可能不连续的最大帧数, 默认0
#     'trackMethod': 'default', # default, or u-track: 直接调用u-track
#     'outputIntegralIntensity': 1, #是否通过高斯拟合函数积分计算总强度, 速度比较慢，默认1
#     'frameStart': 1, # 需要分析的起始帧，默认1
#     'frameEnd': 'inf', #需要分析的终止帧，默认movie的最后一帧，暂未加入！！！
#     'utrackMotionType': 0 # 此参数为utrack中定义的三种运动模式，具体我也不懂，看userguide
#                            # 0-linear motion 1- linear+random motion with constant vel 
#                            # 2- linear+random motion. movement along a straight line but with the possibility of immediate direction reversal
    
# }

# # # 检查路径是否存在
# # if not os.path.exists(workdir):
# #     print(f"工作目录不存在: {workdir}")
# # if not os.path.exists(filename):
# #     print(f"图片文件不存在: {filename}")
# # if not os.path.exists(m_source_dir):
# #     print(f"MATLAB代码目录不存在: {m_source_dir}")

# eng.addpath(m_source_dir)  #打开文件夹
# result = eng.sample(filename, workdir, input_parameters)
# print(result) #1-正确  0-错误
# eng.quit()



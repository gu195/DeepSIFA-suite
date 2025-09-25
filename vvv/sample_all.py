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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from functools import partial
from contextlib import redirect_stdout
import gc


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=r'D:\DeepSIFA_main\data\wt_0119\I0')
    parser.add_argument('--save_path', type=str, default=r'D:\DeepSIFA_main\data\wt_0119\I0')

    # 超参数范围
    # parser.add_argument('--spotTrackingRadius_values', type=int, default=[3], help='List of spot tracking radius values')
    # parser.add_argument('--gaussFitWidth_values', type=int, default=[3], help='List of Gaussian fit width values')
    # parser.add_argument('--frameLength_values', type=int, default=[20],  help='List of frame length values')
    # parser.add_argument('--frameGap_values', type=int, default=[3], help='List of frame gap values')
    
    parser.add_argument('--threshold', type=int, default=165, help='threshold')
    parser.add_argument('--outputIntegralIntensity', type=int, default=1, help='Is the total intensity calculated by integrating the Gaussian fitting function')
    parser.add_argument('--frameStart', type=int, default=1, help='spot tracking radius values')
    parser.add_argument('--frameEnd', type=int, default=2000, help='spot tracking radius values')
    parser.add_argument('--spotTrackingRadius_values', type=int, default=2, help='spot tracking radius values')
    parser.add_argument('--gaussFitWidth_values', type=int, default=5, help='Gaussian fit width values')
    parser.add_argument('--frameLength_values', type=int, default=50,  help='frame length values')
    parser.add_argument('--frameGap_values', type=int, default=2, help='frame gap values')
    parse_config = parser.parse_args()

    return parse_config


def rename_tif_files(folder_path):
    # 获取文件夹中所有 .tif 文件
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    # 对文件名排序（根据字母序或数字序）
    tif_files.sort()
    
    # 遍历排序后的文件，重命名文件
    for index, file_name in enumerate(tif_files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{index}_{file_name}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")



def delete_directories(save_path):
    directories_to_delete = [
        os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME),
        # os.path.join('D:\\', 'DeepSIFA_main', 'data', 'MLKL', 'test', 'v2')
        SAVE_PATH
    ]

    for i, dir_path in enumerate(directories_to_delete):
            if os.path.exists(dir_path):
                # 如果是第二个目录，添加路径包含检查
                if i == 1:
                    required_subpaths = ['DeepSIFA_main', 'data']
                    if not all(subpath in dir_path for subpath in required_subpaths):
                        print(f"1 路径未满足指定子路径条件，不执行删除操作: {dir_path}")
                        continue

                # 计算目录深度
                depth = len(os.path.normpath(dir_path).split(os.sep))
                if depth >= 5:
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    # print(f"1 目录 {dir_path} 删除完毕")
                else:
                    print(f"1 未达到5级目录深度，不执行删除操作: {dir_path}")
            else:
                print(f"1 目录不存在: {dir_path}")


def analyze_image_with_matlab(filename,flag):
    # 启动MATLAB引擎
    eng = matlab.engine.start_matlab()
    file_basename = os.path.basename(filename).replace('.tif','')

    if ' ' in file_basename:
        file_basename = file_basename.replace(' ', '_')

        # 重新命名文件
        new_filename = os.path.join(os.path.dirname(filename), file_basename + '.tif')
        os.rename(filename, new_filename)
        filename = new_filename  # 更新filename为新命名的路径
    # print('1212 filename', filename)
    workdir = os.path.join(r'D:\DeepSIFA_main\data\example', file_basename)
    m_source_dir = r'D:\DeepSIFA_main\CreateTrace'  # MATLAB代码文件夹

    if  not flag:
        # 定义输入参数字典
        input_parameters = {
            # 'spotTrackingRadius': 3,  # 亮点最大跳跃距离，默认3px
            # 'threshold': 40,  # 阈值, 默认2
            # 'gaussFitWidth': 3,  # 高斯拟合宽度控制, 默认3px
            # 'frameLength': 30,  # 光斑最小持续的帧数, 默认20
            # 'frameGap': 2,  # 描述光斑可能不连续的最大帧数, 默认0
            # 'trackMethod': 'default',  # 轨迹跟踪方法
            # 'outputIntegralIntensity': 1,  # 是否计算总强度, 默认1
            # 'frameStart': 1,  # 分析的起始帧
            # 'frameEnd': 'inf',  # 分析的终止帧
            # 'utrackMotionType': 0  # u-track运动模式
        }
        # # 执行MATLAB分析
        # eng.addpath(m_source_dir)  # 添加MATLAB代码文件夹到路径
        # result = eng.sample(filename, workdir, input_parameters)
        # eng.quit()

    else:
        # 定义超参数范围
        spotTrackingRadius_values = parse_config.spotTrackingRadius_values
        gaussFitWidth_values = parse_config.gaussFitWidth_values
        frameLength_values = parse_config.frameLength_values
        frameGap_values = parse_config.frameGap_values


        frameStart = parse_config.frameStart
        frameEnd = parse_config.frameEnd
        spotTrackingRadius = spotTrackingRadius_values
        gaussFitWidth = gaussFitWidth_values
        frameLength = frameLength_values
        frameGap = frameGap_values
        threshold = parse_config.threshold

        input_parameters = {
            'frameStart': frameStart,
            'frameEnd': frameEnd,
            'spotTrackingRadius': spotTrackingRadius,
            'threshold': threshold,
            'gaussFitWidth': gaussFitWidth,
            'frameLength': frameLength,
            'frameGap': frameGap,
            'trackMethod': 'default',
            'outputIntegralIntensity': 1,
            'utrackMotionType': 0
        }

        # 构造超参数后缀
        # params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
        params_suffix = ''
        workdir = os.path.join(r'D:\DeepSIFA_main\data\Cache', file_basename, params_suffix)
        os.makedirs(workdir, exist_ok=True)
        # 执行MATLAB分析
        eng.addpath(m_source_dir)  # 添加MATLAB代码文件夹到路径
        result = eng.sample(filename, workdir, input_parameters)





        # # 遍历所有超参数组合
        # for spotTrackingRadius in spotTrackingRadius_values:
        #     for gaussFitWidth in gaussFitWidth_values:
        #         for frameLength in frameLength_values:
        #             for frameGap in frameGap_values:
        #                 # 定义输入参数
        #                 input_parameters = {
        #                     'spotTrackingRadius': spotTrackingRadius,
        #                     'threshold': 2,
        #                     'gaussFitWidth': gaussFitWidth,
        #                     'frameLength': frameLength,
        #                     'frameGap': frameGap,
        #                     'trackMethod': 'default',
        #                     'outputIntegralIntensity': 1,
        #                     'frameStart': 1,
        #                     'frameEnd': 'inf',
        #                     'utrackMotionType': 0
        #                 }

        #                 # 构造超参数后缀
        #                 params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
        #                 workdir = os.path.join(r'D:\DeepSIFA_main\data\测试', file_basename, params_suffix)

        #                 # 确保输出目录存在
        #                 os.makedirs(workdir, exist_ok=True)
        #                 # 执行MATLAB分析
        #                 eng.addpath(m_source_dir)  # 添加MATLAB代码文件夹到路径
        #                 result = eng.sample(filename, workdir, input_parameters)


# 1 把 track CSV 文件转为 TXT 文件并计算 x 和 y 的平均值
def zhou_convert_csv_to_txt_and_calculate_avg(params_suffix):
    source_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, 'data')
    txt_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    os.makedirs(txt_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith('.csv') and filename.startswith('track'):
            csv_path = os.path.join(source_dir, filename)
            txt_filename = filename.replace('.csv', '.txt')
            txt_path = os.path.join(txt_dir, txt_filename)

            # total_x = 0.0
            # total_y = 0.0
            # row_count = 0

            with open(csv_path, 'r') as csv_file, open(txt_path, 'w') as txt_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)

                for row in csv_reader:
                    frame_number = row[0]
                    x = float(row[1])
                    y = float(row[2])
                    relative_intensity = float(row[3]) - float(row[4])

                    # total_x += x
                    # total_y += y
                    # row_count += 1
                    txt_file.write(f"{frame_number},{x},{y},{relative_intensity}\n")

                # avg_x = total_x / row_count
                # avg_y = total_y / row_count
                # txt_file.write(f"Average,{avg_x},{avg_y},\n")

    print("1 CSV文件转换为 TXT 文件")




# 2 把x y和3种亮度整合到txt文件中
def zhou_process_files(params_suffix):
    source_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, 'data')
    track_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_dir =    os.path.join(SAVE_PATH, params_suffix, 'row_data')
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
                                    header = ['id', 'avg_x', 'avg_y', 'x', 'y', 'bg', 'Square_intensity', 'circular_intensity', 'Gaussian_intensity']
                                    output_writer.writerow(header)
                                    # 遍历 allFramesTrackInten 文件的每一行
                                    for i, line in enumerate(all_lines):
                                        all_frames_values = line.strip().split(',')
                                        if len(all_frames_values) >= 3:  # 确保有足够的列
                                            id = all_frames_values[0]
                                            X = all_frames_values[1]
                                            Y = all_frames_values[2]
                                            BG = float(all_frames_values[3])  # Convert to float
                                            intensity1 = round(float(all_frames_values[4]) - BG * 4, 3)
                                            intensity2 = round(float(all_frames_values[5]) - BG * 9, 3)
                                            intensity3 = round(float(all_frames_values[6]) - BG * 4, 3)

                                            # 写入数据
                                            output_writer.writerow([id, track_value1, track_value2, X, Y, BG, intensity1, intensity2, intensity3])
    print("2 把X,Y,BG,3种亮度整合到txt文件中")



# 3 根据指定 TIF 文件的第一帧生成归一化的 PNG 文件
def zhou_save_first_frame_as_png(tif_path):
    # tif_path =              os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', f'{TIF_NAME}.tif')
    png_output_path =       os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, f'{TIF_NAME}_normalized.png')

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
    print(f"3 PNG文件已保存到 {png_output_path}")





# 4 从指定目录中的 TXT 文件提取坐标，并将结果保存到 CSV 文件
def zhou_extract_coordinates_to_csv(params_suffix):
    # 定义输入目录和输出文件路径
    directory_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_file1 = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    output_file2 = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_原始.csv')
    # 获取符合条件的文件
    txt_files = [f for f in os.listdir(directory_path) if f.startswith('track') and f.endswith('.txt')]

    results = []  # 用于存储结果

    for txt_file in txt_files:
        file_path = os.path.join(directory_path, txt_file)

        # 读取文件内容
        data = np.loadtxt(file_path, delimiter=',')  # 假设文件是逗号分隔的
        x_values = data[:, 1]  # 第一列是 x
        y_values = data[:, 2]  # 第二列是 y

        # 计算平均值
        avg_x = np.mean(x_values)
        avg_y = np.mean(y_values)

        # 计算每个点到平均点的距离
        distances = np.sqrt((x_values - avg_x) ** 2 + (y_values - avg_y) ** 2)

        # 获取最大距离
        max_distance = np.max(distances)

        # 提取 track 后的数字作为 ID
        track_id = txt_file.split('track')[-1].split('.')[0]

        # 保存结果
        results.append([track_id, avg_x, avg_y, max_distance])

    # 保存到 CSV 文件
    df = pd.DataFrame(results, columns=['file_name', 'x', 'y', 'max_distance'])
    df.to_csv(output_file1, index=False)
    df.to_csv(output_file2, index=False)
    percentile_95 = df['max_distance'].quantile(0.95)
    drift_distance = percentile_95
    print(f"4 avg_xy已成功保存到message.csv")
    return drift_distance



# 5 根据drift_distance进行聚类
def group_points(drift_distance, drift_distance_factor, params_suffix):
    # 读取 CSV 文件
    message_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    data = pd.read_csv(message_path)

    # 提取 x, y 坐标和文件名
    file_names = data['file_name'].values
    x_coords = data['x'].values
    y_coords = data['y'].values

    groups = []  # 存储分组结果
    visited = set()  # 记录已处理过的点的索引

    # 计算两点之间的距离
    def calc_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # 遍历所有点，进行分组
    for i in range(len(data)):
        if i in visited:
            continue
        
        group = [file_names[i]]  # 当前组，首先加入当前点
        visited.add(i)

        # 查找与当前点距离小于0.2的其他点
        for j in range(i + 1, len(data)):
            if j in visited:
                continue
            dist = calc_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
            if dist < drift_distance * drift_distance_factor:
                group.append(file_names[j])
                visited.add(j)
        
        # 如果该组有多个点，加入到结果列表
        if len(group) > 1:
            groups.append(group)

    # 写入分组结果到 message_group.csv
    output_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')
    with open(output_path, 'w', newline='') as file:
        file.write("id,LIST\n")
        for idx, group in enumerate(groups, start=1):   
            file.write(f"{idx},{' '.join(map(str, group))}\n")

    # 写入分组结果到 message_group.csv
    output_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group_原始.csv')
    with open(output_path, 'w', newline='') as file:
        file.write("id,LIST\n")
        for idx, group in enumerate(groups, start=1):   
            file.write(f"{idx},{' '.join(map(str, group))}\n")

    print(f"5 drift_distance聚类结果已保存到message_group.csv")



# 5 根据聚类结果，后处理，看是否需要连接轨迹
def merge_and_delete_files(threshold=50):
    """
    该函数根据给定的条件合并 `A` 和 `B` 文件为 `C`，并删除 `B` 文件。

    参数:
    - base_dir: 存储数据文件的基础目录路径
    - message_file_name: message.csv 的文件名，默认是 'message.csv'
    - threshold: 用于判断是否合并的阈值，默认为 50

    结果:
    - 合并 `A` 和 `B` 文件为 `C`，并删除 `B` 文件
    """
    base_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, 'data')
    # 读取 message.csv 文件
    message_file_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')
    message_data = pd.read_csv(message_file_path)

    # 遍历 message.csv 中的每一行，获取 LIST 列的值
    for index, row in message_data.iterrows():
        LIST = row['LIST']  # 假设 'LIST' 是 message.csv 中的列名
        if isinstance(LIST, str):  # 确保 LIST 是字符串
            # 使用空格分割 LIST，并检查分割后的部分数是否为2
            split_list = LIST.split(' ')
            if len(split_list) == 2:  # 确保只有两个元素
                X1, X2 = map(int, LIST.split(' '))  # 假设 LIST 中是逗号分隔的数字
                # 查找 allFramesTrackIntenX1.csv 和 allFramesTrackIntenX2.csv 文件
                file_x1 = os.path.join(base_dir, f'allFramesTrackInten{X1}.csv')
                file_x2 = os.path.join(base_dir, f'allFramesTrackInten{X2}.csv')

                # 读取文件内容
                df_x1 = pd.read_csv(file_x1)
                df_x2 = pd.read_csv(file_x2)

                # 获取第一行第一列的数字
                A1 = df_x1.iloc[0, 0]
                B1 = df_x2.iloc[0, 0]

                # 判断哪个文件第一行开头的数字更小,更小的为A，大的为B
                if A1 < B1:
                    A_file = file_x1
                    A_num = X1
                    B_file = file_x2
                    B_num = X2
                else:
                    A_file = file_x2
                    A_num = X2
                    B_file = file_x1
                    B_num = X1

                # 读取 trackA1.csv 和 trackB1.csv
                track_a1_file = os.path.join(base_dir, f'track{A_num}.csv')
                track_b1_file = os.path.join(base_dir, f'track{B_num}.csv')

                df_track_a1 = pd.read_csv(track_a1_file)
                df_track_b1 = pd.read_csv(track_b1_file)


                track_a1_end_num = df_track_a1.iloc[-1, 0]  # trackA1.csv 最后一行的第一列数字
                track_b1_start_num = df_track_b1.iloc[0, 0]  # trackB1.csv 第一行的第一列数字

                if abs(track_b1_start_num - track_a1_end_num) < threshold:

                    # ------------------------------合并文件----------------------------------
                    df_A = pd.read_csv(A_file, header=None)
                    df_B = pd.read_csv(B_file, header=None)

                    # 获取 A 文件的最后一行第一列数字
                    A_num_end = df_A.iloc[-1, 0]

                    # 根据 A_num_end 从 B 文件中找到对应的行
                    B_part = df_B[df_B.iloc[:, 0] > A_num_end]

                    # # 合并 A 和 B
                    # df_C = pd.concat([df_A, B_part])
                    df_C = pd.concat([df_A, B_part], axis=0, ignore_index=True)

                    # 保存为 allFramesTrackIntenA1.csv
                    output_file = os.path.join(base_dir, f'allFramesTrackInten{A_num}.csv')
                    df_C.to_csv(output_file, index=False, header=None)

                    # 删除 B 文件
                    os.remove(os.path.join(base_dir, f'allFramesTrackInten{B_num}.csv'))
                    os.remove(os.path.join(base_dir, f'Track{B_num}.csv'))
                    print(f"合并成功: {A_file} 和 {B_file} 合并为 {output_file} 并删除 {B_file}")
                    #-------------------------------------------------------------------------------
                    

                    # ---------更新 message.csv 文件，删除 file_name 列中对应的行---------
                    txt_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
                    message = pd.read_csv(os.path.join(txt_dir, 'message.csv'))
                    updated_message_data = message[message['file_name'] != B_num]
                    updated_message_file_path = os.path.join(txt_dir, 'message.csv')
                    updated_message_data.to_csv(updated_message_file_path, index=False)

                    # 更新 message_group.csv 文件，删除 file_name 列中对应的行
                    message_group_data = pd.read_csv(message_file_path)
                    updated_message_group_data = message_group_data[~message_group_data['LIST'].apply(lambda x: str(B_num) in str(x))]
                    updated_message_group_file_path = os.path.join(txt_dir, 'message_group.csv')
                    updated_message_group_data.to_csv(updated_message_group_file_path, index=False)

                    print(f"合并成功: {A_file} 和 {B_file} 合并为 {output_file}")
                    print(f"已更新 message.csv 为 {updated_message_file_path}")
                    print(f"已更新 message_group.csv 为 {updated_message_group_file_path}")
                    #------------------------------------------------------------------------




# 6 原始：数字和圆圈的 PNG 图片已保存到
def zhou_plot_points_on_image1(params_suffix):
    # 定义路径
    image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'{TIF_NAME}_coordinate_原始.png')
    message_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_原始.csv')
    message_group_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group_原始.csv')

    # 读取数据
    image = tifffile.imread(image_path)
    message_data = pd.read_csv(message_file)
    message_group_data = pd.read_csv(message_group_file)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')  # 显示图片
    existing_positions = []  # 用于记录数字放置位置

    # 遍历 message.csv 中的点
    # 记录已经绘制过的 track_id
    drawn_ids = set()
    drawn_group_ids = set()  # 已绘制的 group_id

    for _, row in message_data.iterrows():
        track_id = int(row['file_name'])
        
        # 检查该 track_id 是否已经绘制过
        if track_id in drawn_ids:
            continue
        
        drawn_ids.add(track_id)
        
        x = row['x']
        y = row['y']
        matching_group = message_group_data[message_group_data['LIST'].apply(lambda x: str(track_id) in x.split())]

        if matching_group.empty:
            circle_radius = 1  # 圆圈半径
            # 不在聚类信息中的点
            ax.plot(x, y, 'ro', markersize=1)  # 红点
            circle = plt.Circle((x, y), circle_radius, color='green', fill=False, lw=1)
            ax.add_patch(circle)

            # 在红色圆圈中心添加数字
            ax.annotate(
                track_id,  # 显示的数字是点的 track_id
                xy=(x, y), xytext=(x, y),  # 数字位于圆圈中心
                fontsize=1, color='white', ha='center', va='center',
                bbox=dict(facecolor='green', alpha=0.8, edgecolor='none', pad=1)  # 数字背景为红色
            )
        else:
            # 在聚类信息中的点，围绕圆形分布
            group_list = matching_group.iloc[0]['LIST']
            if isinstance(group_list, str):
                group_items = list(map(int, group_list.split()))
            else:
                group_items = []

            # 绘制聚类的点
            for idx, group_id in enumerate(group_items):
                circle_radius = 1.5
                # 检查 group_id 是否已经绘制过
                if group_id in drawn_group_ids:
                    continue
                drawn_group_ids.add(group_id)
                angle = idx * (2 * np.pi / len(group_items))  # 按顺序分布
                adjusted_x = x + circle_radius * np.cos(angle)
                adjusted_y = y + circle_radius * np.sin(angle)
                # 绘制红色圆圈
                circle = plt.Circle((x, y), circle_radius, color='red', fill=False, lw=1)  # 蓝色圆圈
                ax.add_patch(circle)           
                
                ax.annotate(
                    group_id,
                    xy=(x, y),
                    xytext=(adjusted_x, adjusted_y),
                    fontsize=1, color='white', ha='center', va='center',
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none',pad=1),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1, shrinkA=0, shrinkB=0),
                )


                existing_positions.append((adjusted_x, adjusted_y))


    # 保存标记后的图片
    plt.savefig(output_image_path, dpi=900, bbox_inches='tight')
    print(f"6 荧光坐标图片可视化完毕")
    return output_image_path



# 6 后处理：数字和圆圈的 PNG 图片已保存到
def zhou_plot_points_on_image2(params_suffix):
    # 定义路径
    image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'{TIF_NAME}_coordinate.png')
    message_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    message_group_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')

    # 读取数据
    image = tifffile.imread(image_path)
    message_data = pd.read_csv(message_file)
    message_group_data = pd.read_csv(message_group_file)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')  # 显示图片
    existing_positions = []  # 用于记录数字放置位置

    # 遍历 message.csv 中的点
    # 记录已经绘制过的 track_id
    drawn_ids = set()
    drawn_group_ids = set()  # 已绘制的 group_id

    for _, row in message_data.iterrows():
        track_id = int(row['file_name'])
        
        # 检查该 track_id 是否已经绘制过
        if track_id in drawn_ids:
            continue
        
        drawn_ids.add(track_id)
        
        x = row['x']
        y = row['y']
        matching_group = message_group_data[message_group_data['LIST'].apply(lambda x: str(track_id) in x.split())]

        if matching_group.empty:
            circle_radius = 1  # 圆圈半径
            # 不在聚类信息中的点
            ax.plot(x, y, 'ro', markersize=1)  # 红点
            circle = plt.Circle((x, y), circle_radius, color='green', fill=False, lw=1)
            ax.add_patch(circle)

            # 在红色圆圈中心添加数字
            ax.annotate(
                track_id,  # 显示的数字是点的 track_id
                xy=(x, y), xytext=(x, y),  # 数字位于圆圈中心
                fontsize=1, color='white', ha='center', va='center',
                bbox=dict(facecolor='green', alpha=0.8, edgecolor='none', pad=1)  # 数字背景为红色
            )
        else:
            # 在聚类信息中的点，围绕圆形分布
            group_list = matching_group.iloc[0]['LIST']
            if isinstance(group_list, str):
                group_items = list(map(int, group_list.split()))
            else:
                group_items = []

            # 绘制聚类的点
            for idx, group_id in enumerate(group_items):
                circle_radius = 1.5
                # 检查 group_id 是否已经绘制过
                if group_id in drawn_group_ids:
                    continue
                drawn_group_ids.add(group_id)
                angle = idx * (2 * np.pi / len(group_items))  # 按顺序分布
                adjusted_x = x + circle_radius * np.cos(angle)
                adjusted_y = y + circle_radius * np.sin(angle)
                # 绘制红色圆圈
                circle = plt.Circle((x, y), circle_radius, color='red', fill=False, lw=1)  # 蓝色圆圈
                ax.add_patch(circle)           
                
                ax.annotate(
                    group_id,
                    xy=(x, y),
                    xytext=(adjusted_x, adjusted_y),
                    fontsize=1, color='white', ha='center', va='center',
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none',pad=1),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1, shrinkA=0, shrinkB=0),
                )


                existing_positions.append((adjusted_x, adjusted_y))


    # 保存标记后的图片
    plt.savefig(output_image_path, dpi=900, bbox_inches='tight')
    print(f"6 荧光坐标图片可视化完毕")
    return output_image_path







# 生成测试集
# 生成测试集
# 生成测试集
# 生成测试集

# 1. 生成data存放目录
def create_directory(params_suffix):
    directory_path = os.path.join(SAVE_PATH, params_suffix, 'row_data')
    os.makedirs(directory_path, exist_ok=True)
    print(f"7 测试集目录 '{directory_path}' 已成功创建。")
    return directory_path



# 2. 生成.txt后缀并加上_bad _good
def rename_and_append_suffix(params_suffix):
    for label in ['bad', 'good']:
        directory = os.path.join(SAVE_PATH, params_suffix, label)
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    new_file_name = f"{name}_{label}{ext}" if ext else f"{name}_{label}.txt"
                    new_file_path = os.path.join(directory, new_file_name)
                    os.rename(file_path, new_file_path)

    print("8 测试集生成.txt后缀, 加上_bad _good")



# 3. 移动文件并清理目录
def move_and_clean(params_suffix):
    good_dir = os.path.join(SAVE_PATH, params_suffix, 'good')
    bad_dir = os.path.join(SAVE_PATH, params_suffix, 'bad')
    target_dir = os.path.join(SAVE_PATH, params_suffix, 'row_data')

    # 定义用于移动文件的操作
    for src_dir in [good_dir, bad_dir]:
        if os.path.exists(src_dir):
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dest_file = os.path.join(target_dir, filename)
                if os.path.isfile(src_file):
                    shutil.move(src_file, dest_file)
            os.rmdir(src_dir)  # 删除空目录

    print("9 文件已成功移动并删除目录。")




# 4. alphaK10 获取亮度信息
def get_brightness_info(params_suffix):
    if DATA == 'alphak10':
        directory = os.path.join(SAVE_PATH, params_suffix, 'row_data')
        files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        for file in files:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)
            df[5] = df[1] - df[2]
            df.columns = ['Time[s]', 'CH1', 'BGND1', 'CH2', 'BGND2', 'brightness']
            df.to_csv(file_path, sep="\t", index=False, header=True)
        print("4 alphaK10 获取亮度信息。")




# 5. 把txt文件都转化为npz文件存储
def convert_txt_to_npz(params_suffix):
    directory = os.path.join(SAVE_PATH, params_suffix, 'row_data')

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, os.path.splitext(filename)[0] + '.npz')

            # 打开并处理txt文件，保存为npz文件
            with open(file_path, 'r') as file:
                next(file)  # 跳过首行
                lines  = file.readlines()[:-1]  # 跳过最后一行
                combined_data = []

                for line in lines :
                    splitted_line = re.split(',|\t| ', line.strip())
                    combined_data.append([splitted_line[0], splitted_line[3],splitted_line[4], splitted_line[-1]])#第一行和最后一行

                combined_data = np.array(combined_data)
                assert combined_data.shape[1] == 4, "Data shape is not (num, 2)"

                # 保存为 .npz 文件
                np.savez(output_path, combined_data=combined_data)

    print("10 把txt文件都转化为npz文件存储")




# 6. 归一化
def normalize_data(params_suffix):
    source_dir = os.path.join(SAVE_PATH, params_suffix, 'row_data')
    target_dir = os.path.join(SAVE_PATH, params_suffix, '归一化后npz')
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.npz'):
            file_path = os.path.join(source_dir, filename)
            npz_data = np.load(file_path)
            data = npz_data['combined_data']
            data = np.transpose(data, (1, 0))
            try:
                data = data.astype(np.float64)
            except ValueError:
                print("数据转换失败：数组包含非数值字符串。")
            # Normalize only the last column
            last_column = data[-1]  # Get the last column
            normalized_last_column = (last_column - last_column.min()) / (last_column.max() - last_column.min())
            data[-1] = normalized_last_column  # Replace the last column with its normalized version

            save_path = os.path.join(target_dir, filename)
            np.savez(save_path, combined_data=data)
    print("11 时间序列归一化完成")







def resize_and_smooth(params_suffix):
    # 第 7 段代码：Resize 到 1024，并且高斯平滑
    source_dir = os.path.join(SAVE_PATH, params_suffix, '归一化后npz')
    target_dir = os.path.join(SAVE_PATH, params_suffix, 'processing_data')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.npz'):
            file_path = os.path.join(source_dir, filename)
            npz_data = np.load(file_path)
            data = npz_data['combined_data']
            original_data = data[3, :]

            original_indices = np.linspace(0, 1, len(original_data))
            new_indices = np.linspace(0, 1, NUM_dimension)

            interpolation_function = interp1d(original_indices, original_data, kind='linear')
            interpolated_data = interpolation_function(new_indices)
            # 高斯平滑
            sigma = SIGMA
            interpolated_data = gaussian_filter1d(interpolated_data, sigma)
            save_path = os.path.join(target_dir, filename)
            np.savez(save_path, data=interpolated_data)

    print("12 时间序列Resize to 1024 and Gaussian smoothing completed.")


def delete_directories_npz(params_suffix):
    directories_to_delete = [os.path.join(SAVE_PATH, params_suffix, '归一化后npz')]

    for i, dir_path in enumerate(directories_to_delete):
            if os.path.exists(dir_path):
                # 计算目录深度
                depth = len(os.path.normpath(dir_path).split(os.sep))
                if depth >= 5:
                    # for item in os.listdir(dir_path):
                    #     item_path = os.path.join(dir_path, item)
                    #     if os.path.isfile(item_path) or os.path.islink(item_path):
                    #         os.unlink(item_path)
                    #     elif os.path.isdir(item_path):
                    #         shutil.rmtree(item_path)
                    shutil.rmtree(dir_path)
                    # print(f"1 目录 {dir_path} 删除完毕")
                else:
                    print(f"12 未达到5级目录深度，不执行删除操作: {dir_path}")
            else:
                print(f"12 目录不存在: {dir_path}")


def generate_csv(params_suffix):
    # 第 8 段代码：根据 _bad.npz 或 _good.npz 生成标签与 csv 文件
    directory = os.path.join(SAVE_PATH, params_suffix, 'processing_data')
    data = []

    for file in os.listdir(directory):
        if "bad" in file:
            data.append({'file_name': file, 'label': 0})
        elif "good" in file:
            data.append({'file_name': file, 'label': 1})
        else:
            data.append({'file_name': file, 'label': -1})

    df = pd.DataFrame(data, columns=['file_name', 'label'])
    csv_file_path =os.path.join(SAVE_PATH, params_suffix, f'{len(os.listdir(directory))}.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"13 测试集csv生成完毕.")




def process_file_1(npz_file, save_path, params_suffix):# 可视化 归一化后npz
    # 定义npz文件的路径
    source_dir = os.path.join(save_path, params_suffix, '归一化后npz')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    npz_path = os.path.join(source_dir, npz_file)
    npz_data = np.load(npz_path)
    data = npz_data['combined_data'][-1]

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']

    # 创建 x 轴坐标
    x = range(len(data))
    plt.figure(figsize=(18, 6))  # 设置图形大小
    plt.plot(x, data, label='Data Line', linewidth=0.5)
    plt.scatter(x, data, label='Data Points', s=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('{}'.format(npz_file))
    plt.legend()
    plt.xticks(np.arange(0, len(data), step=200))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        plt.axvspan(start[0], end[0], color='red', alpha=0.3)

    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    save_path = os.path.join(source_dir, 'png', npz_file.replace('.npz', '.png'))
    plt.savefig(save_path, dpi=250)
    plt.clf()
    gc.collect()  # 手动清理内存


def process_file_2(npz_file, save_path, params_suffix):# 可视化归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}
    source_dir = os.path.join(save_path, params_suffix, 'processing_data')
    file_path = os.path.join(source_dir, npz_file)
    npz_data = np.load(file_path)
    data = npz_data['data']

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']

    # 创建 x 轴坐标
    x = range(len(data))
    plt.figure(figsize=(18, 6))  # 设置图形大小
    plt.plot(x, data, label='Data Line', linewidth=0.5)
    plt.scatter(x, data, label='Data Points', s=0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('{}'.format(npz_file))
    plt.legend()
    plt.xticks(np.arange(0, len(data), step=200))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        plt.axvspan(start[0], end[0], color='red', alpha=0.3)

    os.makedirs(os.path.join(source_dir, 'png2'), exist_ok=True)
    save_path = os.path.join(source_dir, 'png2', npz_file.replace('.npz', '.png'))
    plt.savefig(save_path, dpi=250)
    plt.clf()
    gc.collect()  # 手动清理内存





# 修改一下画布的长度，保存的地址，多线程运行
def process_file_3(npz_file, save_path, params_suffix):
    # 定义npz文件的路径
    directory = os.path.join(save_path, params_suffix, 'row_data')
    source_dir = os.path.join(save_path, params_suffix, 'processing_data')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    npz_path = os.path.join(directory, npz_file)
    # 加载npz文件
    npz_data = np.load(npz_path)
    
    # 提取存储的数据，假设key是 'combined_data'
    data = npz_data['combined_data']
    id = data[:, 0].astype(float)
    x = data[:, 1].astype(float)
    y = data[:, 2].astype(float)
    intensity = data[:, 3].astype(float)

    # 创建一个包含三个子图的画布
    plt.figure(figsize=(24, 18))

    # 第一个子图：x 曲线图
    plt.subplot(3, 1, 1)  # 3行1列，第一个子图
    plt.plot(id, x, label='X', color='blue')
    plt.title(f'{npz_file.replace(".npz", "")} - X Curve', fontsize=20)
    plt.xlabel('ID', fontsize=16)
    plt.ylabel('X', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 第二个子图：y 曲线图
    plt.subplot(3, 1, 2)  # 3行1列，第二个子图
    plt.plot(id, y, label='Y', color='green')
    plt.title(f'{npz_file.replace(".npz", "")} - Y Curve', fontsize=20)
    plt.xlabel('ID', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 第三个子图：强度曲线图
    plt.subplot(3, 1, 3)  # 3行1列，第三个子图
    plt.plot(id, intensity, label='Intensity', color='red')
    plt.title(f'{npz_file.replace(".npz", "")} - Intensity Curve', fontsize=20)
    plt.xlabel('ID', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 保存图像
    save_path = os.path.join(source_dir, 'png', npz_file.replace('.npz', '.png'))
    plt.savefig(save_path, dpi=250)
    plt.close()  # 关闭图像，防止内存过多占用




def copy_image_to_directory(output_image_path, directory):
    filename = os.path.basename(output_image_path)
    # 拼接目标路径
    target_directory = os.path.join(os.path.dirname(directory), filename)
    
    # 复制图片到目标路径
    shutil.copy(output_image_path, target_directory)
    print(f"15 荧光坐标图片已复制到: {target_directory}")
    return target_directory
    




if __name__ == "__main__":
    DATA = 'MLKL'
    NUM_dimension = 1024
    SIGMA = 1
    # 超参数范围
    parse_config = get_cfg()
    threshold_values = parse_config.threshold
    frameStart_values = parse_config.frameStart
    frameEnd_values = parse_config.frameEnd
    spotTrackingRadius_values = parse_config.spotTrackingRadius_values
    gaussFitWidth_values = parse_config.gaussFitWidth_values
    frameLength_values = parse_config.frameLength_values
    frameGap_values = parse_config.frameGap_values
    # 打印超参数
    print("threshold:", threshold_values)
    print("frameStart:", frameStart_values)
    print("frameEnd:", frameEnd_values) 
    print("spotTrackingRadius_values:", spotTrackingRadius_values)
    print("gaussFitWidth_values:", gaussFitWidth_values)
    print("frameLength_values:", frameLength_values)
    print("frameGap_values:", frameGap_values)


    # 获取文件所在文件夹路径
    folder_path = parse_config.filename
    for filename in os.listdir(folder_path):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and ' ' in filename:
            # 替换空格为下划线
            new_filename = filename.replace(' ', '_')
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} -> {new_file_path}')


    # 遍历文件夹中的所有 .tif 文件
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    for idx, filename in enumerate(tif_files):
        file_path = os.path.join(folder_path, filename)
        TIF_NAME = filename.replace('.tif', '')
        SAVE_PATH = os.path.join(parse_config.save_path, TIF_NAME)
        print('选取的TIRF图片为',TIF_NAME)
        print('测试集保存路径为',SAVE_PATH)

        ''''''
        delete_directories(SAVE_PATH)
        # 调用 MATLAB 分析
        analyze_image_with_matlab(file_path, 1)
        spotTrackingRadius = spotTrackingRadius_values
        gaussFitWidth = gaussFitWidth_values
        frameLength = frameLength_values
        frameGap = frameGap_values
        # 构造超参数后缀
        params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
        params_suffix = ''
        os.makedirs(os.path.join(SAVE_PATH, params_suffix), exist_ok=True)        # 生成超参数相关的 SAVE_PATH
        print(f"matlab end, post processing begin")
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                zhou_convert_csv_to_txt_and_calculate_avg(params_suffix)
                zhou_save_first_frame_as_png(file_path)
                drift_distance = zhou_extract_coordinates_to_csv(params_suffix)
                group_points(drift_distance, 0.2, params_suffix)
                merge_and_delete_files(50)#ATTENTION
                zhou_process_files(params_suffix)
                zhou_plot_points_on_image1(params_suffix)
                output_image_path = zhou_plot_points_on_image2(params_suffix)
        print('后处理-第一阶段-生成荧光曲线 完成')
        # print('-----------------------------------------------------')


        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                # 生成测试集和曲线png
                matplotlib.use('Agg')
                directory_path = create_directory(params_suffix)
                rename_and_append_suffix(params_suffix)
                move_and_clean(params_suffix)
                get_brightness_info(params_suffix)
                convert_txt_to_npz(params_suffix)
                normalize_data(params_suffix)
                resize_and_smooth(params_suffix)
                delete_directories_npz(params_suffix)
                generate_csv(params_suffix)
                num_workers = cpu_count()  # 获取 CPU 核心数

                # source_dir = os.path.join(SAVE_PATH, params_suffix, '归一化后npz')
                # npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
                # # 使用 partial 绑定 params_suffix 参数
                # process_with_suffix = partial(process_file_1, save_path=SAVE_PATH, params_suffix=params_suffix)
                # # 创建进程池并传递绑定函数
                # with Pool(processes=num_workers) as pool:
                #     list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))



                source_dir = os.path.join(SAVE_PATH, params_suffix, 'processing_data')
                npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
                # 使用 partial 绑定 params_suffix 参数
                process_with_suffix = partial(process_file_2, save_path=SAVE_PATH, params_suffix=params_suffix)
                # 创建进程池并传递绑定函数
                with Pool(processes=num_workers) as pool:
                    list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

                
                source_dir = os.path.join(SAVE_PATH, params_suffix, 'row_data')
                npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
                # 使用 partial 绑定 params_suffix 参数
                process_with_suffix = partial(process_file_3, save_path=SAVE_PATH, params_suffix=params_suffix)
                # 创建进程池并传递绑定函数
                with Pool(processes=num_workers) as pool:
                    list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))
                print("14 时间序列可视化完成")
                target_directory = copy_image_to_directory(output_image_path, directory_path)
        print('后处理-第二阶段-生成测试集 完成')
        print('-----------------------------------------------------')




    # 寻找最佳超参数
    # for spotTrackingRadius in spotTrackingRadius_values:
    #     for gaussFitWidth in gaussFitWidth_values:
    #         for frameLength in frameLength_values:
    #             for frameGap in frameGap_values:
    #                 # 构造超参数后缀
    #                 params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
    #                 # 生成超参数相关的 SAVE_PATH
    #                 SAVE_PATH = parse_config.save_path
    #                 os.makedirs(os.path.join(parse_config.save_path, params_suffix), exist_ok=True)
    #                 print(f"Processing: {params_suffix}")
    #                 # print(f"TIF_NAME: {TIF_NAME}")
    #                 # print(f"SAVE_PATH: {SAVE_PATH1}")

    #                 zhou_convert_csv_to_txt_and_calculate_avg(params_suffix)
    #                 zhou_process_files(params_suffix)
    #                 zhou_save_first_frame_as_png()
    #                 drift_distance = zhou_extract_coordinates_to_csv(params_suffix)
    #                 group_points(drift_distance, params_suffix)
    #                 output_image_path = zhou_plot_points_on_image1(params_suffix)
    #                 print('第一阶段-生成荧光曲线 完成')
    #                 print('-----------------------------------------------------')



    #                 # 生成测试集和曲线png
    #                 matplotlib.use('Agg')
    #                 directory_path = create_directory(params_suffix)
    #                 rename_and_append_suffix(params_suffix)
    #                 move_and_clean(params_suffix)
    #                 get_brightness_info(params_suffix)
    #                 convert_txt_to_npz(params_suffix)
    #                 normalize_data(params_suffix)
    #                 resize_and_smooth(params_suffix)
    #                 generate_csv(params_suffix)
    #                 num_workers = cpu_count()  # 获取 CPU 核心数
    #                 source_dir = os.path.join(SAVE_PATH, params_suffix, '原始数据')
    #                 npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    #                 # 使用 partial 绑定 params_suffix 参数
    #                 process_with_suffix = partial(process_file_2, save_path=SAVE_PATH, num_dimension=NUM_dimension, sigma=SIGMA, params_suffix=params_suffix)

    #                 # 创建进程池并传递绑定函数
    #                 with Pool(processes=num_workers) as pool:
    #                     list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))
    #                 print("14 时间序列可视化完成")
    #                 target_directory = copy_image_to_directory(output_image_path, directory_path)
    #                 print('第二阶段-生成测试集 完成')
    #                 print('-----------------------------------------------------')







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



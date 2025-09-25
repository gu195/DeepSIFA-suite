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

ALLOWED_ROOTS = [
    r"D:\DeepSIFA_main\data",  # 原工程根
    r"D:\aaa\bbb\data",        # 你的常用根
    r"D:\aaa\vvv\data",        # 你新的根
]

import os

def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))

def is_in_allowed_roots(path: str) -> bool:
    p = _norm(path)
    for root in ALLOWED_ROOTS:
        try:
            if os.path.commonpath([p, _norm(root)]) == _norm(root):
                return True
        except ValueError:
            # 跨盘符会抛异常，直接跳过
            pass
    return False

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matlab_workdir', type=str, default=r'D:\aaa\vvv\data\Cache')
    parser.add_argument('--matlab_m_source_dir', type=str, default=r'D:\aaa\vvv\CreateTrace')# MATLAB代码文件夹
    parser.add_argument('--filename', type=str, default=r'D:\aaa\vvv\data\MLKL\S55C_label_alex55_10nM_1.tif')
    parser.add_argument('--save_path', type=str, default=r'D:\aaa\vvv\data\MLKL\v1')

    # 超参数范围
    # parser.add_argument('--spotTrackingRadius_values', type=int, default=[3], help='List of spot tracking radius values')
    # parser.add_argument('--gaussFitWidth_values', type=int, default=[3], help='List of Gaussian fit width values')
    # parser.add_argument('--frameLength_values', type=int, default=[20],  help='List of frame length values')
    # parser.add_argument('--frameGap_values', type=int, default=[3], help='List of frame gap values')
    
    parser.add_argument('--threshold', type=int, default=200, help='threshold')
    parser.add_argument('--outputIntegralIntensity', type=int, default=1, help='Is the total intensity calculated by integrating the Gaussian fitting function')
    parser.add_argument('--frameStart', type=int, default=1, help='spot tracking radius values')
    parser.add_argument('--frameEnd', type=int, default=2000, help='spot tracking radius values')
    parser.add_argument('--spotTrackingRadius_values', type=int, default=3, help='spot tracking radius values')
    parser.add_argument('--gaussFitWidth_values', type=int, default=6, help='Gaussian fit width values')#是直径
    parser.add_argument('--frameLength_values', type=int, default=50,  help='frame length values')
    parser.add_argument('--frameGap_values', type=int, default=1, help='frame gap values')
    parse_config = parser.parse_args()

    return parse_config


def delete_directories():
    directories_to_delete = [
        os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME),
        # os.path.join('D:\\', 'DeepSIFA_main', 'data', 'MLKL', 'test', 'v2')
        SAVE_PATH
    ]

    for i, dir_path in enumerate(directories_to_delete):
            if os.path.exists(dir_path):
                # 如果是第二个目录，添加路径包含检查
                if i == 1:
                   dir_path = os.path.normpath(dir_path)  # 统一分隔符
                   if not is_in_allowed_roots(dir_path):
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


def analyze_image_with_matlab(parse_config,flag):
    # 启动MATLAB引擎
    eng = matlab.engine.start_matlab()
    filename = parse_config.filename
    file_basename = os.path.basename(filename).replace('.tif','')

    if ' ' in file_basename:
        file_basename = file_basename.replace(' ', '_')
        # 重新命名文件
        new_filename = os.path.join(os.path.dirname(filename), file_basename + '.tif')
        os.rename(filename, new_filename)
        filename = new_filename  # 更新filename为新命名的路径
    # print('1212 filename', filename)
    # workdir = os.path.join(parse_config.matlab_workdir, file_basename)
    m_source_dir = parse_config.matlab_m_source_dir

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
        workdir = os.path.join(parse_config.matlab_workdir, file_basename, params_suffix)
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
def zhou_convert_csv_to_txt_and_calculate_avg(parse_config, params_suffix):
    source_dir = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, 'data')
    txt_dir = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
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
def zhou_process_files(parse_config,params_suffix):
    source_dir = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, 'data')
    track_dir = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
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
                                            intensity1 = round(float(all_frames_values[4]), 3)
                                            intensity2 = round(float(all_frames_values[5]), 3)
                                            intensity3 = round(float(all_frames_values[6]), 3)
                                            # intensity3 = round(float(all_frames_values[6]) - BG * ((parse_config.gaussFitWidth_values**2 -1) *0.5), 3)
                                            # 写入数据
                                            output_writer.writerow([id, track_value1, track_value2, X, Y, BG, intensity1, intensity2, intensity3])
    print("2 把X,Y,BG,3种亮度整合到txt文件中")



# 3 根据指定 TIF 文件的第一帧生成归一化的 PNG 文件
def zhou_save_first_frame_as_png(parse_config):
    tif_path = parse_config.filename
    # tif_path =              os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', f'{TIF_NAME}.tif')
    png_output_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, f'{TIF_NAME}_normalized.png')

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
def zhou_extract_coordinates_to_csv(parse_config,params_suffix):
    # 定义输入目录和输出文件路径
    directory_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_file1 = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    output_file2 = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_原始.csv')
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
    df = pd.DataFrame(results, columns=['file_name', 'avg_x', 'avg_y', 'max_distance'])
    df.to_csv(output_file1, index=False)
    df.to_csv(output_file2, index=False)
    percentile_95 = df['max_distance'].quantile(0.95)
    drift_distance = percentile_95
    print(f"4 avg_xy已成功保存到message.csv")
    return drift_distance




# 5 根据drift_distance进行聚类
def group_points(drift_distance, drift_distance_factor, parse_config,params_suffix):
    # 读取 CSV 文件
    message_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    data = pd.read_csv(message_path)

    # 提取 x, y 坐标和文件名
    file_names = data['file_name'].values
    x_coords = data['avg_x'].values
    y_coords = data['avg_y'].values

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
            # if dist < drift_distance * drift_distance_factor:
            if dist < drift_distance * drift_distance_factor and abs(x_coords[i] - x_coords[j]) < drift_distance and abs(y_coords[i] - y_coords[j]) < drift_distance:
                group.append(file_names[j])
                visited.add(j)
        
        # 如果该组有多个点，加入到结果列表
        if len(group) > 1:
            groups.append(group)

    # 写入分组结果到 message_group.csv
    output_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')
    with open(output_path, 'w', newline='') as file:
        file.write("id,LIST\n")
        for idx, group in enumerate(groups, start=1):   
            file.write(f"{idx},{' '.join(map(str, group))}\n")

    # 写入分组结果到 message_group.csv
    output_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group_原始.csv')
    with open(output_path, 'w', newline='') as file:
        file.write("id,LIST\n")
        for idx, group in enumerate(groups, start=1):   
            file.write(f"{idx},{' '.join(map(str, group))}\n")

    print(f"5 drift_distance聚类结果已保存到message_group.csv")



# 5 根据聚类结果，后处理，看是否需要连接轨迹
# 要能精确合并，强度不能有突变。不能把不是一个位置的轨迹合并到
# 这里面肯定有bug \png_团\id42_256_267_285 明明是对2个原始才操作，居然有3个元素也会消失，还是有bug
import re
import glob
import pandas as pd

def merge_and_delete_files(parse_config, threshold=50):
    """
    根据 message_group.csv 的成对关系合并轨迹：
    - 缺失的 track*.csv / allFramesTrackInten*.csv -> 打印 WARN 并跳过，不再报错崩溃
    - 合并后：更新 allFramesTrackIntenA.csv；删除 B 的文件（兼容 Track/track 大小写）
    - 同步更新 message.csv / message_group.csv 并记录 merge_log.txt
    threshold: 两条轨迹首尾帧差的阈值（帧）
    """
    # 基础目录（与原脚本一致）
    base_dir = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, 'data')
    txt_dir  = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    message_file_path = os.path.join(txt_dir, 'message_group.csv')

    if not os.path.exists(message_file_path):
        print(f"[WARN] message_group.csv not found: {message_file_path}")
        return

    message_data = pd.read_csv(message_file_path)

    def _safe_remove(p):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            print(f"[WARN] remove failed: {p} -> {e}")

    num_merged, skipped = 0, 0

    for _, row in message_data.iterrows():
        LIST = row.get('LIST', '')
        group_id = row.get('id', None)

        if not isinstance(LIST, str):
            skipped += 1
            continue
        parts = LIST.split()
        if len(parts) != 2:
            # 只处理“两两合并”；其他复杂分组先跳过
            skipped += 1
            continue

        # 解析成两个编号
        try:
            X1, X2 = map(int, parts)
        except Exception:
            skipped += 1
            continue

        track_x1 = os.path.join(base_dir, f'track{X1}.csv')
        track_x2 = os.path.join(base_dir, f'track{X2}.csv')

        # ===== 1) 轨迹文件存在性检查 =====
        if not os.path.exists(track_x1) or not os.path.exists(track_x2):
            print(f"[WARN] skip pair ({X1}, {X2}): missing "
                  f"{'track'+str(X1)+'.csv ' if not os.path.exists(track_x1) else ''}"
                  f"{'track'+str(X2)+'.csv'  if not os.path.exists(track_x2) else ''}")
            skipped += 1
            continue

        # ===== 2) 读取两个轨迹 csv =====
        try:
            df_x1 = pd.read_csv(track_x1)
            df_x2 = pd.read_csv(track_x2)
        except Exception as e:
            print(f"[WARN] skip pair ({X1}, {X2}): read_csv failed -> {e}")
            skipped += 1
            continue
        if df_x1.empty or df_x2.empty:
            print(f"[WARN] skip pair ({X1}, {X2}): empty track")
            skipped += 1
            continue

        # 时间早的为 A，晚的为 B
        A1 = df_x1.iloc[0, 0]
        B1 = df_x2.iloc[0, 0]
        if A1 <= B1:
            A_num, B_num = X1, X2
            A_track_file, B_track_file = track_x1, track_x2
        else:
            A_num, B_num = X2, X1
            A_track_file, B_track_file = track_x2, track_x1

        # 重新按 A/B 读取（保证顺序正确）
        df_A = pd.read_csv(A_track_file)
        df_B = pd.read_csv(B_track_file)
        if df_A.empty or df_B.empty:
            print(f"[WARN] skip pair ({A_num}, {B_num}): empty track")
            skipped += 1
            continue

        track_a_end   = df_A.iloc[-1, 0]
        track_b_start = df_B.iloc[0, 0]

        # ===== 3) 时间是否“相接” =====
        if abs(track_b_start - track_a_end) >= threshold:
            # 不满足时间拼接条件，跳过
            continue

        # ===== 4) allFrames 文件存在性检查并读取 =====
        A_all = os.path.join(base_dir, f'allFramesTrackInten{A_num}.csv')
        B_all = os.path.join(base_dir, f'allFramesTrackInten{B_num}.csv')
        if not os.path.exists(A_all) or not os.path.exists(B_all):
            print(f"[WARN] skip pair ({A_num}, {B_num}): missing "
                  f"{'allFramesTrackInten'+str(A_num)+'.csv ' if not os.path.exists(A_all) else ''}"
                  f"{'allFramesTrackInten'+str(B_num)+'.csv'  if not os.path.exists(B_all) else ''}")
            skipped += 1
            continue

        try:
            df_A_all = pd.read_csv(A_all, header=None)
            df_B_all = pd.read_csv(B_all, header=None)
        except Exception as e:
            print(f"[WARN] skip pair ({A_num}, {B_num}): read allFrames failed -> {e}")
            skipped += 1
            continue

        # ===== 5) 对齐到 A 的结束帧 =====
        if track_a_end not in df_A_all[0].values or track_a_end not in df_B_all[0].values:
            print(f"[WARN] skip pair ({A_num}, {B_num}): cannot align at frame {track_a_end}")
            skipped += 1
            continue

        end_idx   = df_A_all[df_A_all[0] == track_a_end].index[0]
        start_idx = df_B_all[df_B_all[0] == track_a_end].index[0]

        A_keep = df_A_all.iloc[:end_idx + 1]
        B_keep = df_B_all.iloc[start_idx + 1:]

        df_merged = pd.concat([A_keep, B_keep], axis=0, ignore_index=True)
        out_merged = os.path.join(base_dir, f'allFramesTrackInten{A_num}.csv')
        df_merged.to_csv(out_merged, index=False, header=None)

        # ===== 6) 删除 B 的文件（兼容大小写 Track/track）=====
        _safe_remove(os.path.join(base_dir, f'allFramesTrackInten{B_num}.csv'))
        _safe_remove(os.path.join(base_dir, f'Track{B_num}.csv'))
        _safe_remove(os.path.join(base_dir, f'track{B_num}.csv'))

        # ===== 7) 同步更新 message.csv / message_group.csv / merge_log.txt =====
        try:
            msg_path = os.path.join(txt_dir, 'message.csv')
            if os.path.exists(msg_path):
                message = pd.read_csv(msg_path)
                if 'file_name' in message.columns:
                    message = message[message['file_name'] != B_num]
                    message.to_csv(msg_path, index=False)

            mg = pd.read_csv(message_file_path)
            if 'id' in mg.columns:
                mg = mg[mg['id'] != group_id]   # 简化处理：丢弃当前合并组
                mg.to_csv(message_file_path, index=False)

            with open(os.path.join(txt_dir, 'merge_log.txt'), 'a', encoding='utf-8') as f:
                f.write(f'id{group_id}_{A_num}_{B_num}\n')
        except Exception as e:
            print(f"[WARN] update message files failed: {e}")

        num_merged += 1
        print(f"[OK] merged pair ({A_num}, {B_num}) -> {os.path.basename(out_merged)}")

    print(f"[merge_and_delete_files] merged={num_merged}, skipped={skipped}")

                    #------------------------------------------------------------------------




# 6 原始：数字和圆圈的 PNG 图片已保存到
def zhou_plot_points_on_image1(parse_config,params_suffix):#
    # 定义路径
    image_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_image_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'{TIF_NAME}_coordinate_原始.png')
    message_file = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_原始.csv')
    message_group_file = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group_原始.csv')

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
        
        x = row['avg_x']
        y = row['avg_y']
        matching_group = message_group_data[message_group_data['LIST'].apply(lambda x: str(track_id) in x.split())]

        if matching_group.empty:
            circle_radius = 1  # 圆圈半径
            # 不在聚类信息中的点
            ax.plot(x, y, 'ro', markersize=1)  # 红点
            circle = plt.Circle((x, y), circle_radius, color='green', fill=False, lw=1)
            ax.add_patch(circle)

            # 在红色圆圈中心添加数字
            ax.annotate(
                f"ID: {track_id}\n({x:.2f}, {y:.2f})",  # 显示的文本包括 track_id 和坐标
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
                    f"ID: {group_id}\n({x:.2f}, {y:.2f})",  # 显示的文本内容包括 group_id 和坐标
                    xy=(x, y),# 注释箭头的起始位置
                    xytext=(adjusted_x, adjusted_y),# 注释文本的位置
                    fontsize=1, color='white', ha='center', va='center',
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none',pad=1),# 设置文本框的样式
                    arrowprops=dict(arrowstyle="->", color="red", lw=1, shrinkA=0, shrinkB=0),# 设置箭头的样式
                )

                existing_positions.append((adjusted_x, adjusted_y))


    # 保存标记后的图片
    plt.savefig(output_image_path, dpi=900, bbox_inches='tight')
    print(f"6 荧光坐标图片可视化完毕")
    return output_image_path



# 6 后处理：数字和圆圈的 PNG 图片已保存到
def zhou_plot_points_on_image2(parse_config,params_suffix):
    # 定义路径
    image_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_image_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'{TIF_NAME}_coordinate.png')
    message_file = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    message_group_file = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')

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
        
        x = row['avg_x']
        y = row['avg_y']
        matching_group = message_group_data[message_group_data['LIST'].apply(lambda x: str(track_id) in x.split())]

        if matching_group.empty:
            circle_radius = 1  # 圆圈半径
            # 不在聚类信息中的点
            ax.plot(x, y, 'ro', markersize=1)  # 红点
            circle = plt.Circle((x, y), circle_radius, color='green', fill=False, lw=1)
            ax.add_patch(circle)

            # 在红色圆圈中心添加数字
            ax.annotate(
                f"ID: {track_id}\n({x:.2f}, {y:.2f})",  # 显示的文本包括 track_id 和坐标
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
                    f"ID: {group_id}\n({x:.2f}, {y:.2f})",  # 显示的文本包括 group_id 和坐标
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

    x = x - 1
    y = y - 1

    # 创建一个包含三个子图的画布
    plt.figure(figsize=(24, 18))

    # 第一个子图：x 曲线图
    plt.subplot(3, 1, 1)  # 3行1列，第一个子图
    plt.plot(id, x, label='X', color='blue', linewidth=0.7)
    plt.title(f'{npz_file.replace(".npz", "")} - X coordinate', fontsize=20)
    plt.xlabel('TIME', fontsize=16)
    plt.ylabel('X', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 第二个子图：y 曲线图
    plt.subplot(3, 1, 2)  # 3行1列，第二个子图
    plt.plot(id, y, label='Y', color='green', linewidth=0.7)
    plt.title(f'{npz_file.replace(".npz", "")} -Y coordinate', fontsize=20)
    plt.xlabel('TIME', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 第三个子图：强度曲线图
    plt.subplot(3, 1, 3)  # 3行1列，第三个子图
    plt.plot(id, intensity, label='Intensity', color='red', linewidth=0.7)
    plt.title(f'{npz_file.replace(".npz", "")} - Intensity Curve', fontsize=20)
    plt.xlabel('TIME', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 调整子图之间的间距，增加hspace来增加垂直间隔
    plt.subplots_adjust(hspace=0.3)  # 增加子图之间的间隔

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




def move_images_to_group(TIF_NAME, parse_config, params_suffix, save_path):
    # 构建路径
    message_group_csv_path = os.path.join(parse_config.matlab_workdir, TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group_原始.csv')
    png_dir = os.path.join(save_path, params_suffix, 'processing_data', 'png')
    group_dir = os.path.join(save_path, params_suffix, 'processing_data', 'png_团')

    # 读取 CSV 文件
    message_group_df = pd.read_csv(message_group_csv_path)

    # 确保目标文件夹存在
    if not os.path.exists(group_dir):
        os.makedirs(group_dir)

    # 遍历每行数据
    for _, row in message_group_df.iterrows():
        # 提取 id 和 LIST 列
        group_id = row['id']
        list_values = row['LIST'].split()  # 以空格分割字符串

        # 构建目标子目录路径
        group_subdir = os.path.join(group_dir, f'id{group_id}_' + '_'.join(list_values))

        # 创建子目录
        if not os.path.exists(group_subdir):
            os.makedirs(group_subdir)

        # 查找并复制 PNG 文件
        for value in list_values:
            png_filename = f'track{value}.png'
            png_path = os.path.join(png_dir, png_filename)
            if os.path.exists(png_path):
                # 复制文件到新目录
                shutil.copy(png_path, group_subdir)
            else:
                print(f"Warning: {png_filename} not found in {png_dir}")

    print("All images have been moved successfully.")





    




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
    TIF_NAME = os.path.basename(parse_config.filename).replace('.tif','')
    SAVE_PATH = parse_config.save_path
    print('选取的TIRF图片为',TIF_NAME)
    print('测试集保存路径为',SAVE_PATH)
    delete_directories()

    ''''''
    # 获取文件所在文件夹路径
    folder_path = os.path.dirname(parse_config.filename)
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

    # 调用 MATLAB 分析
    analyze_image_with_matlab(parse_config, 1)

    spotTrackingRadius = spotTrackingRadius_values
    gaussFitWidth = gaussFitWidth_values
    frameLength = frameLength_values
    frameGap = frameGap_values
    # 构造超参数后缀
    params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
    params_suffix = ''
    # 生成超参数相关的 SAVE_PATH
    SAVE_PATH = parse_config.save_path
    os.makedirs(os.path.join(parse_config.save_path, params_suffix), exist_ok=True)
    print(f"matlab end, post processing begin")
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            zhou_convert_csv_to_txt_and_calculate_avg(parse_config,params_suffix)
            zhou_save_first_frame_as_png(parse_config)
            drift_distance = zhou_extract_coordinates_to_csv(parse_config,params_suffix)
            group_points(drift_distance, 0.1, parse_config, params_suffix)            
            merge_and_delete_files(parse_config, 50) # 连接轨迹
            zhou_process_files(parse_config,params_suffix)
            zhou_plot_points_on_image1(parse_config,params_suffix)
            output_image_path = zhou_plot_points_on_image2(parse_config,params_suffix)
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
            move_images_to_group(TIF_NAME, parse_config, params_suffix, save_path=SAVE_PATH)


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



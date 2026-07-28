# # -*- coding: utf-8 -*-
# """
# Created on Thu Oct 17 09:48:08 2024

# @author: zhou-
# 1-Install matlabengine, if it is already installed, uninstall it first
# Open cd "matlabroot\extern\engines\python"
# python -m pip install .

# 2-Run the following code

# 2024-11-02: Added u-track, which can be called directly. For details, see utrackInterface.m
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
matplotlib.use('Agg')  # uses interfaceless Agg backend
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

    # hyperparameter range
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
    # Get all .tif files in the folder
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    # Sort file names (alphabetical or numerical)
    tif_files.sort()
    
    # Traverse the sorted files and rename the files
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
                # If it is the second directory, add path inclusion check
                if i == 1:
                    required_subpaths = ['DeepSIFA_main', 'data']
                    if not all(subpath in dir_path for subpath in required_subpaths):
                        print(f"1 路径未满足指定子路径条件，不执行删除操作: {dir_path}")
                        continue

                # Calculate directory depth
                depth = len(os.path.normpath(dir_path).split(os.sep))
                if depth >= 5:
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    # print(f"1 directory {dir_path} has been deleted")
                else:
                    print(f"1 未达到5级目录深度，不执行删除操作: {dir_path}")
            else:
                print(f"1 目录不存在: {dir_path}")


def analyze_image_with_matlab(filename,flag):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    file_basename = os.path.basename(filename).replace('.tif','')

    if ' ' in file_basename:
        file_basename = file_basename.replace(' ', '_')

        # Rename files
        new_filename = os.path.join(os.path.dirname(filename), file_basename + '.tif')
        os.rename(filename, new_filename)
        filename = new_filename  # Update filename to the newly named path
    # print('1212 filename', filename)
    workdir = os.path.join(r'D:\DeepSIFA_main\data\example', file_basename)
    m_source_dir = r'D:\DeepSIFA_main\CreateTrace'  # MATLAB code folder

    if  not flag:
        # defines input parameter dictionary
        input_parameters = {
            # 'spotTrackingRadius': 3, # Maximum jumping distance of highlights, default 3px
            # 'threshold': 40, # Threshold, default 2
            # 'gaussFitWidth': 3, # Gaussian fitting width control, default 3px
            # 'frameLength': 30, # The minimum number of frames the light spot lasts, default 20
            # 'frameGap': 2, #Describe the maximum number of frames in which the light spot may be discontinuous, default 0
            # 'trackMethod': 'default', # Track tracking method
            # 'outputIntegralIntensity': 1, # Whether to calculate the total intensity, default 1
            # 'frameStart': 1, # The starting frame of analysis
            # 'frameEnd': 'inf', # The end frame of analysis
            # 'utrackMotionType': 0 # u-track motion mode
        }
        # # Perform MATLAB analysis
        # eng.addpath(m_source_dir) # Add the MATLAB code folder to the path
        # result = eng.sample(filename, workdir, input_parameters)
        # eng.quit()

    else:
        # Define hyperparameter range
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

        # constructs hyperparameter suffix
        # params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
        params_suffix = ''
        workdir = os.path.join(r'D:\DeepSIFA_main\data\Cache', file_basename, params_suffix)
        os.makedirs(workdir, exist_ok=True)
        # Perform MATLAB analysis
        eng.addpath(m_source_dir)  # Add MATLAB code folder to path
        result = eng.sample(filename, workdir, input_parameters)





        # # Traverse all hyperparameter combinations
        # for spotTrackingRadius in spotTrackingRadius_values:
        #     for gaussFitWidth in gaussFitWidth_values:
        #         for frameLength in frameLength_values:
        #             for frameGap in frameGap_values:
        #                 # Define input parameters
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

        #                 # Construct hyperparameter suffix
        #                 params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
        #                 workdir = os.path.join(r'D:\DeepSIFA_main\data\test', file_basename, params_suffix)

        #                 # Make sure the output directory exists
        #                 os.makedirs(workdir, exist_ok=True)
        #                 # Perform MATLAB analysis
        #                 eng.addpath(m_source_dir) # Add the MATLAB code folder to the path
        #                 result = eng.sample(filename, workdir, input_parameters)


# 1 Convert track CSV file to TXT file and calculate the average of x and y
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




# 2 Integrate x y and 3 brightness into txt file
def zhou_process_files(params_suffix):
    source_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, 'data')
    track_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_dir =    os.path.join(SAVE_PATH, params_suffix, 'row_data')
    os.makedirs(output_dir, exist_ok=True)

    # Find all files starting with 'track'
    for filename in os.listdir(track_dir):
        if filename.startswith('track') and filename.endswith('.txt'):
            track_number = re.search(r'track(\d+)', filename)
            if track_number:
                track_num = track_number.group(1)
                track_file_path = os.path.join(track_dir, filename)

                # Read the last line of the track file
                with open(track_file_path, 'r') as track_file:
                    lines = track_file.readlines()
                    last_line = lines[-1].strip().split(',')
                    if len(last_line) >= 3:  # ensures there are enough columns
                        track_value1 = f"{float(last_line[1]):.3f}"  # Keep 3 decimal places
                        track_value2 = f"{float(last_line[2]):.3f}"  # Keep 3 decimal places

                        # Find the corresponding allFramesTrackInten file
                        all_frames_file = f'allFramesTrackInten{track_num}.csv'
                        all_frames_path = os.path.join(source_dir, all_frames_file)

                        if os.path.exists(all_frames_path):
                            with open(all_frames_path, 'r') as all_frames:
                                all_lines = all_frames.readlines()
                                # Write {number}.txt file
                                output_file_path = os.path.join(output_dir, f'track{track_num}.txt')
                                with open(output_file_path, 'w', newline='') as output_file:
                                    output_writer = csv.writer(output_file)
                                    header = ['id', 'avg_x', 'avg_y', 'x', 'y', 'bg', 'Square_intensity', 'circular_intensity', 'Gaussian_intensity']
                                    output_writer.writerow(header)
                                    # Traverse each line of the allFramesTrackInten file
                                    for i, line in enumerate(all_lines):
                                        all_frames_values = line.strip().split(',')
                                        if len(all_frames_values) >= 3:  # ensures there are enough columns
                                            id = all_frames_values[0]
                                            X = all_frames_values[1]
                                            Y = all_frames_values[2]
                                            BG = float(all_frames_values[3])  # Convert to float
                                            intensity1 = round(float(all_frames_values[4]) - BG * 4, 3)
                                            intensity2 = round(float(all_frames_values[5]) - BG * 9, 3)
                                            intensity3 = round(float(all_frames_values[6]) - BG * 4, 3)

                                            # Write data
                                            output_writer.writerow([id, track_value1, track_value2, X, Y, BG, intensity1, intensity2, intensity3])
    print("2 把X,Y,BG,3种亮度整合到txt文件中")



# 3 Generates a normalized PNG file based on the first frame of the specified TIF file
def zhou_save_first_frame_as_png(tif_path):
    # tif_path =              os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', f'{TIF_NAME}.tif')
    png_output_path =       os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, f'{TIF_NAME}_normalized.png')

    # Read the first frame of the TIF file
    with tifffile.TiffFile(tif_path) as tif:
        first_frame = tif.pages[0].asarray()

    # counts the minimum and maximum values in first_frame and normalizes them
    min_pixel_value = np.min(first_frame)
    max_pixel_value = np.max(first_frame)
    normalized_frame = ((first_frame - min_pixel_value) / (max_pixel_value - min_pixel_value)) * 255
    normalized_frame = normalized_frame.astype(np.uint8)  # converted to 8-bit integer type

    # Save the first frame after normalization as a PNG file
    tifffile.imwrite(png_output_path, normalized_frame)
    print(f"3 PNG文件已保存到 {png_output_path}")





# 4 Extract coordinates from a TXT file in the specified directory and save the results to a CSV file
def zhou_extract_coordinates_to_csv(params_suffix):
    # Define input directory and output file path
    directory_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_file1 = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    output_file2 = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_原始.csv')
    # Get qualified files
    txt_files = [f for f in os.listdir(directory_path) if f.startswith('track') and f.endswith('.txt')]

    results = []  # is used to store the results

    for txt_file in txt_files:
        file_path = os.path.join(directory_path, txt_file)

        # Read file content
        data = np.loadtxt(file_path, delimiter=',')  # Assume the files are comma delimited
        x_values = data[:, 1]  # The first column is x
        y_values = data[:, 2]  # The second column is y

        # Calculate the average
        avg_x = np.mean(x_values)
        avg_y = np.mean(y_values)

        # Calculate the distance from each point to the average point
        distances = np.sqrt((x_values - avg_x) ** 2 + (y_values - avg_y) ** 2)

        # Get the maximum distance
        max_distance = np.max(distances)

        # Extract the number after track as ID
        track_id = txt_file.split('track')[-1].split('.')[0]

        # Save results
        results.append([track_id, avg_x, avg_y, max_distance])

    # Save to CSV file
    df = pd.DataFrame(results, columns=['file_name', 'x', 'y', 'max_distance'])
    df.to_csv(output_file1, index=False)
    df.to_csv(output_file2, index=False)
    percentile_95 = df['max_distance'].quantile(0.95)
    drift_distance = percentile_95
    print(f"4 avg_xy已成功保存到message.csv")
    return drift_distance



# 5 Clustering based on drift_distance
def group_points(drift_distance, drift_distance_factor, params_suffix):
    # Read CSV file
    message_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    data = pd.read_csv(message_path)

    # Extract x, y coordinates and file name
    file_names = data['file_name'].values
    x_coords = data['x'].values
    y_coords = data['y'].values

    groups = []  # stores grouping results
    visited = set()  # records the index of the processed point

    # Calculate the distance between two points
    def calc_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Traverse all points and group them
    for i in range(len(data)):
        if i in visited:
            continue
        
        group = [file_names[i]]  # Current group, first join the current point
        visited.add(i)

        # Find other points whose distance from the current point is less than 0.2
        for j in range(i + 1, len(data)):
            if j in visited:
                continue
            dist = calc_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
            if dist < drift_distance * drift_distance_factor:
                group.append(file_names[j])
                visited.add(j)
        
        # If the group has multiple points, add them to the result list
        if len(group) > 1:
            groups.append(group)

    # Write grouping results to message_group.csv
    output_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')
    with open(output_path, 'w', newline='') as file:
        file.write("id,LIST\n")
        for idx, group in enumerate(groups, start=1):   
            file.write(f"{idx},{' '.join(map(str, group))}\n")

    # Write grouping results to message_group.csv
    output_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group_原始.csv')
    with open(output_path, 'w', newline='') as file:
        file.write("id,LIST\n")
        for idx, group in enumerate(groups, start=1):   
            file.write(f"{idx},{' '.join(map(str, group))}\n")

    print(f"5 drift_distance聚类结果已保存到message_group.csv")



# 5 According to the clustering results, post-process to see whether the trajectories need to be connected
def merge_and_delete_files(threshold=50):
    """
    Merge the A and B trajectory files when the configured condition is met.

    ``threshold`` controls whether the two trajectories are eligible for
    merging. The merged data replace the A file and the redundant B file is
    removed.
    """
    base_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, 'data')
    # reads message.csv file
    message_file_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')
    message_data = pd.read_csv(message_file_path)

    # Traverse each row in message.csv and obtain the value of the LIST column
    for index, row in message_data.iterrows():
        LIST = row['LIST']  # Assume 'LIST' is a column name in message.csv
        if isinstance(LIST, str):  # ensures that LIST is a string
            # Use spaces to split LIST and check whether the number of split parts is 2
            split_list = LIST.split(' ')
            if len(split_list) == 2:  # ensures there are only two elements
                X1, X2 = map(int, LIST.split(' '))  # Assume that the LIST is a comma-separated number
                # Find allFramesTrackIntenX1.csv and allFramesTrackIntenX2.csv files
                file_x1 = os.path.join(base_dir, f'allFramesTrackInten{X1}.csv')
                file_x2 = os.path.join(base_dir, f'allFramesTrackInten{X2}.csv')

                # Read file content
                df_x1 = pd.read_csv(file_x1)
                df_x2 = pd.read_csv(file_x2)

                # Get the number in the first row and column
                A1 = df_x1.iloc[0, 0]
                B1 = df_x2.iloc[0, 0]

                # Determine which file has a smaller number at the beginning of the first line, the smaller one is A, and the larger one is B
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

                # reads trackA1.csv and trackB1.csv
                track_a1_file = os.path.join(base_dir, f'track{A_num}.csv')
                track_b1_file = os.path.join(base_dir, f'track{B_num}.csv')

                df_track_a1 = pd.read_csv(track_a1_file)
                df_track_b1 = pd.read_csv(track_b1_file)


                track_a1_end_num = df_track_a1.iloc[-1, 0]  # The first column of numbers in the last row of trackA1.csv
                track_b1_start_num = df_track_b1.iloc[0, 0]  # The first column of numbers in the first row of trackB1.csv

                if abs(track_b1_start_num - track_a1_end_num) < threshold:

                    # -------------------------------Merge files----------------------------------
                    df_A = pd.read_csv(A_file, header=None)
                    df_B = pd.read_csv(B_file, header=None)

                    # Get the first column number of the last row of file A
                    A_num_end = df_A.iloc[-1, 0]

                    # Find the corresponding line from the B file based on A_num_end
                    B_part = df_B[df_B.iloc[:, 0] > A_num_end]

                    # # Merge A and B
                    # df_C = pd.concat([df_A, B_part])
                    df_C = pd.concat([df_A, B_part], axis=0, ignore_index=True)

                    # Save as allFramesTrackIntenA1.csv
                    output_file = os.path.join(base_dir, f'allFramesTrackInten{A_num}.csv')
                    df_C.to_csv(output_file, index=False, header=None)

                    # Delete B file
                    os.remove(os.path.join(base_dir, f'allFramesTrackInten{B_num}.csv'))
                    os.remove(os.path.join(base_dir, f'Track{B_num}.csv'))
                    print(f"合并成功: {A_file} 和 {B_file} 合并为 {output_file} 并删除 {B_file}")
                    #-------------------------------------------------------------------------------
                    

                    # ---------Update the message.csv file and delete the corresponding row in the file_name column---------
                    txt_dir = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
                    message = pd.read_csv(os.path.join(txt_dir, 'message.csv'))
                    updated_message_data = message[message['file_name'] != B_num]
                    updated_message_file_path = os.path.join(txt_dir, 'message.csv')
                    updated_message_data.to_csv(updated_message_file_path, index=False)

                    # Update the message_group.csv file and delete the corresponding row in the file_name column
                    message_group_data = pd.read_csv(message_file_path)
                    updated_message_group_data = message_group_data[~message_group_data['LIST'].apply(lambda x: str(B_num) in str(x))]
                    updated_message_group_file_path = os.path.join(txt_dir, 'message_group.csv')
                    updated_message_group_data.to_csv(updated_message_group_file_path, index=False)

                    print(f"合并成功: {A_file} 和 {B_file} 合并为 {output_file}")
                    print(f"已更新 message.csv 为 {updated_message_file_path}")
                    print(f"已更新 message_group.csv 为 {updated_message_group_file_path}")
                    #------------------------------------------------------------------------




# 6 Original: PNG image of numbers and circles saved to
def zhou_plot_points_on_image1(params_suffix):
    # define path
    image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'{TIF_NAME}_coordinate_原始.png')
    message_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_原始.csv')
    message_group_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group_原始.csv')

    # read data
    image = tifffile.imread(image_path)
    message_data = pd.read_csv(message_file)
    message_group_data = pd.read_csv(message_group_file)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')  # Show picture
    existing_positions = []  # is used to record the digital placement position

    # Traverse the points in message.csv
    # records the track_id that has been drawn
    drawn_ids = set()
    drawn_group_ids = set()  # drawn group_id

    for _, row in message_data.iterrows():
        track_id = int(row['file_name'])
        
        # Check whether the track_id has been drawn
        if track_id in drawn_ids:
            continue
        
        drawn_ids.add(track_id)
        
        x = row['x']
        y = row['y']
        matching_group = message_group_data[message_group_data['LIST'].apply(lambda x: str(track_id) in x.split())]

        if matching_group.empty:
            circle_radius = 1  # circle radius
            # Points not in the clustering information
            ax.plot(x, y, 'ro', markersize=1)  # red dot
            circle = plt.Circle((x, y), circle_radius, color='green', fill=False, lw=1)
            ax.add_patch(circle)

            # Add numbers in the center of the red circle
            ax.annotate(
                track_id,  # The number displayed is the track_id of the point
                xy=(x, y), xytext=(x, y),  # Number is in the center of the circle
                fontsize=1, color='white', ha='center', va='center',
                bbox=dict(facecolor='green', alpha=0.8, edgecolor='none', pad=1)  # Number background is red
            )
        else:
            # points in the clustering information, distributed around a circle
            group_list = matching_group.iloc[0]['LIST']
            if isinstance(group_list, str):
                group_items = list(map(int, group_list.split()))
            else:
                group_items = []

            # Draw clustered points
            for idx, group_id in enumerate(group_items):
                circle_radius = 1.5
                # Check whether group_id has been drawn
                if group_id in drawn_group_ids:
                    continue
                drawn_group_ids.add(group_id)
                angle = idx * (2 * np.pi / len(group_items))  # distributed in order
                adjusted_x = x + circle_radius * np.cos(angle)
                adjusted_y = y + circle_radius * np.sin(angle)
                # Draw a red circle
                circle = plt.Circle((x, y), circle_radius, color='red', fill=False, lw=1)  # blue circle
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


    # Save the marked picture
    plt.savefig(output_image_path, dpi=900, bbox_inches='tight')
    print(f"6 荧光坐标图片可视化完毕")
    return output_image_path



# 6 Post-processing: PNG images of numbers and circles saved to
def zhou_plot_points_on_image2(params_suffix):
    # define path
    image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}')
    output_image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'{TIF_NAME}_coordinate.png')
    message_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message.csv')
    message_group_file = os.path.join('D:\\', 'DeepSIFA_main', 'data', 'Cache', TIF_NAME, params_suffix, f'txt_{TIF_NAME}', 'message_group.csv')

    # read data
    image = tifffile.imread(image_path)
    message_data = pd.read_csv(message_file)
    message_group_data = pd.read_csv(message_group_file)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')  # Show picture
    existing_positions = []  # is used to record the digital placement position

    # Traverse the points in message.csv
    # records the track_id that has been drawn
    drawn_ids = set()
    drawn_group_ids = set()  # drawn group_id

    for _, row in message_data.iterrows():
        track_id = int(row['file_name'])
        
        # Check whether the track_id has been drawn
        if track_id in drawn_ids:
            continue
        
        drawn_ids.add(track_id)
        
        x = row['x']
        y = row['y']
        matching_group = message_group_data[message_group_data['LIST'].apply(lambda x: str(track_id) in x.split())]

        if matching_group.empty:
            circle_radius = 1  # circle radius
            # Points not in the clustering information
            ax.plot(x, y, 'ro', markersize=1)  # red dot
            circle = plt.Circle((x, y), circle_radius, color='green', fill=False, lw=1)
            ax.add_patch(circle)

            # Add numbers in the center of the red circle
            ax.annotate(
                track_id,  # The number displayed is the track_id of the point
                xy=(x, y), xytext=(x, y),  # Number is in the center of the circle
                fontsize=1, color='white', ha='center', va='center',
                bbox=dict(facecolor='green', alpha=0.8, edgecolor='none', pad=1)  # Number background is red
            )
        else:
            # points in the clustering information, distributed around a circle
            group_list = matching_group.iloc[0]['LIST']
            if isinstance(group_list, str):
                group_items = list(map(int, group_list.split()))
            else:
                group_items = []

            # Draw clustered points
            for idx, group_id in enumerate(group_items):
                circle_radius = 1.5
                # Check whether group_id has been drawn
                if group_id in drawn_group_ids:
                    continue
                drawn_group_ids.add(group_id)
                angle = idx * (2 * np.pi / len(group_items))  # distributed in order
                adjusted_x = x + circle_radius * np.cos(angle)
                adjusted_y = y + circle_radius * np.sin(angle)
                # Draw a red circle
                circle = plt.Circle((x, y), circle_radius, color='red', fill=False, lw=1)  # blue circle
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


    # Save the marked picture
    plt.savefig(output_image_path, dpi=900, bbox_inches='tight')
    print(f"6 荧光坐标图片可视化完毕")
    return output_image_path







# Generate test set
# Generate test set
# Generate test set
# Generate test set

# 1. Generate data storage directory
def create_directory(params_suffix):
    directory_path = os.path.join(SAVE_PATH, params_suffix, 'row_data')
    os.makedirs(directory_path, exist_ok=True)
    print(f"7 测试集目录 '{directory_path}' 已成功创建。")
    return directory_path



# 2. Generate .txt suffix and add _bad _good
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



# 3. Move files and clean directories
def move_and_clean(params_suffix):
    good_dir = os.path.join(SAVE_PATH, params_suffix, 'good')
    bad_dir = os.path.join(SAVE_PATH, params_suffix, 'bad')
    target_dir = os.path.join(SAVE_PATH, params_suffix, 'row_data')

    # Defines the operation for moving files
    for src_dir in [good_dir, bad_dir]:
        if os.path.exists(src_dir):
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dest_file = os.path.join(target_dir, filename)
                if os.path.isfile(src_file):
                    shutil.move(src_file, dest_file)
            os.rmdir(src_dir)  # Delete empty directories

    print("9 文件已成功移动并删除目录。")




# 4. alphaK10 obtains brightness information
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




# 5. Convert txt files to npz files for storage
def convert_txt_to_npz(params_suffix):
    directory = os.path.join(SAVE_PATH, params_suffix, 'row_data')

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, os.path.splitext(filename)[0] + '.npz')

            # Open and process txt files and save them as npz files
            with open(file_path, 'r') as file:
                next(file)  # Skip the first line
                lines  = file.readlines()[:-1]  # Skip the last line
                combined_data = []

                for line in lines :
                    splitted_line = re.split(',|\t| ', line.strip())
                    combined_data.append([splitted_line[0], splitted_line[3],splitted_line[4], splitted_line[-1]])#first and last lines

                combined_data = np.array(combined_data)
                assert combined_data.shape[1] == 4, "Data shape is not (num, 2)"

                # Save as .npz file
                np.savez(output_path, combined_data=combined_data)

    print("10 把txt文件都转化为npz文件存储")




# 6. Normalization
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
    # 7th code segment: Resize to 1024 and Gaussian smoothing
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
            # Gaussian smoothing
            sigma = SIGMA
            interpolated_data = gaussian_filter1d(interpolated_data, sigma)
            save_path = os.path.join(target_dir, filename)
            np.savez(save_path, data=interpolated_data)

    print("12 时间序列Resize to 1024 and Gaussian smoothing completed.")


def delete_directories_npz(params_suffix):
    directories_to_delete = [os.path.join(SAVE_PATH, params_suffix, '归一化后npz')]

    for i, dir_path in enumerate(directories_to_delete):
            if os.path.exists(dir_path):
                # Calculate directory depth
                depth = len(os.path.normpath(dir_path).split(os.sep))
                if depth >= 5:
                    # for item in os.listdir(dir_path):
                    #     item_path = os.path.join(dir_path, item)
                    #     if os.path.isfile(item_path) or os.path.islink(item_path):
                    #         os.unlink(item_path)
                    #     elif os.path.isdir(item_path):
                    #         shutil.rmtree(item_path)
                    shutil.rmtree(dir_path)
                    # print(f"1 directory {dir_path} has been deleted")
                else:
                    print(f"12 未达到5级目录深度，不执行删除操作: {dir_path}")
            else:
                print(f"12 目录不存在: {dir_path}")


def generate_csv(params_suffix):
    # Section 8: Generate labels and csv files based on _bad.npz or _good.npz
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




def process_file_1(npz_file, save_path, params_suffix):# Visualization npz after normalization
    # defines the path of npz file
    source_dir = os.path.join(save_path, params_suffix, '归一化后npz')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    npz_path = os.path.join(source_dir, npz_file)
    npz_data = np.load(npz_path)
    data = npz_data['combined_data'][-1]

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']

    # Create x-axis coordinate
    x = range(len(data))
    plt.figure(figsize=(18, 6))  # Set graphic size
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
    gc.collect()  # Manually clear memory


def process_file_2(npz_file, save_path, params_suffix):# Visualized normalized interpolation npz_{NUM_dimension}_Gaussian smoothing {SIGMA}
    source_dir = os.path.join(save_path, params_suffix, 'processing_data')
    file_path = os.path.join(source_dir, npz_file)
    npz_data = np.load(file_path)
    data = npz_data['data']

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']

    # Create x-axis coordinate
    x = range(len(data))
    plt.figure(figsize=(18, 6))  # Set graphic size
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
    gc.collect()  # Manually clear memory





# Modify the length of the canvas, saved address, multi-threaded operation
def process_file_3(npz_file, save_path, params_suffix):
    # defines the path of npz file
    directory = os.path.join(save_path, params_suffix, 'row_data')
    source_dir = os.path.join(save_path, params_suffix, 'processing_data')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    npz_path = os.path.join(directory, npz_file)
    # Load npz file
    npz_data = np.load(npz_path)
    
    # extracts the stored data, assuming the key is 'combined_data'
    data = npz_data['combined_data']
    id = data[:, 0].astype(float)
    x = data[:, 1].astype(float)
    y = data[:, 2].astype(float)
    intensity = data[:, 3].astype(float)

    # Create a canvas containing three subgraphs
    plt.figure(figsize=(24, 18))

    # First subgraph: x curve
    plt.subplot(3, 1, 1)  # 3 rows and 1 column, the first subgraph
    plt.plot(id, x, label='X', color='blue')
    plt.title(f'{npz_file.replace(".npz", "")} - X Curve', fontsize=20)
    plt.xlabel('ID', fontsize=16)
    plt.ylabel('X', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Second subgraph: y curve
    plt.subplot(3, 1, 2)  # 3 rows and 1 column, second sub-picture
    plt.plot(id, y, label='Y', color='green')
    plt.title(f'{npz_file.replace(".npz", "")} - Y Curve', fontsize=20)
    plt.xlabel('ID', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # The third sub-picture: intensity curve
    plt.subplot(3, 1, 3)  # 3 rows and 1 column, the third sub-picture
    plt.plot(id, intensity, label='Intensity', color='red')
    plt.title(f'{npz_file.replace(".npz", "")} - Intensity Curve', fontsize=20)
    plt.xlabel('ID', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Save image
    save_path = os.path.join(source_dir, 'png', npz_file.replace('.npz', '.png'))
    plt.savefig(save_path, dpi=250)
    plt.close()  # Close the image to prevent excessive memory usage




def copy_image_to_directory(output_image_path, directory):
    filename = os.path.basename(output_image_path)
    # splicing target path
    target_directory = os.path.join(os.path.dirname(directory), filename)
    
    # Copy the picture to the target path
    shutil.copy(output_image_path, target_directory)
    print(f"15 荧光坐标图片已复制到: {target_directory}")
    return target_directory
    




if __name__ == "__main__":
    DATA = 'MLKL'
    NUM_dimension = 1024
    SIGMA = 1
    # hyperparameter range
    parse_config = get_cfg()
    threshold_values = parse_config.threshold
    frameStart_values = parse_config.frameStart
    frameEnd_values = parse_config.frameEnd
    spotTrackingRadius_values = parse_config.spotTrackingRadius_values
    gaussFitWidth_values = parse_config.gaussFitWidth_values
    frameLength_values = parse_config.frameLength_values
    frameGap_values = parse_config.frameGap_values
    # Print hyperparameters
    print("threshold:", threshold_values)
    print("frameStart:", frameStart_values)
    print("frameEnd:", frameEnd_values) 
    print("spotTrackingRadius_values:", spotTrackingRadius_values)
    print("gaussFitWidth_values:", gaussFitWidth_values)
    print("frameLength_values:", frameLength_values)
    print("frameGap_values:", frameGap_values)


    # Get the folder path where the file is located
    folder_path = parse_config.filename
    for filename in os.listdir(folder_path):
        # constructs a complete file path
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and ' ' in filename:
            # replace spaces with underscores
            new_filename = filename.replace(' ', '_')
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename file
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} -> {new_file_path}')


    # Traverse all .tif files in a folder
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    for idx, filename in enumerate(tif_files):
        file_path = os.path.join(folder_path, filename)
        TIF_NAME = filename.replace('.tif', '')
        SAVE_PATH = os.path.join(parse_config.save_path, TIF_NAME)
        print('选取的TIRF图片为',TIF_NAME)
        print('测试集保存路径为',SAVE_PATH)

        ''''''
        delete_directories(SAVE_PATH)
        # calls MATLAB analysis
        analyze_image_with_matlab(file_path, 1)
        spotTrackingRadius = spotTrackingRadius_values
        gaussFitWidth = gaussFitWidth_values
        frameLength = frameLength_values
        frameGap = frameGap_values
        # constructs hyperparameter suffix
        params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
        params_suffix = ''
        os.makedirs(os.path.join(SAVE_PATH, params_suffix), exist_ok=True)        # Generate hyperparameter related SAVE_PATH
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
                # generate test set and curve png
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
                num_workers = cpu_count()  # Get the number of CPU cores

                # source_dir = os.path.join(SAVE_PATH, params_suffix, 'npz after normalization')
                # npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
                # # Use partial to bind the params_suffix parameter
                # process_with_suffix = partial(process_file_1, save_path=SAVE_PATH, params_suffix=params_suffix)
                # # Create a process pool and pass the binding function
                # with Pool(processes=num_workers) as pool:
                #     list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))



                source_dir = os.path.join(SAVE_PATH, params_suffix, 'processing_data')
                npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
                # Use partial to bind the params_suffix parameter
                process_with_suffix = partial(process_file_2, save_path=SAVE_PATH, params_suffix=params_suffix)
                # Create a process pool and pass the binding function
                with Pool(processes=num_workers) as pool:
                    list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

                
                source_dir = os.path.join(SAVE_PATH, params_suffix, 'row_data')
                npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
                # Use partial to bind the params_suffix parameter
                process_with_suffix = partial(process_file_3, save_path=SAVE_PATH, params_suffix=params_suffix)
                # Create a process pool and pass the binding function
                with Pool(processes=num_workers) as pool:
                    list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))
                print("14 时间序列可视化完成")
                target_directory = copy_image_to_directory(output_image_path, directory_path)
        print('后处理-第二阶段-生成测试集 完成')
        print('-----------------------------------------------------')




    # Find the best hyperparameters
    # for spotTrackingRadius in spotTrackingRadius_values:
    #     for gaussFitWidth in gaussFitWidth_values:
    #         for frameLength in frameLength_values:
    #             for frameGap in frameGap_values:
    #                 # Construct hyperparameter suffix
    #                 params_suffix = f"spotRadius{spotTrackingRadius}_guassWidth{gaussFitWidth}_Length{frameLength}_Gap{frameGap}"
    #                 # Generate SAVE_PATH related to hyperparameters
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
    #                 print('Phase 1 - Generation of fluorescence curve completed')
    #                 print('-----------------------------------------------------')



    #                 # Generate test set and curve png
    #                 matplotlib.use('Agg')
    #                 directory_path = create_directory(params_suffix)
    #                 rename_and_append_suffix(params_suffix)
    #                 move_and_clean(params_suffix)
    #                 get_brightness_info(params_suffix)
    #                 convert_txt_to_npz(params_suffix)
    #                 normalize_data(params_suffix)
    #                 resize_and_smooth(params_suffix)
    #                 generate_csv(params_suffix)
    #                 num_workers = cpu_count() # Get the number of CPU cores
    #                 source_dir = os.path.join(SAVE_PATH, params_suffix, 'original data')
    #                 npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    #                 # Use partial to bind the params_suffix parameter
    #                 process_with_suffix = partial(process_file_2, save_path=SAVE_PATH, num_dimension=NUM_dimension, sigma=SIGMA, params_suffix=params_suffix)

    #                 # Create a process pool and pass the binding function
    #                 with Pool(processes=num_workers) as pool:
    #                     list(tqdm(pool.imap(process_with_suffix, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))
    #                 print("14 time series visualization completed")
    #                 target_directory = copy_image_to_directory(output_image_path, directory_path)
    #                 print('Second phase - test set generation completed')
    #                 print('-----------------------------------------------------')







# #import trackingSM
# import os
# import matlab.engine

# eng = matlab.engine.start_matlab() #Start matlab

# workdir = r'D:\DeepSIFA_main\data\MLKL\S55C_label_alex55_15nM' #Data output directory
# filename = r'D:\DeepSIFA_main\data\MLKL\S55C_label_alex55_15nM.tif' #Picture file to be analyzed
# m_source_dir = r'D:\DeepSIFA_main\CreateTrace' #mainWithoutUI including the matlab code folder

# # Create a Python dictionary to simulate MATLAB struct, used to describe input parameters
# # The value must be an integer or character type
# input_parameters = {
#     'spotTrackingRadius': 3, # Maximum jumping distance of highlights, default 3px
#     'threshold': 2, # Threshold, default 2
#     'gaussFitWidth': 3, # Gaussian fitting width control, 2*gaussFitWidth-1, default 3px
#     'frameLength': 30, # The minimum number of frames the light spot lasts, default 20
#     'frameGap': 2, #Describe the maximum number of frames in which the light spot may be discontinuous, default 0
#     'trackMethod': 'default', # default, or u-track: call u-track directly
#     'outputIntegralIntensity': 1, #Whether to calculate the total intensity through Gaussian fitting function integration, the speed is relatively slow, the default is 1
#     'frameStart': 1, # The starting frame to be analyzed, default 1
#     'frameEnd': 'inf', #The end frame that needs to be analyzed is the last frame of the movie by default and has not been added yet! ! !
#     'utrackMotionType': 0 # This parameter is the three motion modes defined in utrack. I don't know the details. See the userguide.
#                            # 0-linear motion 1- linear+random motion with constant vel 
#                            # 2- linear+random motion. movement along a straight line but with the possibility of immediate direction reversal
    
# }

# # # Check whether the path exists
# # if not os.path.exists(workdir):
# # print(f"The working directory does not exist: {workdir}")
# # if not os.path.exists(filename):
# # print(f"Picture file does not exist: {filename}")
# # if not os.path.exists(m_source_dir):
# # print(f"MATLAB code directory does not exist: {m_source_dir}")

# eng.addpath(m_source_dir) #Open the folder
# result = eng.sample(filename, workdir, input_parameters)
# print(result) #1-correct 0-wrong
# eng.quit()


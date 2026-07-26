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
            # Calculate directory depth
            depth = len(os.path.normpath(dir_path).split(os.sep))
            if depth >= 5:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                # print("0 directory deleted")
            else:
                print("0 未达到8级目录深度，不执行删除操作。")
        else:
            print(f"0 目录不存在: {dir_path}")


def analyze_image_with_matlab(filename):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    # Set path
    file_basename = os.path.basename(filename).replace('.tif','')
    workdir = os.path.join(r'D:\DeepSIFA_main\data\测试', file_basename)
    # workdir = r'D:\DeepSIFA_main\data\test\S55C_label_alex55_1.5nM' #Data output directory
    m_source_dir = r'D:\DeepSIFA_main\CreateTrace'  # MATLAB code folder

    # defines input parameter dictionary
    input_parameters = {
        'spotTrackingRadius': 3,  # Maximum jumping distance of highlights, default 3px
        'threshold': 2,  # threshold, default 2
        'gaussFitWidth': 3,  # Gaussian fitting width control, default 3px
        'frameLength': 30,  # Minimum number of frames for light spot duration, default 20
        'frameGap': 2,  # describes the maximum number of frames in which the light spot may be discontinuous, default 0
        'trackMethod': 'default',  # trajectory tracking method
        'outputIntegralIntensity': 1,  # Whether to calculate the total intensity, default 1
        'frameStart': 1,  # Starting frame for analysis
        'frameEnd': 'inf',  # Termination frame analyzed
        'utrackMotionType': 0  # u-track sports mode
    }

    # # Check whether the path exists
    # if not os.path.exists(workdir):
    #     print(f"The working directory does not exist: {workdir}")
    # if not os.path.exists(filename):
    #     print(f"Image file does not exist: {filename}")
    # if not os.path.exists(m_source_dir):
    #     print(f"MATLAB code directory does not exist: {m_source_dir}")

    # Perform MATLAB analysis
    eng.addpath(m_source_dir)  # Add MATLAB code folder to path
    result = eng.sample(filename, workdir, input_parameters)
    # print(result) # Output result 1-correct, 0-error
    eng.quit()


# 1 Convert track CSV file to TXT file and calculate the average of x and y
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




# 2 Integrate x y and 3 brightness into txt file
def zhou_process_files():
    source_dir =    os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'data')
    track_dir =     os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'txt_{TIF_NAME}')
    output_dir =    os.path.join('D:\\', 'DeepSIFA_main', 'data', 'MLKL','test', f'v{NUM}', '原始数据')
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

                                    # Traverse each line of the allFramesTrackInten file
                                    for i, line in enumerate(all_lines):
                                        all_frames_values = line.strip().split(',')
                                        if len(all_frames_values) >= 3:  # ensures there are enough columns
                                            all_value1 = all_frames_values[1]
                                            all_value2 = all_frames_values[2]
                                            all_value3 = all_frames_values[3]

                                            # Write data
                                            output_writer.writerow([i + 1, track_value1, track_value2, all_value1, all_value2, all_value3])
    print("2 zhou_把x y和3种亮度整合到txt文件中")



# 3 Generates a normalized PNG file based on the first frame of the specified TIF file
def zhou_save_first_frame_as_png():
    tif_path =              os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', f'{TIF_NAME}.tif')
    png_output_path =       os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'{TIF_NAME}_normalized.png')

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
    print(f"3 zhou_PNG 文件已保存到 {png_output_path}")



# 4 Draw red points on a PNG image based on the average x and y coordinates in the TXT file.
def zhou_plot_points_on_image():
    # Define image path and TXT file directory
    image_path =        os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'{TIF_NAME}_normalized.png')
    txt_directory =     os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'txt_{TIF_NAME}')
    output_image_path = os.path.join('D:\\', 'DeepSIFA_main', 'data', '测试', TIF_NAME, f'{TIF_NAME}_with_points.png')
    # Read PNG images
    image = tifffile.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')  # Show picture
    
    # Traverse each TXT file
    txt_files = [os.path.join(txt_directory, f) for f in os.listdir(txt_directory) if f.endswith('.txt')]
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            lines = file.readlines()  # Read all lines
            line = lines[-1]  # Get the last line
            # Find rows containing 'Average'
            if 'Average' in line:
                # Split lines with commas and extract x and y coordinates
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    x = float(parts[1])  # Second value (x coordinate)
                    y = float(parts[2])  # third value (y coordinate)
                    # Draw red dots on the picture
                    ax.plot(x, y, 'ro', markersize=5)  # 'ro' means red point

    # Save the marked picture
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"4 zhou_带标记的 PNG 图片已保存到 {output_image_path}")



# 5 Extracts coordinates from a TXT file in the specified directory and saves the results to a CSV file.
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
                        x = parts[1].strip()  # Get the data after the first comma
                        y = parts[2].strip()  # Get the data after the second comma
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



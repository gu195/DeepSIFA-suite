import os
import shutil
import pandas as pd
import numpy as np
import re
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import gc

DATA = 'mlkl'
NUM = '1'
NUM_dimension = 1024
SIGMA = 1



# 1 Generate data storage directory
# Define directory path
directory_path = './data/{}/test/v{}/原始数据'.format(DATA,NUM)
# Create directory
os.makedirs(directory_path, exist_ok=True)
print(f"1目录 '{directory_path}' 已成功创建。")



# # 2 Generate .txt suffix, add _bad _good
import os
# defines the directory to be processed
bad_directory = './data/{}/test/v{}/bad'.format(DATA, NUM)
good_directory = './data/{}/test/v{}/good'.format(DATA, NUM)
def rename_files(directory, suffix):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Make sure it is a file and not a directory
        if os.path.isfile(file_path):
            # Separate file name and suffix
            name, ext = os.path.splitext(filename)
            if ext:  # if there is a suffix
                new_file_name = f"{name}{suffix}{ext}"
            else:  # if there is no suffix
                new_file_name = f"{name}{suffix}.txt"
            new_file_path = os.path.join(directory, new_file_name)
            os.rename(file_path, new_file_path)
            # print(f"Renamed: {file_path} -> {new_file_path}")

# handles bad directory
rename_files(bad_directory, '_bad')
# handles good directory
rename_files(good_directory, '_good')
print("2 生成.txt后缀,加上_bad _good")





# 3 mobile file
# Define directory path
good_dir = f'./data/{DATA}/test/v{NUM}/good'
bad_dir = f'./data/{DATA}/test/v{NUM}/bad'
target_dir = f'./data/{DATA}/test/v{NUM}/原始数据'
# Function to move files
def move_files(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        if os.path.isfile(src_file):
            shutil.move(src_file, dest_file)

if os.path.exists(good_dir):
    move_files(good_dir, target_dir)
    os.rmdir(good_dir)

if os.path.exists(bad_dir):
    move_files(bad_dir, target_dir)
    os.rmdir(bad_dir)
print("3文件已成功移动并删除目录。")




# 4 alphaK10 obtains brightness information
if DATA == 'alphak10':
    # Define directory path
    directory = './data/{}/test/v{}/原始数据'.format(DATA,NUM)
    # Read all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    # Traverse each file
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)

        # Calculate the value of the second column minus the third column and write the result to the sixth column
        df[5] = df[1] - df[2]
        # Add a column named brightness
        df.columns = ['Time[s]', 'CH1', 'BGND1', 'CH2', 'BGND2', 'brightness']
        # Write the results back to the original file
        df.to_csv(file_path, sep="\t", index=False, header=True)
        # print(f"Processed file: {file}")
    print("4 alphaK10获取亮度信息.")




# 5 Convert txt files to npz files for storage
# This code reads a file, extracts the first and last column data of each row, and saves it in .npz format.
# Define directory path
directory = './data/{}/test/v{}/原始数据'.format(DATA, NUM)
def read_and_save_columns(file_path, output_path):
    with open(file_path, 'r') as file:
        # Skip the first line
        next(file)
        combined_data = []

        for line in file:
            splitted_line = line.strip()
            splitted_line = re.split(',|\t', splitted_line)
            combined_data.append([splitted_line[0], splitted_line[-1]])

        combined_data = np.array(combined_data)
        # ensures that the data shape is (number, 2)
        assert combined_data.shape[1] == 2, "Data shape is not (num, 2)"
        # Save as .npz file
        np.savez(output_path, combined_data=combined_data)

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, os.path.splitext(filename)[0] + '.npz')
        read_and_save_columns(file_path, output_path)
print("5 把txt文件都转化为npz文件存储.")




# 6.Normalization
# Set the paths of original data and normalized data
source_dir = './data/{}/test/v{}/原始数据'.format(DATA, NUM)
target_dir = './data/{}/test/v{}/归一化后npz'.format(DATA, NUM)

# Make sure the target directory exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Traverse all npz files in the original data directory
for filename in os.listdir(source_dir):
    if filename.endswith('.npz'):
        # Read each npz file
        file_path = os.path.join(source_dir, filename)
        npz_data = np.load(file_path)
        data = npz_data['combined_data']
        data = np.transpose(data, (1, 0))
        try:
            data = data.astype(np.float64)
        except ValueError:
            print("数据转换失败：数组包含非数值字符串。")
        row = data[1, :]

        normalized_row = (row - row.min()) / (row.max() - row.min())
        # print(normalized_row)

        data[1, :] = normalized_row
        # Save the normalized data
        save_path = os.path.join(target_dir, filename)
        np.savez(save_path, combined_data=data)
print("6.归一化.")





## 7.resize to 1024, and Gaussian smoothing see Gaussian smoothing.py
source_dir = './data/{}/test/v{}/归一化后npz/'.format(DATA, NUM)
target_dir = './data/{}/test/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA, NUM,NUM_dimension,SIGMA)

# Make sure the target directory exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Traverse all npz files in the original data directory
for filename in os.listdir(source_dir):
    if filename.endswith('.npz'):
        # Read each npz file
        file_path = os.path.join(source_dir, filename)
        npz_data = np.load(file_path)
        data = npz_data['combined_data']
        # print(data.shape)
        original_data = data[1, :]
        # Assume original_data is the original time series data and original_data is a one-dimensional array
        # creation time point
        original_indices = np.linspace(0, 1, len(original_data))
        new_indices = np.linspace(0, 1, NUM_dimension)
        # Create interpolation function
        interpolation_function = interp1d(original_indices, original_data, kind='linear')
        # Apply interpolation
        interpolated_data = interpolation_function(new_indices)
        # Set the standard deviation (sigma) of Gaussian smoothing
        sigma = SIGMA
        # Gaussian smoothing of time series data
        interpolated_data = gaussian_filter1d(interpolated_data, sigma)
        data_ = interpolated_data
        # Save the normalized data
        save_path = os.path.join(target_dir, filename)
        np.savez(save_path, data=data_)
print("7.resize到1024，并且高斯平滑.")






# 8 Generate labels and csv files based on _bad.npz or _good.npz
directory = './data/{}/test/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA, NUM,NUM_dimension,SIGMA)
data = []
# Traverse the directory and find all txt files
for file in os.listdir(directory):
    if "bad" in file:
        data.append({'file_name': file, 'label': 0})
    elif "good" in file:
        data.append({'file_name': file, 'label': 1})
    else:
        data.append({'file_name': file, 'label': -1})

# Create DataFrame from list
df = pd.DataFrame(data, columns=['file_name', 'label'])
csv_file_path = './data/{}/test/v{}/{}.csv'.format(DATA, NUM,len(os.listdir(directory)))
df.to_csv(csv_file_path, index=False)
print("8 生成csv文件.",f"CSV file saved at {csv_file_path}")






# # 9 Generate png image
matplotlib.use('Agg')
source_dir = './data/{}/test/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA, NUM,NUM_dimension,SIGMA)
# Get all npz files in the original data directory
npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]

# Use tqdm to add a progress bar
for filename in tqdm(npz_files, desc='Processing', unit='file'):
    # Read each npz file
    file_path = os.path.join(source_dir, filename)
    npz_data = np.load(file_path)
    data = npz_data['data']
    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']
    # Create x-axis coordinate
    x = range(len(data))
    plt.figure(figsize=(24, 6))  # Set the size of the graphic to 10 inches wide and 5 inches high
    
    # Draw curves and scatter plots
    plt.plot(x, data, label='Data Line', linewidth=0.5)  # sets line width to 1
    plt.scatter(x, data, label='Data Points', s=0.5)  # Draw scatter points
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('{}'.format(filename))
    # plt.title('{}'.format(filename.replace('_good', '').replace('_bad', '')))
    plt.legend()  # Display legend

    # Set the abscissa scale and grid lines
    plt.xticks(np.arange(0, len(data), step=200))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        # Mark the red area within the specified range
        plt.axvspan(start[0], end[0], color='red', alpha=0.3)

    os.makedirs(os.path.join(source_dir, 'png1'), exist_ok=True)
    save_path = os.path.join(source_dir, 'png1', filename)
    save_path = save_path.replace('.npz', '.png')
    plt.savefig(save_path)
    plt.clf()  # Clear the current graph so that the next file can draw a new graph
    gc.collect()
print("9 生成png图像")





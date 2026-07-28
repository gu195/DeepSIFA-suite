import os
import shutil
import pandas as pd
import numpy as np
import re
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gc
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset
import json
import glob
import matplotlib

DATA = 'mlkl'
NUM = '1'
NUM_dimension = 1024
SIGMA = 1
randomSeed = 42




# 1 Generate data storage directory
# Define directory path
directory_path = './data/{}/train/v{}/原始数据'.format(DATA,NUM)
# Create directory
os.makedirs(directory_path, exist_ok=True)
print(f"1目录 '{directory_path}' 已成功创建。")




# # 2 Generate .txt suffix, add _bad _good
import os
# defines the directory to be processed
bad_directory = './data/{}/train/v{}/bad'.format(DATA, NUM)
good_directory = './data/{}/train/v{}/good'.format(DATA, NUM)
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
good_dir = f'./data/{DATA}/train/v{NUM}/good'
bad_dir = f'./data/{DATA}/train/v{NUM}/bad'
target_dir = f'./data/{DATA}/train/v{NUM}/原始数据'
# Function to move files
def move_files(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        if os.path.isfile(src_file):
            shutil.move(src_file, dest_file)

# Move files in the good directory
move_files(good_dir, target_dir)
# Move files in bad directory
move_files(bad_dir, target_dir)
# Delete good and bad directories
os.rmdir(good_dir)
os.rmdir(bad_dir)
print("3文件已成功移动并删除目录。")




# 4 alphaK10 obtains brightness information
if DATA == 'alphak10':
    # Define directory path
    directory = './data/{}/train/v{}/原始数据'.format(DATA,NUM)
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
directory = './data/{}/train/v{}/原始数据'.format(DATA,NUM)
def read_and_save_columns(file_path, output_path):
    with open(file_path, 'r') as file:
        # Skip the first line
        next(file)
        combined_data = []            #creates an empty list named combined_data

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
source_dir = './data/{}/train/v{}/原始数据'.format(DATA, NUM)
target_dir = './data/{}/train/v{}/归一化后npz'.format(DATA, NUM)

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
source_dir = './data/{}/train/v{}/归一化后npz/'.format(DATA, NUM)
target_dir = './data/{}/train/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA, NUM,NUM_dimension,SIGMA)

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
directory = './data/{}/train/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA, NUM,NUM_dimension,SIGMA)
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
csv_file_path = './data/{}/train/v{}/{}.csv'.format(DATA,NUM,len(os.listdir(directory)))
df.to_csv(csv_file_path, index=False)
print("8 根据_bad.npz或_good.npz生成标签与csv文件.",f"CSV file saved at {csv_file_path}")





# 9 Extract trainset_data_1.json from StratifiedKFold
class XrayDataset_5_fold(Dataset):#5-fold cross validation
    def __init__(self,
                 args):
        csv_files = glob.glob(os.path.join(args.csv_dir, '*.csv'))
        df = pd.read_csv(csv_files[0])

        self.npz_list = np.array(df['file_name'])
        self.label_list4 = np.array(df['label'])
        self.data_path = args.data_dir

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.npz_list[index]
        npz_path = os.path.join(self.data_path, name)

        npz_data = np.load(npz_path)
        data = npz_data['combined_data']

        label_cls = torch.tensor(self.label_list4[index])
        
        return data, label_cls, name

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default='512', type=int)
    parser.add_argument('--bt_size', type=int, default=16)  
    # github path
    parser.add_argument('--data_dir', type=str, default='./data/{}/train/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA,NUM,NUM_dimension,SIGMA))
    parser.add_argument('--csv_dir', type=str, default='./data/{}/train/v{}/'.format(DATA,NUM))
    parser.add_argument('--result_dir', type=str, default='./data/{}/train/v{}/5折交叉验证_{}'.format(DATA,NUM,randomSeed))
    parse_config = parser.parse_args()
    # print(parse_config)
    return parse_config


gc.collect()
torch.cuda.empty_cache()
parse_config = get_cfg()

# -------------------------- build dataloaders --------------------------#
dataset = XrayDataset_5_fold(parse_config)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomSeed)
train_loaders = []
eval_loaders = []

# Create a dictionary to save the file names and labels of all trainsets
trainset_data = {
    "train": {},
    "val": {}
}
train_data = {}
val_data = {}
idx11 = 0
os.makedirs(parse_config.result_dir,exist_ok=True)
for train_idx, val_idx in skf.split(dataset, dataset.label_list4):
    idx11 += 1
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    train_subset_filenames = [dataset.npz_list[idx] for idx in train_idx]
    train_subset_labels = [int(dataset.label_list4[idx]) for idx in train_idx]

    val_subset_filenames = [dataset.npz_list[idx] for idx in val_idx]
    val_subset_labels = [int(dataset.label_list4[idx]) for idx in val_idx] 


    # Use the zip function to pair the file name and label one by one and add it to trainset_data
    train_paired_data = dict(zip(train_subset_filenames, train_subset_labels))
    trainset_data['train'] = train_paired_data
    val_paired_data = dict(zip(val_subset_filenames, val_subset_labels))
    trainset_data['val'] = val_paired_data

    trainset = torch.utils.data.DataLoader(train_subset, batch_size=parse_config.bt_size, shuffle=True, drop_last=True)
    valset = torch.utils.data.DataLoader(val_subset, batch_size=1)
    train_loaders.append(trainset)
    eval_loaders.append(valset)

    with open( parse_config.result_dir + '/trainset_data' + '_'+str(idx11) + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(trainset_data, json_file, ensure_ascii=False, indent=4)
print("9从StratifiedKFold提取trainset_data_1.json")




# 10 Generate csv_v2 3 labels for each fold train and val from split_dataset.py
def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_dir', type=str, default='./data/{}/train/v{}/'.format(DATA,NUM))
    parser.add_argument('--source_dir', type=str, default='./data/{}/train/v{}/5折交叉验证_{}'.format(DATA,NUM,randomSeed))
    parser.add_argument('--result_dir', type=str, default='./data/{}/train/v{}/5折交叉验证_{}'.format(DATA,NUM,randomSeed))
    parse_config = parser.parse_args()
    # print(parse_config)
    return parse_config


gc.collect()
parse_config = get_cfg()

for i in range(1, 6):
    with open(parse_config.source_dir + f'/trainset_data_{i}.json', 'r') as json_file:
        data = json.load(json_file)
    train_data = data['train']
    val_data = data['val']
    csv_files = glob.glob(os.path.join(parse_config.csv_dir, '*.csv'))
    df = pd.read_csv(csv_files[0])
    new_df = df[['file_name', 'label']]

    # Divide the data into two parts: train and val according to the file name in the JSON file
    train_df = new_df[new_df['file_name'].isin(train_data.keys())]
    val_df = new_df[new_df['file_name'].isin(val_data.keys())]

    train_df.to_csv(parse_config.result_dir + f'/train_fold{i}_多标签.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(parse_config.result_dir + f'/val_fold{i}_多标签.csv', index=False, encoding='utf-8-sig')
    # print(f'Processing {i}th data set completed.')
print("10 从split_dataset.py生成每一折train和val的csv_v2 3个标签")




# # 11 Generate png image
matplotlib.use('Agg')
source_dir = './data/{}/train/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA,NUM,NUM_dimension,SIGMA)
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

print("11 生成png图像")


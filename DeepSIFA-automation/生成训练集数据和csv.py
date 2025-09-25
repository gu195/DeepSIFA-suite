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




# 1 生成data存放目录
# 定义目录路径
directory_path = './data/{}/train/v{}/原始数据'.format(DATA,NUM)
# 创建目录
os.makedirs(directory_path, exist_ok=True)
print(f"1目录 '{directory_path}' 已成功创建。")




# # 2 生成.txt后缀,加上_bad _good
import os
# 定义要处理的目录
bad_directory = './data/{}/train/v{}/bad'.format(DATA, NUM)
good_directory = './data/{}/train/v{}/good'.format(DATA, NUM)
def rename_files(directory, suffix):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # 确保是文件而不是目录
        if os.path.isfile(file_path):
            # 分离文件名和后缀
            name, ext = os.path.splitext(filename)
            if ext:  # 如果有后缀
                new_file_name = f"{name}{suffix}{ext}"
            else:  # 如果没有后缀
                new_file_name = f"{name}{suffix}.txt"
            new_file_path = os.path.join(directory, new_file_name)
            os.rename(file_path, new_file_path)
            # print(f"已重命名: {file_path} -> {new_file_path}")

# 处理 bad 目录
rename_files(bad_directory, '_bad')
# 处理 good 目录
rename_files(good_directory, '_good')
print("2 生成.txt后缀,加上_bad _good")



# 3移动文件
# 定义目录路径
good_dir = f'./data/{DATA}/train/v{NUM}/good'
bad_dir = f'./data/{DATA}/train/v{NUM}/bad'
target_dir = f'./data/{DATA}/train/v{NUM}/原始数据'
# 移动文件的函数
def move_files(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        if os.path.isfile(src_file):
            shutil.move(src_file, dest_file)

# 移动 good 目录下的文件
move_files(good_dir, target_dir)
# 移动 bad 目录下的文件
move_files(bad_dir, target_dir)
# 删除 good 和 bad 目录
os.rmdir(good_dir)
os.rmdir(bad_dir)
print("3文件已成功移动并删除目录。")




# 4 alphaK10获取亮度信息
if DATA == 'alphak10':
    # 定义目录路径
    directory = './data/{}/train/v{}/原始数据'.format(DATA,NUM)
    # 读取目录下的所有文件
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    # 遍历每个文件
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)

        # 计算第二列减去第三列的值，并将结果写入第六列
        df[5] = df[1] - df[2]
        # 添加列名为brightness
        df.columns = ['Time[s]', 'CH1', 'BGND1', 'CH2', 'BGND2', 'brightness']
        # 将结果写回原文件
        df.to_csv(file_path, sep="\t", index=False, header=True)
        # print(f"Processed file: {file}")
    print("4 alphaK10获取亮度信息.")




# 5 把txt文件都转化为npz文件存储
# 这段代码读取一个文件，提取每行的第一列和最后一列数据，并保存为 .npz 格式。
# 定义目录路径
directory = './data/{}/train/v{}/原始数据'.format(DATA,NUM)
def read_and_save_columns(file_path, output_path):
    with open(file_path, 'r') as file:
        # 跳过首行
        next(file)
        combined_data = []

        for line in file:
            splitted_line = line.strip()
            splitted_line = re.split(',|\t', splitted_line)
            combined_data.append([splitted_line[0], splitted_line[-1]])

        combined_data = np.array(combined_data)
        # 确保数据形状是 (数目, 2)
        assert combined_data.shape[1] == 2, "Data shape is not (num, 2)"
        # 保存为 .npz 文件
        np.savez(output_path, combined_data=combined_data)

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, os.path.splitext(filename)[0] + '.npz')
        read_and_save_columns(file_path, output_path)
print("5 把txt文件都转化为npz文件存储.")





# 6.归一化
# 设置原始数据和归一化后数据的路径
source_dir = './data/{}/train/v{}/原始数据'.format(DATA, NUM)
target_dir = './data/{}/train/v{}/归一化后npz'.format(DATA, NUM)

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历原始数据目录中的所有npz文件
for filename in os.listdir(source_dir):
    if filename.endswith('.npz'):
        # 读取每个npz文件
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
        # 保存归一化后的数据
        save_path = os.path.join(target_dir, filename)
        np.savez(save_path, combined_data=data)
print("6.归一化.")




## 7.resize到1024，并且高斯平滑  见高斯平滑.py
source_dir = './data/{}/train/v{}/归一化后npz/'.format(DATA, NUM)
target_dir = './data/{}/train/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA, NUM,NUM_dimension,SIGMA)

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历原始数据目录中的所有npz文件
for filename in os.listdir(source_dir):
    if filename.endswith('.npz'):
        # 读取每个npz文件
        file_path = os.path.join(source_dir, filename)
        npz_data = np.load(file_path)
        data = npz_data['combined_data']
        # print(data.shape)
        original_data = data[1, :]
        # 假设 original_data 是原始时间序列数据 且 original_data 是一个一维数组
        # 创建时间点
        original_indices = np.linspace(0, 1, len(original_data))
        new_indices = np.linspace(0, 1, NUM_dimension)
        # 创建插值函数
        interpolation_function = interp1d(original_indices, original_data, kind='linear')
        # 应用插值
        interpolated_data = interpolation_function(new_indices)
        # 设置高斯平滑的标准差（sigma）
        sigma = SIGMA
        # 对时间序列数据进行高斯平滑
        interpolated_data = gaussian_filter1d(interpolated_data, sigma)
        data_ = interpolated_data
        # 保存归一化后的数据
        save_path = os.path.join(target_dir, filename)
        np.savez(save_path, data=data_)
print("7.resize到1024，并且高斯平滑.")




# 8 根据_bad.npz或_good.npz生成标签与csv文件
directory = './data/{}/train/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA, NUM,NUM_dimension,SIGMA)
data = []
# 遍历目录，找到所有的txt文件
for file in os.listdir(directory):
    if "bad" in file:
        data.append({'file_name': file, 'label': 0})
    elif "good" in file:
        data.append({'file_name': file, 'label': 1})
    else:
        data.append({'file_name': file, 'label': -1})

# 从列表创建DataFrame
df = pd.DataFrame(data, columns=['file_name', 'label'])
csv_file_path = './data/{}/train/v{}/{}.csv'.format(DATA,NUM,len(os.listdir(directory)))
df.to_csv(csv_file_path, index=False)
print("8 根据_bad.npz或_good.npz生成标签与csv文件.",f"CSV file saved at {csv_file_path}")





# 9从StratifiedKFold提取trainset_data_1.json
class XrayDataset_5_fold(Dataset):#5折交叉验证用
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
    # github上路径
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

# 创建一个字典来保存所有trainset的文件名和标签
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


    # 使用zip函数将文件名和标签一一配对，并添加到trainset_data中
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




# 10 从split_dataset.py生成每一折train和val的csv_v2 3个标签
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

    # 根据JSON文件中的文件名将数据分为train和val两部分
    train_df = new_df[new_df['file_name'].isin(train_data.keys())]
    val_df = new_df[new_df['file_name'].isin(val_data.keys())]

    train_df.to_csv(parse_config.result_dir + f'/train_fold{i}_多标签.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(parse_config.result_dir + f'/val_fold{i}_多标签.csv', index=False, encoding='utf-8-sig')
    # print(f'处理第 {i} 个数据集完成。')
print("10 从split_dataset.py生成每一折train和val的csv_v2 3个标签")




# # 11 生成png图像
matplotlib.use('Agg')
source_dir = './data/{}/train/v{}/归一化插值后npz_{}_高斯平滑{}/'.format(DATA,NUM,NUM_dimension,SIGMA)
# 获取原始数据目录中的所有npz文件
npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]

# 使用 tqdm 添加进度条
for filename in tqdm(npz_files, desc='Processing', unit='file'):
    # 读取每个npz文件
    file_path = os.path.join(source_dir, filename)
    npz_data = np.load(file_path)
    data = npz_data['data']
    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']
    # 创建x轴坐标
    x = range(len(data))
    plt.figure(figsize=(24, 6))  # 设置图形的大小为宽10英寸，高5英寸
    
    # 绘制曲线和散点图
    plt.plot(x, data, label='Data Line', linewidth=0.5)  # 将线条宽度设置为1
    plt.scatter(x, data, label='Data Points', s=0.5)  # 绘制散点
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('{}'.format(filename))
    # plt.title('{}'.format(filename.replace('_good', '').replace('_bad', '')))
    plt.legend()  # 显示图例

    # 设置横坐标刻度和栅格线
    plt.xticks(np.arange(0, len(data), step=200))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        # 在指定范围内标出红色区域
        plt.axvspan(start[0], end[0], color='red', alpha=0.3)

    os.makedirs(os.path.join(source_dir, 'png1'), exist_ok=True)
    save_path = os.path.join(source_dir, 'png1', filename)
    save_path = save_path.replace('.npz', '.png')
    plt.savefig(save_path)
    plt.clf()  # 清除当前图形，以便下一个文件可以绘制新图
    gc.collect()

print("11 生成png图像")


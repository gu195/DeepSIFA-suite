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
from multiprocessing import Pool, cpu_count

DATA = 'MLKL'
NUM = '1'
NUM_dimension = 1024
SIGMA = 1




# # 1. 生成data存放目录
# def create_directory():
#     directory_path = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')
#     os.makedirs(directory_path, exist_ok=True)
#     print(f"1. 目录 '{directory_path}' 已成功创建。")
#     return directory_path


def create_directory():
    directory_path = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')
    # # 检查目录是否存在，如果存在则删除
    # if os.path.exists(directory_path):
    #     shutil.rmtree(directory_path)
    #     print(f"已存在的目录 '{directory_path}' 已被删除。")
    
    # 创建新目录
    os.makedirs(directory_path, exist_ok=True)
    print(f"1. 目录 '{directory_path}' 已成功创建。")
    return directory_path



# 2. 生成.txt后缀并加上_bad _good
def rename_and_append_suffix():
    for label in ['bad', 'good']:
        directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', label)
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    name, ext = os.path.splitext(filename)
                    new_file_name = f"{name}_{label}{ext}" if ext else f"{name}_{label}.txt"
                    new_file_path = os.path.join(directory, new_file_name)
                    os.rename(file_path, new_file_path)

    print("2. 生成.txt后缀, 加上_bad _good")



# 3. 移动文件并清理目录
def move_and_clean():
    good_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'good')
    bad_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'bad')
    target_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')

    # 定义用于移动文件的操作
    for src_dir in [good_dir, bad_dir]:
        if os.path.exists(src_dir):
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dest_file = os.path.join(target_dir, filename)
                if os.path.isfile(src_file):
                    shutil.move(src_file, dest_file)
            os.rmdir(src_dir)  # 删除空目录

    print("3. 文件已成功移动并删除目录。")




# 4. alphaK10 获取亮度信息
def get_brightness_info():
    if DATA == 'alphak10':
        directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')
        files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        for file in files:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)
            df[5] = df[1] - df[2]
            df.columns = ['Time[s]', 'CH1', 'BGND1', 'CH2', 'BGND2', 'brightness']
            df.to_csv(file_path, sep="\t", index=False, header=True)
        print("4. alphaK10 获取亮度信息。")




# 5. 把txt文件都转化为npz文件存储
def convert_txt_to_npz():
    directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')

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
                    combined_data.append([splitted_line[0], splitted_line[1],splitted_line[2], splitted_line[-1]])#第一行和最后一行

                combined_data = np.array(combined_data)
                assert combined_data.shape[1] == 4, "Data shape is not (num, 2)"

                # 保存为 .npz 文件
                np.savez(output_path, combined_data=combined_data)

    print("5. 把txt文件都转化为npz文件存储。")



# 不执行归一化
def no_normalize_data():
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')
    target_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    
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
            last_column = data[:, -1]  # Get the last column
            normalized_last_column = last_column
            data[:, -1] = normalized_last_column  # Replace the last column with its normalized version

            save_path = os.path.join(target_dir, filename)
            np.savez(save_path, combined_data=data)
    print("6. no归一化完成。")


# 6. 归一化
def normalize_data():
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')
    target_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    
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
    print("6. 归一化完成。")





def resize_and_smooth():
    # 第 7 段代码：Resize 到 1024，并且高斯平滑
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    target_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', f'归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}')

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

    print("7. Resize to 1024 and Gaussian smoothing completed.")






def generate_csv():
    # 第 8 段代码：根据 _bad.npz 或 _good.npz 生成标签与 csv 文件
    directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', f'归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}')
    data = []

    for file in os.listdir(directory):
        if "bad" in file:
            data.append({'file_name': file, 'label': 0})
        elif "good" in file:
            data.append({'file_name': file, 'label': 1})
        else:
            data.append({'file_name': file, 'label': -1})

    df = pd.DataFrame(data, columns=['file_name', 'label'])
    csv_file_path = './data/{}/test/v{}/{}.csv'.format(DATA, NUM, len(os.listdir(directory)))
    df.to_csv(csv_file_path, index=False)
    print(f"8. CSV file generated and saved at {csv_file_path}.")






def process_file_1(npz_file):# 可视化 归一化后npz
    # 定义npz文件的路径
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    npz_path = os.path.join(source_dir, npz_file)

    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    # file_path = os.path.join(source_dir, filename)
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

    save_path = os.path.join(source_dir, 'png', npz_file.replace('.npz', '.png'))
    plt.savefig(save_path)
    plt.clf()
    gc.collect()  # 手动清理内存




def process_file_2(npz_file):# 可视化归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}
    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', f'归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}')
    # file_path = os.path.join(source_dir, filename)

    # 定义npz文件的路径
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', f'归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}')
    os.makedirs(os.path.join(source_dir, 'png2'), exist_ok=True)
    npz_path = os.path.join(source_dir, npz_file)
    npz_data = np.load(npz_path)
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
    plt.savefig(save_path)
    plt.clf()
    gc.collect()  # 手动清理内存


# 修改一下画布的长度，保存的地址，多线程运行
def process_file_3(npz_file):#可视化原始数据
    # 定义npz文件的路径
    directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', f'归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)


    npz_path = os.path.join(directory, npz_file)

    # 加载npz文件
    npz_data = np.load(npz_path)
    
    # 提取存储的数据，假设key是 'combined_data'
    data = npz_data['combined_data']

    # # x轴为第一列，y轴为第二列
    # x = data[:, 0].astype(float)
    # Intensity = data[:, 3].astype(float)

    # # 绘制曲线
    # plt.figure(figsize=(24, 18))
    # plt.plot(x, Intensity, label=f'{npz_file}', color='blue')
    # # 调整X和Y轴刻度的字体大小
    # plt.tick_params(axis='both', which='major', labelsize=14)  # 设置x、y轴刻度字体大小

    # plt.title(f'{npz_file.replace(".npz", "")}', fontsize=20)  # 调大标题字体
    # plt.xlabel('Frame', fontsize=16)  # 调大x轴标签字体
    # plt.ylabel('Intensity', fontsize=16)  # 调大y轴标签字体
    # plt.grid(True)
    # plt.legend(fontsize=16)  # 调大图例字体
    # save_path = os.path.join(source_dir, 'png2', npz_file.replace('.npz', '.png'))
    # plt.savefig(save_path, dpi=300)
    # x轴为第一列，y轴为第二列，强度为第四列

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
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭图像，防止内存过多占用





if __name__ == "__main__":
    matplotlib.use('Agg')
    create_directory()
    rename_and_append_suffix()
    move_and_clean()
    get_brightness_info()
    convert_txt_to_npz()
    normalize_data()
    # no_normalize_data()
    resize_and_smooth()
    generate_csv()


    num_workers = cpu_count()  # 获取 CPU 核心数
    # # 生成png图像
    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    # npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    # # 遍历每个文件，逐一调用 process_file_1 函数
    # for filename in tqdm(npz_files, desc='Processing PNGs', unit='file'):
    #     process_file_1(filename)


    # # 生成png图像
    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', f'归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}')
    # npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    # # 遍历每个文件，逐一调用 process_file_2 函数
    # for filename in tqdm(npz_files, desc='Processing PNGs', unit='file'):
    #     process_file_2(filename)

    # 10 生成png图像
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_1, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    # 10 生成png图像
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', f'归一化插值后npz_{NUM_dimension}_高斯平滑{SIGMA}')
    npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_2, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    # # 10 生成png图像
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '原始数据')
    npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_3, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    print("10. PNG generation completed.")
    print('生成数据 完成！！！')




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




# # 1. Generate data storage directory
# def create_directory():
#     directory_path = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'original data')
#     os.makedirs(directory_path, exist_ok=True)
#     print(f"1. Directory '{directory_path}' has been created successfully.")
#     return directory_path


def create_directory():
    directory_path = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')
    # # Check whether the directory exists, delete it if it exists
    # if os.path.exists(directory_path):
    #     shutil.rmtree(directory_path)
    #     print(f"The existing directory '{directory_path}' has been deleted.")
    
    # Create new directory
    os.makedirs(directory_path, exist_ok=True)
    print(f"1. 目录 '{directory_path}' 已成功创建。")
    return directory_path



# 2. Generate .txt suffix and add _bad _good
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



# 3. Move files and clean directories
def move_and_clean():
    good_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'good')
    bad_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'bad')
    target_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')

    # Defines the operation for moving files
    for src_dir in [good_dir, bad_dir]:
        if os.path.exists(src_dir):
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dest_file = os.path.join(target_dir, filename)
                if os.path.isfile(src_file):
                    shutil.move(src_file, dest_file)
            os.rmdir(src_dir)  # Delete empty directories

    print("3. 文件已成功移动并删除目录。")




# 4. alphaK10 obtains brightness information
def get_brightness_info():
    if DATA == 'alphak10':
        directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')
        files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        for file in files:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1)
            df[5] = df[1] - df[2]
            df.columns = ['Time[s]', 'CH1', 'BGND1', 'CH2', 'BGND2', 'brightness']
            df.to_csv(file_path, sep="\t", index=False, header=True)
        print("4. alphaK10 获取亮度信息。")




# 5. Convert txt files to npz files for storage
def convert_txt_to_npz():
    directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')

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
                    combined_data.append([splitted_line[0], splitted_line[1],splitted_line[2], splitted_line[-1]])#first and last lines

                combined_data = np.array(combined_data)
                assert combined_data.shape[1] == 4, "Data shape is not (num, 2)"

                # Save as .npz file
                np.savez(output_path, combined_data=combined_data)

    print("5. 把txt文件都转化为npz文件存储。")



# No normalization is performed
def no_normalize_data():
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')
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


# 6. Normalization
def normalize_data():
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')
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
    # 7th code segment: Resize to 1024 and Gaussian smoothing
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    target_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'processing_data')

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

    print("7. Resize to 1024 and Gaussian smoothing completed.")






def generate_csv():
    # Section 8: Generate labels and csv files based on _bad.npz or _good.npz
    directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'processing_data')
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






def process_file_1(npz_file):# Visualization npz after normalization
    # defines the path of npz file
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    npz_path = os.path.join(source_dir, npz_file)

    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'npz after normalization')
    # file_path = os.path.join(source_dir, filename)
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

    save_path = os.path.join(source_dir, 'png', npz_file.replace('.npz', '.png'))
    plt.savefig(save_path)
    plt.clf()
    gc.collect()  # Manually clear memory




def process_file_2(npz_file):# Visualized normalized interpolation npz_{NUM_dimension}_Gaussian smoothing {SIGMA}
    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'processing_data')
    # file_path = os.path.join(source_dir, filename)

    # defines the path of npz file
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'processing_data')
    os.makedirs(os.path.join(source_dir, 'png2'), exist_ok=True)
    npz_path = os.path.join(source_dir, npz_file)
    npz_data = np.load(npz_path)
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
    plt.savefig(save_path)
    plt.clf()
    gc.collect()  # Manually clear memory


# Modify the length of the canvas, saved address, multi-threaded operation
def process_file_3(npz_file):#Visualize row_data
    # defines the path of npz file
    directory = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'processing_data')
    os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)


    npz_path = os.path.join(directory, npz_file)

    # Load npz file
    npz_data = np.load(npz_path)
    
    # extracts the stored data, assuming the key is 'combined_data'
    data = npz_data['combined_data']

    # # The x-axis is the first column, the y-axis is the second column
    # x = data[:, 0].astype(float)
    # Intensity = data[:, 3].astype(float)

    # # Draw curve
    # plt.figure(figsize=(24, 18))
    # plt.plot(x, Intensity, label=f'{npz_file}', color='blue')
    # # Adjust the font size of the X and Y axis scales
    # plt.tick_params(axis='both', which='major', labelsize=14) # Set the x and y axis scale font size

    # plt.title(f'{npz_file.replace(".npz", "")}', fontsize=20) # Increase the title font size
    # plt.xlabel('Frame', fontsize=16) # Increase the x-axis label font
    # plt.ylabel('Intensity', fontsize=16) # Increase the font size of the y-axis label
    # plt.grid(True)
    # plt.legend(fontsize=16) # Increase the legend font
    # save_path = os.path.join(source_dir, 'png2', npz_file.replace('.npz', '.png'))
    # plt.savefig(save_path, dpi=300)
    # The x-axis is the first column, the y-axis is the second column, and the intensity is the fourth column

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
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close the image to prevent excessive memory usage





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


    num_workers = cpu_count()  # Get the number of CPU cores
    # # Generate png image
    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'npz after normalization')
    # npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    # # Traverse each file and call the process_file_1 function one by one
    # for filename in tqdm(npz_files, desc='Processing PNGs', unit='file'):
    #     process_file_1(filename)


    # # Generate png image
    # source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'processing_data')
    # npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    # # Traverse each file and call the process_file_2 function one by one
    # for filename in tqdm(npz_files, desc='Processing PNGs', unit='file'):
    #     process_file_2(filename)

    # 10 Generate png image
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', '归一化后npz')
    npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_1, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    # 10 Generate png image
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'processing_data')
    npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_2, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    # # 10 Generate png image
    source_dir = os.path.join('.', 'data', str(DATA), 'test', f'v{NUM}', 'row_data')
    npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file_3, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    print("10. PNG generation completed.")
    print('生成数据 完成！！！')




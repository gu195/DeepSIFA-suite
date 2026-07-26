# Visualize x coordinate
import matplotlib
matplotlib.use('Agg')  # Switch to GUI-less backend
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import shutil
import time
# source_dir = r'D:\DeepSIFA_main\data\wt0117test\I\0216-v9-circle-g6-200-1.2-Gaussian filter\row_data'
source_dir = r'D:\DeepSIFA_main\data\wt0117测试\I\0314-v1-circle-g6-s3-f3-200-1.2-asy\row_data'
png_dir = os.path.join(source_dir, 'png_论文')

if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
    time.sleep(2)  # Wait 1 second to ensure complete deletion
os.makedirs(png_dir, exist_ok=True)

# Get all npz files in the original data directory
npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.txt')]


def process_file(filename):
    file_path = os.path.join(source_dir, filename)
    df = pd.read_csv(file_path, delimiter=',', header=1)  

    # Get the second and third columns of data
    intensity = df.iloc[:, 8].astype(float)
    column_2 = df.iloc[:, 3].astype(float)  # cast to float
    column_3 = df.iloc[:, 4].astype(float)  # cast to float
    data_I = intensity
    data_x = column_2 - 1
    data_y = column_3 - 1

    # Create x-axis coordinate
    x = range(len(data_x))

    # Create a subgraph with 3 rows and 1 column
    fig, axes = plt.subplots(3, 1, figsize=(26, 18))  # (row, column), here is 2 rows and 1 column
    # fig, axes = plt.subplots(3, 1, figsize=(15, 16)) # (rows, columns), here are 2 rows and 1 column
    # fig, axes = plt.subplots(3, 1, figsize=(24, 18)) # (rows, columns), here are 2 rows and 1 column

    # Draw the first subgraph (data_x)
    axes[0].plot(x, data_x, label='Data X', linewidth=2.5, color='#2CA02C')
    axes[0].set_xlabel('Time(s)', fontsize=38, labelpad=10)
    axes[0].set_ylabel('X-coordinate', fontsize=38, labelpad=10)
    axes[0].tick_params(axis='y', labelsize=25)
    axes[0].set_xticks([])  # Hide x-axis scale

    # Draw the second subgraph (data_y)
    axes[1].plot(x, data_y, label='Data Y', linewidth=2.5, color='#FF7F0E')
    axes[1].set_xlabel('Time(s)', fontsize=38, labelpad=10)
    axes[1].set_ylabel('Y-coordinate', fontsize=38, labelpad=10)
    axes[1].tick_params(axis='y', labelsize=25)
    axes[1].set_xticks([])  # Hide x-axis scale

    # Draw the third subgraph (I)
    axes[2].plot(x, data_I, label='Intensity', linewidth=2.5, color='#1F77B4')
    axes[2].set_xlabel('Time(s)', fontsize=38, labelpad=10)
    axes[2].set_ylabel('Intensity', fontsize=38, labelpad=10)
    axes[2].tick_params(axis='y', labelsize=25)
    axes[2].set_xticks([])  # Hide x-axis scale

    # Unified setting of border style
    for ax in axes:
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(2)

    # Adjust the spacing between sub-pictures
    plt.tight_layout()

    # Save image
    save_path = os.path.join(png_dir, filename.replace('.txt', '.png'))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.close()  # completely releases resources


# defines multi-process processing
if __name__ == '__main__':
    # num_workers = cpu_count() # Get the number of CPU cores
    # print(f'Using {num_workers} workers for parallel processing.')
    # # Use Pool for multi-process processing
    # with Pool(processes=num_workers) as pool:
    #     list(tqdm(pool.imap(process_file, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    # print("All PNGs have been generated.")

    # Process files one by one instead of using multiple processes
    for filename in tqdm(npz_files, total=len(npz_files), desc='Processing PNGs', unit='file'):
        process_file(filename)  # directly calls the processing function
    print("All PNGs have been generated.")






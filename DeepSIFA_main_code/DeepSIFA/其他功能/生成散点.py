import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
source_dir = r'D:\DeepSIFA_main\data\MLKL\test\v4\归一化插值后npz_1024_高斯平滑1'
png_dir = os.path.join(source_dir, 'png')
# If the directory exists, delete it first
if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
# Create a new png directory
os.makedirs(png_dir, exist_ok=True)


# Get all npz files in the original data directory
npz_files = [filename for filename in os.listdir(source_dir) if filename.endswith('.npz')]


def process_file(filename):
    file_path = os.path.join(source_dir, filename)
    npz_data = np.load(file_path)
    data = npz_data['data']

    # Get 'start' and 'end' data
    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        start = npz_data['start']
        end = npz_data['end']

    # Adjust the minimum value of the y-axis to -200
    min_value = data.min()
    shift_value = 0 - min_value  # difference
    data = data + shift_value  # Add the difference to each point

    # Create x-axis coordinate
    x = range(len(data))
    plt.figure(figsize=(18, 6))  # Set graphic size

    # draws a curve, the color is blue
    plt.plot(x, data, label='Data Line', linewidth=2.5, color='#1F77B4')  # blue

    # Set x-axis
    plt.xlabel('Time(s)', fontsize=24, labelpad=10)  # Set x-axis label and font size
    plt.xticks([], [])  # Hide x-axis scale

    # Set y-axis
    plt.ylabel('Intensity', fontsize=24, labelpad=10)  # Set y-axis label and font size
    plt.yticks(fontsize=20)  # Adjust the y-axis scale font size

    # Get the current axis object
    ax = plt.gca()

    # Set the thickness of the border
    line_width = 2 # Uniform line thickness

    # Enable and set the top and right borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # Customize the color and line width of the border
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(line_width)

    # Mark the red area within the specified range
    if 'start' in npz_data.keys() and 'end' in npz_data.keys():
        plt.axvspan(start[0], end[0], color='red', alpha=0.3)

    # # Save image
    # os.makedirs(os.path.join(source_dir, 'png'), exist_ok=True)
    save_path = os.path.join(source_dir, 'png', filename.replace('.npz', '.png'))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=500)
    plt.clf()  # Clear the current graph so that the next file can draw a new graph



# defines multi-process processing
if __name__ == '__main__':
    num_workers = cpu_count()  # Get the number of CPU cores
    print(f'Using {num_workers} workers for parallel processing.')

    # Using Pool for multi-process processing
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_file, npz_files), total=len(npz_files), desc='Processing PNGs', unit='file'))

    print("All PNGs have been generated.")







# # Compare after Gaussian filtering and before Gaussian filtering
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# from tqdm import tqdm

# source_dir1 = '/home/node01/linchen/data/alphaK10/V2/third batch/npz_1024 after normalized interpolation'
# source_dir2 = '/home/node01/linchen/data/alphaK10/V2/third batch/npz_1024_Gaussian smoothing 1' after normalized interpolation
# source_dir3 = '/home/node01/linchen/data/alphaK10/V2/third batch/npz_1024_Gaussian smoothing 2' after normalized interpolation

# # Get all npz files in the original data directory
# npz_files1 = [filename for filename in os.listdir(source_dir1) if filename.endswith('.npz')]
# npz_files2 = [filename for filename in os.listdir(source_dir2) if filename.endswith('.npz')]

# # Use tqdm to add a progress bar
# for filename in tqdm(npz_files1, desc='Processing', unit='file'):
#     # Read each npz file
#     file_path1 = os.path.join(source_dir1, filename)
#     npz_data1 = np.load(file_path1)
#     data1 = npz_data1['data']

#     file_path2 = os.path.join(source_dir2, filename)
#     npz_data2 = np.load(file_path2)
#     data2 = npz_data2['data']

#     file_path3 = os.path.join(source_dir3, filename)
#     npz_data3 = np.load(file_path3)
#     data3 = npz_data3['data']

#     if 'start' in npz_data1.keys() and 'end' in npz_data1.keys():
#         start = npz_data1['start']
#         end = npz_data1['end']
#     # Create x-axis coordinate
#     x = range(len(data1))
#     plt.figure(figsize=(24, 6)) # Set the size of the figure to 10 inches wide and 5 inches high
    
#     # Draw curves and scatter plots
#     plt.plot(x, data1, 'k', label='original data', linewidth=0.5) # Set the line width to 1
#     plt.plot(x, data2, '--', label='filtered, sigma=1', linewidth=1) # Set the line width to 1
#     plt.plot(x, data3, ':', label='filtered, sigma=2', linewidth=1) # Set the line width to 1
    
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.title('{}'.format(filename))
#     plt.legend() # Display legend

#     # Set the abscissa scale and grid lines
#     plt.xticks(np.arange(0, len(data1), step=25))
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)

#     if 'start' in npz_data1.keys() and 'end' in npz_data1.keys():
#         # Mark the red area within the specified range
#         plt.axvspan(start[0], end[0], color='red', alpha=0.3)

#     os.makedirs(os.path.join(source_dir1, 'png'), exist_ok=True)
#     save_path = os.path.join(source_dir1, 'png', filename)
#     save_path = save_path.replace('.npz', '.png')
#     plt.savefig(save_path, dpi=400)
#     plt.clf() # Clear the current graph so that the next file can draw a new graph

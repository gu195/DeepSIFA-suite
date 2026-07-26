#############################################################################################
#############################################################################################
#############################################################################################
#############Draw the probability diagram of the curve
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

FOLD = '5'
# definition file path
data_dir = '/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3_数据增强/归一化插值后npz_1024_高斯平滑1/png_没有信息'
target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{FOLD}/png7_8/'
target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{FOLD}/png8_10/'
input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{FOLD}/score.csv'
os.makedirs(target_dir1,    exist_ok=True)
os.makedirs(target_dir2,    exist_ok=True)

# Read CSV file
df = pd.read_csv(input_file)
# Filters out data with scores within the specified range
df = df[df['label'] == 0]
df_filtered1 = df[(df['score'] >= 0.7) & (df['score'] < 0.8)]
df_filtered2 = df[(df['score'] >= 0.8) & (df['score'] <= 1)]

# Get the list of files that meet the conditions
files_to_copy1 = df_filtered1['name']  # Assume that the file name is listed as 'file_name' in the DataFrame example
files_to_copy2 = df_filtered2['name']

# Copy files to the target directory
for file in files_to_copy1:
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        shutil.copy(source_file, target_dir1)

for file in files_to_copy2:
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        shutil.copy(source_file, target_dir2)








# # #############################################################################################
# # #############################################################################################
# # #############################################################################################
# # #############Draw the probability diagram of the curve
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

FOLD = '5'
# definition file path
data_dir = '/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3_数据增强/归一化插值后npz_1024_高斯平滑1/png_没有信息'
target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{FOLD}/png2_3/'
target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{FOLD}/png0_2/'
input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{FOLD}/score.csv'
os.makedirs(target_dir1,    exist_ok=True)
os.makedirs(target_dir2,    exist_ok=True)

# Read CSV file
df = pd.read_csv(input_file)
# Filters out data with scores within the specified range
df = df[df['label'] == 1]
df_filtered1 = df[(df['score'] > 0.2) & (df['score'] <= 0.3)]
df_filtered2 = df[(df['score'] >= 0) & (df['score'] <= 0.2)]

# Get the list of files that meet the conditions
files_to_copy1 = df_filtered1['name']  # Assume that the file name is listed as 'file_name' in the DataFrame example
files_to_copy2 = df_filtered2['name']

# Copy files to the target directory
for file in files_to_copy1:
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        shutil.copy(source_file, target_dir1)

for file in files_to_copy2:
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        shutil.copy(source_file, target_dir2)

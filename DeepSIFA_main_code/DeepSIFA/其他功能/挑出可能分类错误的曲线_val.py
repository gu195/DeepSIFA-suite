#############################################################################################
#############################################################################################
#############################################################################################
#############Draw the probability diagram of the curve
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import shutil

# FOLD = '3'
# # Define file path
# data_dir = '/home/node01/linchen/data/alphaK10/V2/summary/1+2+3+4_data enhancement_modified/after normalized interpolation npz_1024_Gaussian smoothing 1/png1'
# target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_modified/logs_val verification/fold{FOLD}/png0.2_0.4/'
# target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_modified/logs_val verification/fold{FOLD}/png0.4_0.7/'
# input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_modified/logs_val verification/fold{FOLD}/score.csv'
# os.makedirs(target_dir1,    exist_ok=True)
# os.makedirs(target_dir2,    exist_ok=True)

# # Read CSV file
# df = pd.read_csv(input_file)
# # Filter out data with scores within the specified range
# df = df[df['label'] == 0]
# df_filtered1 = df[(df['score'] > 0.2) & (df['score'] <= 0.4)]
# df_filtered2 = df[(df['score'] > 0.4) & (df['score'] <= 0.7)]

# # Get the list of files that meet the conditions
# files_to_copy1 = df_filtered1['name'] # Assume that the file name is in the 'file_name' column in the DataFrame example
# files_to_copy2 = df_filtered2['name']

# # Copy files to the target directory
# for file in files_to_copy1:
#     source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
#     if os.path.isfile(source_file):
#         shutil.copy(source_file, target_dir1)

# for file in files_to_copy2:
#     source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
#     if os.path.isfile(source_file):
#         shutil.copy(source_file, target_dir2)


import pandas as pd
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

FOLD = '3'
# definition file path
data_dir = '/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3+4_数据增强_修改后/归一化插值后npz_1024_高斯平滑1/png1'
target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/png0.2_0.4/'
target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/png0.4_0.7/'
input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/score.csv'
os.makedirs(target_dir1, exist_ok=True)
os.makedirs(target_dir2, exist_ok=True)

# Read CSV file
df = pd.read_csv(input_file)
# Filters out data with scores within the specified range
df = df[df['label'] == 0]
df_filtered1 = df[(df['score'] > 0.2) & (df['score'] <= 0.4)]
df_filtered2 = df[(df['score'] > 0.4) & (df['score'] <= 0.7)]

# Get the list of files that meet the conditions
files_to_copy1 = df_filtered1[['name', 'score']]
files_to_copy2 = df_filtered2[['name', 'score']]

def add_confidence_to_image(image_path, score, output_path):
    # Open picture
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Replace with the font path on your system
    font_size = 30  # Adjust font size
    font = ImageFont.truetype(font_path, font_size)

    # Add confidence text
    text = f'Confidence: {score:.4f}'
    text_position = (1050, 10)  # Text position, upper left corner
    text_color = (255, 0, 0)  # text color, red
    draw.text(text_position, text, fill=text_color, font=font)

    # Save image with text
    img.save(output_path)

# Copy the file and add confidence information to the image
for _, row in files_to_copy1.iterrows():
    file = row['name']
    score = row['score']
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    target_file = os.path.join(target_dir1, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        add_confidence_to_image(source_file, score, target_file)

for _, row in files_to_copy2.iterrows():
    file = row['name']
    score = row['score']
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    target_file = os.path.join(target_dir2, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        add_confidence_to_image(source_file, score, target_file)









# # # #############################################################################################
# # # #############################################################################################
# # # #############################################################################################
# # # #############Draw the probability diagram of the curve
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import shutil

# FOLD = '5'
# # Define file path
# data_dir = '/home/node01/linchen/data/alphaK10/V2/summary/1+2+3_data enhancement/normalized interpolation npz_1024_Gaussian smoothing 1/png_no information'
# target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold{FOLD}/png2_3/'
# target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold{FOLD}/png0_2/'
# input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold{FOLD}/score.csv'
# os.makedirs(target_dir1,    exist_ok=True)
# os.makedirs(target_dir2,    exist_ok=True)

# # Read CSV file
# df = pd.read_csv(input_file)
# # Filter out data with scores within the specified range
# df = df[df['label'] == 1]
# df_filtered1 = df[(df['score'] > 0.2) & (df['score'] <= 0.3)]
# df_filtered2 = df[(df['score'] >= 0) & (df['score'] <= 0.2)]

# # Get the list of files that meet the conditions
# files_to_copy1 = df_filtered1['name'] # Assume that the file name is in the 'file_name' column in the DataFrame example
# files_to_copy2 = df_filtered2['name']

# # Copy files to the target directory
# for file in files_to_copy1:
#     source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
#     if os.path.isfile(source_file):
#         shutil.copy(source_file, target_dir1)

# for file in files_to_copy2:
#     source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
#     if os.path.isfile(source_file):
#         shutil.copy(source_file, target_dir2)

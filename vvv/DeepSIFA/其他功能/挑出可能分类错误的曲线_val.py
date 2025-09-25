#############################################################################################
#############################################################################################
#############################################################################################
#############画好曲线的概率图
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import shutil

# FOLD = '3'
# # 定义文件路径
# data_dir = '/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3+4_数据增强_修改后/归一化插值后npz_1024_高斯平滑1/png1'
# target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/png0.2_0.4/'
# target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/png0.4_0.7/'
# input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/score.csv'
# os.makedirs(target_dir1,    exist_ok=True)
# os.makedirs(target_dir2,    exist_ok=True)

# # 读取CSV文件
# df = pd.read_csv(input_file)
# # 筛选出分数在指定范围内的数据
# df = df[df['label'] == 0]
# df_filtered1 = df[(df['score'] > 0.2) & (df['score'] <= 0.4)]
# df_filtered2 = df[(df['score'] > 0.4) & (df['score'] <= 0.7)]

# # 获取符合条件的文件列表
# files_to_copy1 = df_filtered1['name']  # 假设文件名在 DataFrame 中以 'file_name' 列示例
# files_to_copy2 = df_filtered2['name']

# # 复制文件到目标目录
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
# 定义文件路径
data_dir = '/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3+4_数据增强_修改后/归一化插值后npz_1024_高斯平滑1/png1'
target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/png0.2_0.4/'
target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/png0.4_0.7/'
input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.2_修改后/logs_val验证/fold{FOLD}/score.csv'
os.makedirs(target_dir1, exist_ok=True)
os.makedirs(target_dir2, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(input_file)
# 筛选出分数在指定范围内的数据
df = df[df['label'] == 0]
df_filtered1 = df[(df['score'] > 0.2) & (df['score'] <= 0.4)]
df_filtered2 = df[(df['score'] > 0.4) & (df['score'] <= 0.7)]

# 获取符合条件的文件列表
files_to_copy1 = df_filtered1[['name', 'score']]
files_to_copy2 = df_filtered2[['name', 'score']]

def add_confidence_to_image(image_path, score, output_path):
    # 打开图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # 替换为你系统中的字体路径
    font_size = 30  # 调整字体大小
    font = ImageFont.truetype(font_path, font_size)

    # 添加置信度文本
    text = f'Confidence: {score:.4f}'
    text_position = (1050, 10)  # 文本位置，左上角
    text_color = (255, 0, 0)  # 文本颜色，红色
    draw.text(text_position, text, fill=text_color, font=font)

    # 保存带有文本的图片
    img.save(output_path)

# 复制文件并在图片上添加置信度信息
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
# # # #############画好曲线的概率图
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import shutil

# FOLD = '5'
# # 定义文件路径
# data_dir = '/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3_数据增强/归一化插值后npz_1024_高斯平滑1/png_没有信息'
# target_dir1 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold{FOLD}/png2_3/'
# target_dir2 = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold{FOLD}/png0_2/'
# input_file = f'/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold{FOLD}/score.csv'
# os.makedirs(target_dir1,    exist_ok=True)
# os.makedirs(target_dir2,    exist_ok=True)

# # 读取CSV文件
# df = pd.read_csv(input_file)
# # 筛选出分数在指定范围内的数据
# df = df[df['label'] == 1]
# df_filtered1 = df[(df['score'] > 0.2) & (df['score'] <= 0.3)]
# df_filtered2 = df[(df['score'] >= 0) & (df['score'] <= 0.2)]

# # 获取符合条件的文件列表
# files_to_copy1 = df_filtered1['name']  # 假设文件名在 DataFrame 中以 'file_name' 列示例
# files_to_copy2 = df_filtered2['name']

# # 复制文件到目标目录
# for file in files_to_copy1:
#     source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
#     if os.path.isfile(source_file):
#         shutil.copy(source_file, target_dir1)

# for file in files_to_copy2:
#     source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
#     if os.path.isfile(source_file):
#         shutil.copy(source_file, target_dir2)

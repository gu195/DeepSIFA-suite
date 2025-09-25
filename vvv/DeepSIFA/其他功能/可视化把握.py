# # 1可视化把握把预测错误的保存到fold{}/score.csv中
# import pandas as pd

# # 读取原始的 score.csv 文件
# score_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/score.csv'
# data = pd.read_csv(score_file_path)

# # 找出 label 列和 pre 列不相等的行
# wrong_data = data[data['label'] != data['pre']]

# # 保存错误行到 wrong.csv 文件中
# wrong_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong.csv'
# wrong_data.to_csv(wrong_file_path, index=False)




# # 2把fold{}/score.csv中把握小于70的可视化出来
# import pandas as pd

# # 读取原始的 wrong.csv 文件
# wrong_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong.csv'
# data = pd.read_csv(wrong_file_path)

# # 找出 score 列小于等于 0.70 的行
# wrong_070_data = data[data['score'] <= 0.70]

# # 保存 score 列小于等于 0.70 的行到 wrong_0.7.csv 文件中
# wrong_070_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong_0.7.csv'
# wrong_070_data.to_csv(wrong_070_file_path, index=False)



# # # 3把预测错误的.tiff图片复制一份到/fold3/wrong_img
# import os
# import shutil
# import pandas as pd

# # 读取 wrong.csv 文件
# wrong_csv_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong.csv'
# data = pd.read_csv(wrong_csv_path)

# # 原始数据目录和目标目录
# source_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/data/F/第五批/原始数据'
# target_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong_img'

# # 确保目标目录存在
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)

# # 遍历 wrong.csv 中的每一行
# for index, row in data.iterrows():
#     # 获取文件名
#     file_name = row['name'].replace('.npz', '.tiff')
#     # 构建原始文件路径和目标文件路径
#     source_file_path = os.path.join(source_dir, file_name)
#     target_file_path = os.path.join(target_dir, file_name)
#     # 复制文件并将 .npz 文件改为 .tiff 文件
#     shutil.copy(source_file_path, target_file_path)



# #4 把后缀名字去掉
import os
import shutil
target_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong_img'
filtered_target_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong_img筛选'

if not os.path.exists(filtered_target_dir):
    os.makedirs(filtered_target_dir)

# 遍历目标目录下的所有文件
for filename in os.listdir(target_dir):
    if filename.endswith('.tiff'):
        # 获取文件的路径
        file_path = os.path.join(target_dir, filename)
        # 去掉文件名中最后一个下划线后面的内容
        new_filename = filename.rsplit('_', 1)[0] + '.tiff'
        # 新文件的路径
        new_file_path = os.path.join(filtered_target_dir, new_filename)
        # 复制文件到新的目标目录
        shutil.copyfile(file_path, new_file_path)







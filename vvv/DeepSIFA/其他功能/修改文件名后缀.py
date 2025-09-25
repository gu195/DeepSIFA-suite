import os

# 定义目录路径
# directory = '/home/node01/linchen/data/K10/训练集/1'
# directory = '/home/node01/linchen/data/K10/训练集/2'
# directory = '/home/node01/linchen/data/K10/验证集/1'
directory = '/home/node01/linchen/data/K10/验证集/2'


# 遍历目录中的文件
for filename in os.listdir(directory):

    # 生成新的文件名
    new_filename = filename.replace('_good', '').replace('_bad', '')
    # 获取完整的文件路径
    old_filepath = os.path.join(directory, filename)
    new_filepath = os.path.join(directory, new_filename)
    # 重命名文件
    os.rename(old_filepath, new_filepath)
    print(f'Renamed: {filename} -> {new_filename}')

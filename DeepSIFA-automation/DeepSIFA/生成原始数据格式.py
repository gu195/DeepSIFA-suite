# 复原原始数据


# 1移动到bad good
import os
import shutil

# 定义源目录和目标目录
source_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\原始数据'
bad_target_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\bad'
good_target_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\good'

# 创建目标目录，如果不存在
os.makedirs(bad_target_dir, exist_ok=True)
os.makedirs(good_target_dir, exist_ok=True)

# 遍历源目录中的文件
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    if '_bad' in filename:
        shutil.move(file_path, os.path.join(bad_target_dir, filename))
    elif '_good' in filename:
        shutil.move(file_path, os.path.join(good_target_dir, filename))

print("文件移动完成！")





# 2删除.npz文件
import os

# 定义目标目录
bad_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\bad'

# 遍历目标目录中的文件
for filename in os.listdir(bad_dir):
    if filename.endswith('.npz'):
        file_path = os.path.join(bad_dir, filename)
        os.remove(file_path)
        print(f"已删除: {file_path}")

# 定义目标目录
good_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\good'

# 遍历目标目录中的文件
for filename in os.listdir(good_dir):
    if filename.endswith('.npz'):
        file_path = os.path.join(good_dir, filename)
        os.remove(file_path)
        print(f"已删除: {file_path}")

print("所有 .npz 文件已删除！")




# 3 去掉_bad _good
import os
# 定义目标目录
bad_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\bad'

# 遍历目标目录中的文件
for filename in os.listdir(bad_dir):
    if '_bad' in filename:
        # 构建旧的文件路径和新的文件路径
        old_file_path = os.path.join(bad_dir, filename)
        new_file_name = filename.replace('_bad', '')
        new_file_path = os.path.join(bad_dir, new_file_name)
        
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"已重命名: {old_file_path} -> {new_file_path}")
print("所有文件名中的 '_bad' 已去掉！")


# 定义目标目录
good_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\good'
# 遍历目标目录中的文件
for filename in os.listdir(good_dir):
    if '_good' in filename:
        # 构建旧的文件路径和新的文件路径
        old_file_path = os.path.join(good_dir, filename)
        new_file_name = filename.replace('_good', '')
        new_file_path = os.path.join(good_dir, new_file_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"已重命名: {old_file_path} -> {new_file_path}")

print("所有文件名中的 '_good' 已去掉！")



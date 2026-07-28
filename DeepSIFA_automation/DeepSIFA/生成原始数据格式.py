# Restore original data


# 1 moved to bad good
import os
import shutil

# Define source directory and target directory
source_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\原始数据'
bad_target_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\bad'
good_target_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\good'

# Create the target directory if it does not exist
os.makedirs(bad_target_dir, exist_ok=True)
os.makedirs(good_target_dir, exist_ok=True)

# Traverse files in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    if '_bad' in filename:
        shutil.move(file_path, os.path.join(bad_target_dir, filename))
    elif '_good' in filename:
        shutil.move(file_path, os.path.join(good_target_dir, filename))

print("文件移动完成！")





# 2Delete .npz files
import os

# Define the target directory
bad_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\bad'

# Traverse files in the target directory
for filename in os.listdir(bad_dir):
    if filename.endswith('.npz'):
        file_path = os.path.join(bad_dir, filename)
        os.remove(file_path)
        print(f"已删除: {file_path}")

# Define the target directory
good_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\good'

# Traverse files in the target directory
for filename in os.listdir(good_dir):
    if filename.endswith('.npz'):
        file_path = os.path.join(good_dir, filename)
        os.remove(file_path)
        print(f"已删除: {file_path}")

print("所有 .npz 文件已删除！")




# 3 Remove _bad _good
import os
# Define the target directory
bad_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\bad'

# Traverse files in the target directory
for filename in os.listdir(bad_dir):
    if '_bad' in filename:
        # Build old file path and new file path
        old_file_path = os.path.join(bad_dir, filename)
        new_file_name = filename.replace('_bad', '')
        new_file_path = os.path.join(bad_dir, new_file_name)
        
        # Rename file
        os.rename(old_file_path, new_file_path)
        print(f"已重命名: {old_file_path} -> {new_file_path}")
print("所有文件名中的 '_bad' 已去掉！")


# Define the target directory
good_dir = r'D:\1DeepSIFA\data\mlkl\test\v1\good'
# Traverse files in the target directory
for filename in os.listdir(good_dir):
    if '_good' in filename:
        # Build old file path and new file path
        old_file_path = os.path.join(good_dir, filename)
        new_file_name = filename.replace('_good', '')
        new_file_path = os.path.join(good_dir, new_file_name)
        # Rename file
        os.rename(old_file_path, new_file_path)
        print(f"已重命名: {old_file_path} -> {new_file_path}")

print("所有文件名中的 '_good' 已去掉！")



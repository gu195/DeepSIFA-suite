import os

# Define target path
folder_path = r'D:\DeepSIFA_main\data\I1_0315\bad_5'

# Traverse all files in the folder
for filename in os.listdir(folder_path):
    old_file = os.path.join(folder_path, filename)
    
    # Only processes files, skipping folders
    if os.path.isfile(old_file):
        # New file name plus "1_2_"
        new_file = os.path.join(folder_path, f"1_5_{filename}")
        os.rename(old_file, new_file)

print("文件重命名完成！")

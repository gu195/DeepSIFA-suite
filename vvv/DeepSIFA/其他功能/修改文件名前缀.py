import os

# 定义目标路径
folder_path = r'D:\DeepSIFA_main\data\I1_0315\bad_5'

# 遍历文件夹下的所有文件
for filename in os.listdir(folder_path):
    old_file = os.path.join(folder_path, filename)
    
    # 只处理文件，跳过文件夹
    if os.path.isfile(old_file):
        # 新文件名加上 "1_2_"
        new_file = os.path.join(folder_path, f"1_5_{filename}")
        os.rename(old_file, new_file)

print("文件重命名完成！")

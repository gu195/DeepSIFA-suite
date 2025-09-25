import os, argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt



def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high', type=str, default='0.7', help='High confidence threshold')
    parser.add_argument('--low', type=str, default='0.1', help='Low confidence threshold')
    parser.add_argument('--base_dir', type=str, default=os.path.join('data', 'wt_0119', 'I'), help='Base directory for source data')
    # 解析命令行参数
    parse_config = parser.parse_args()
    return parse_config



def move_and_rename_files(csv_file_path, txt_source_dir, destination_dir, suffix):
    df = pd.read_csv(csv_file_path)
    
    # 确保目标目录存在
    os.makedirs(destination_dir, exist_ok=True)

    # 遍历 file_name 列中的每个文件名
    for index, row in df.iterrows():
        file_name = row['name']
        score = row['score']
        
        # 将 .npz 后缀替换为 .txt
        txt_file_name = file_name.replace('.npz', suffix)
        txt_file_path = os.path.join(txt_source_dir, txt_file_name)
        destination_file_path = os.path.join(destination_dir, txt_file_name)
        
        # 检查 .txt 文件是否存在
        if os.path.exists(txt_file_path):
            shutil.copy(txt_file_path, destination_dir)
            prefix, ext = os.path.splitext(file_name)
            new_file_name = prefix + f"_{score}" + suffix
            new_file_path = os.path.join(destination_dir, new_file_name)
            
            # 重命名复制后的文件
            os.rename(destination_file_path, new_file_path)
        else:
            print(f"文件未找到: {txt_file_path}")





def process_and_sort_images(parse_config):
    # 定义源图片目录和目标目录
    base_dir_png = os.path.join(BASE_DIR, 'processing_data', 'png')
    csv_file_path =  os.path.join(DEEP_SIFA_DIR, 'score.csv')
    destination_dir = os.path.join(DEEP_SIFA_DIR, 'scoresort')

    df = pd.read_csv(csv_file_path)
    # 按 'score' 列降序排序
    sorted_df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # 确保目标目录存在（如果存在，先删除后重新创建）
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    os.makedirs(destination_dir, exist_ok=True)
    
    # 遍历排序后的文件，并按顺序拷贝和重命名
    for index, row in sorted_df.iterrows():
        file_name = row['name']
        score = row['score']
        
        # 拼接源文件路径和目标文件路径
        source_file_path = os.path.join(base_dir_png, file_name.replace('.npz', '.png'))
        prefix, ext = os.path.splitext(file_name)
        destination_file_name = f"{index + 1}_" + prefix + f"_{score}.png" # 根据排序索引重命名
        destination_file_path = os.path.join(destination_dir, destination_file_name)
        shutil.copy(source_file_path, destination_file_path)


    print(f"排序后的图片已保存到目录: {destination_dir}")





def count_images_and_log(parse_config):
    # 定义三个目录路径
    base_dir = os.path.join(DEEP_SIFA_DIR, 'low{}_high{}').format(parse_config.low, parse_config.high)
    dirs = {
        'bad': os.path.join(base_dir, 'bad_图片'),
        'good': os.path.join(base_dir, 'good_图片'),
        'ambiguous': os.path.join(base_dir, 'vague_图片')
    }
    
    log_file = os.path.join(base_dir, 'log.txt')
    
    with open(log_file, 'w') as f:
        for category, path in dirs.items():
            if os.path.exists(path):
                # 统计目录下的图片数量
                image_count = len([file for file in os.listdir(path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
                # 写入log文件
                f.write(f'{category}: {image_count}\n')
                # print(f'{category} has {image_count} images')
            else:
                f.write(f'{category} directory does not exist\n')
                print(f'{category} directory does not exist')
    
    print(f"Results saved to {log_file}")






parse_config = get_cfg()
print('分层 base_dir路径', parse_config.base_dir)
folder_path = parse_config.base_dir
# 遍历文件夹中的所有 .tif 文件
tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
for idx, filename in enumerate(tif_files):
    file_path = os.path.join(folder_path, filename)
    TIF_NAME = filename.replace('.tif', '')
    print('选取的TIRF图片为',TIF_NAME)
    BASE_DIR = os.path.join(parse_config.base_dir, TIF_NAME)
    base_dir_txt = os.path.join(BASE_DIR, 'row_data')
    base_dir_png = os.path.join(BASE_DIR, 'processing_data', 'png')
    DEEP_SIFA_DIR = os.path.join('DeepSIFA', 'logs_test_{}'.format(TIF_NAME))
    csv_file_path =  os.path.join(DEEP_SIFA_DIR, 'score.csv')

    df = pd.read_csv(csv_file_path)
    # 将每一行转换为字典，形成 allscore 列表
    allscore = df.to_dict(orient='records')
    # 初始化两个列表
    subscore1 = []
    subscore2 = []
    subscore3 = []
    # 遍历 allscore 列表
    for result in allscore:
        if result['score'] >= float(parse_config.high):
            subscore1.append(result)
        if result['score'] <= float(parse_config.low):
            subscore2.append(result)
        if result['score'] > float(parse_config.low) and result['score'] < float(parse_config.high):
            subscore3.append(result)

    # 将 subscore1, subscore2, subscore3 保存为不同的 CSV 文件
    directory = os.path.join(DEEP_SIFA_DIR, 'low{}_high{}'.format(parse_config.low, parse_config.high))
    if os.path.exists(directory):
        shutil.rmtree(directory)  # 删除整个目录
    os.makedirs(directory, exist_ok=True)  # 重新创建目录
    pd.DataFrame(subscore1).to_csv(os.path.join(directory, 'scoreHIGH.csv'), index=False)
    pd.DataFrame(subscore2).to_csv(os.path.join(directory, 'scoreLOW.csv'), index=False)
    pd.DataFrame(subscore3).to_csv(os.path.join(directory, 'scoreMIDDLE.csv'), index=False)


    # 高置信度
    move_and_rename_files(
        csv_file_path=os.path.join(directory, 'scoreHIGH.csv'),
        txt_source_dir=base_dir_txt,
        destination_dir=os.path.join(directory, 'good_txt'),
        suffix='.txt'
    )

    # 低置信度
    move_and_rename_files(
        csv_file_path=os.path.join(directory, 'scoreLOW.csv'),
        txt_source_dir=base_dir_txt,
        destination_dir=os.path.join(directory, 'bad_txt'),
        suffix='.txt'
    )

    # 模糊
    move_and_rename_files(
        csv_file_path=os.path.join(directory, 'scoreMIDDLE.csv'),
        txt_source_dir=base_dir_txt,
        destination_dir=os.path.join(directory, 'vague_txt'),
        suffix='.txt'
    )

    # 高置信度 (png)
    move_and_rename_files(
        csv_file_path=os.path.join(directory, 'scoreHIGH.csv'),
        txt_source_dir=base_dir_png,
        destination_dir=os.path.join(directory, 'good_图片'),
        suffix='.png'
    )

    # 低置信度 (png)
    move_and_rename_files(
        csv_file_path=os.path.join(directory, 'scoreLOW.csv'),
        txt_source_dir=base_dir_png,
        destination_dir=os.path.join(directory, 'bad_图片'),
        suffix='.png'
    )

    # 模糊 (png)
    move_and_rename_files(
        csv_file_path=os.path.join(directory, 'scoreMIDDLE.csv'),
        txt_source_dir=base_dir_png,
        destination_dir=os.path.join(directory, 'vague_图片'),
        suffix='.png'
    )


    count_images_and_log(parse_config)
    process_and_sort_images(parse_config)
    print("分层 完成")











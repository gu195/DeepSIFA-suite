import os, argparse
import shutil
import pandas as pd
import matplotlib.pyplot as plt
SAVE_LOG = 'logs_10_4'


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high', type=str, default='0.7', help='High confidence threshold')
    parser.add_argument('--low', type=str, default='0.1', help='Low confidence threshold')
    parser.add_argument('--base_dir', type=str, default=os.path.join('data', 'MLKL', 'test', 'v9'), help='Base directory for source data')
    # Parse command line parameters
    parse_config = parser.parse_args()
    return parse_config



def move_and_rename_files(csv_file_path, txt_source_dir, destination_dir, suffix):
    df = pd.read_csv(csv_file_path)
    
    # Make sure the target directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate over each file name in the file_name column
    for index, row in df.iterrows():
        file_name = row['name']
        score = row['score']
        
        # Replace the .npz suffix with .txt
        txt_file_name = file_name.replace('.npz', suffix)
        txt_file_path = os.path.join(txt_source_dir, txt_file_name)
        destination_file_path = os.path.join(destination_dir, txt_file_name)
        
        # Check if the .txt file exists
        if os.path.exists(txt_file_path):
            shutil.copy(txt_file_path, destination_dir)
            prefix, ext = os.path.splitext(file_name)
            new_file_name = prefix + f"_{score}" + suffix
            new_file_path = os.path.join(destination_dir, new_file_name)
            
            # Rename the copied file
            os.rename(destination_file_path, new_file_path)
        else:
            print(f"文件未找到: {txt_file_path}")





def process_and_sort_images(parse_config):
    # Define source image directory and target directory
    base_dir_png = os.path.join(parse_config.base_dir, 'processing_data', 'png')
    csv_file_path =  os.path.join('DeepSIFA', SAVE_LOG, 'score.csv')
    destination_dir = os.path.join('DeepSIFA', SAVE_LOG, 'scoresort')

    df = pd.read_csv(csv_file_path)
    # Sort by 'score' column in descending order
    sorted_df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # Make sure the target directory exists (if it exists, delete it first and then recreate it)
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    os.makedirs(destination_dir, exist_ok=True)
    
    # Traverse the sorted files, copy and rename them in order
    for index, row in sorted_df.iterrows():
        file_name = row['name']
        score = row['score']
        
        # Splice source file path and target file path
        source_file_path = os.path.join(base_dir_png, file_name.replace('.npz', '.png'))
        prefix, ext = os.path.splitext(file_name)
        destination_file_name = f"{index + 1}_" + prefix + f"_{score}.png" # Rename based on sorting index
        destination_file_path = os.path.join(destination_dir, destination_file_name)
        shutil.copy(source_file_path, destination_file_path)


    print(f"排序后的图片已保存到目录: {destination_dir}")





def count_images_and_log(parse_config):
    # defines three directory paths
    base_dir = os.path.join('DeepSIFA', SAVE_LOG, 'low{}_high{}').format(parse_config.low, parse_config.high)
    dirs = {
        'bad': os.path.join(base_dir, 'bad_图片'),
        'good': os.path.join(base_dir, 'good_图片'),
        'ambiguous': os.path.join(base_dir, 'vague_图片')
    }
    
    log_file = os.path.join(base_dir, 'log.txt')
    
    with open(log_file, 'w') as f:
        for category, path in dirs.items():
            if os.path.exists(path):
                # counts the number of pictures in the directory
                image_count = len([file for file in os.listdir(path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
                # Write to log file
                f.write(f'{category}: {image_count}\n')
                # print(f'{category} has {image_count} images')
            else:
                f.write(f'{category} directory does not exist\n')
                print(f'{category} directory does not exist')
    
    print(f"Results saved to {log_file}")




parse_config = get_cfg()
print('分层 base_dir路径', parse_config.base_dir)
base_dir_txt = os.path.join(parse_config.base_dir, 'row_data')
base_dir_png = os.path.join(parse_config.base_dir, 'processing_data', 'png')
deep_sifa_dir = os.path.join('DeepSIFA', SAVE_LOG)
csv_file_path =  os.path.join(deep_sifa_dir, 'score.csv')

df = pd.read_csv(csv_file_path)
# Convert each row into a dictionary to form an allscore list
allscore = df.to_dict(orient='records')
# initializes two lists
subscore1 = []
subscore2 = []
subscore3 = []
# Traverse the allscore list
for result in allscore:
    if result['score'] >= float(parse_config.high):
        subscore1.append(result)
    if result['score'] <= float(parse_config.low):
        subscore2.append(result)
    if result['score'] > float(parse_config.low) and result['score'] < float(parse_config.high):
        subscore3.append(result)

# Save subscore1, subscore2, subscore3 as different CSV files
directory = os.path.join(deep_sifa_dir, 'low{}_high{}'.format(parse_config.low, parse_config.high))
if os.path.exists(directory):
    shutil.rmtree(directory)  # Delete the entire directory
os.makedirs(directory, exist_ok=True)  # Re-create directory
pd.DataFrame(subscore1).to_csv(os.path.join(directory, 'scoreHIGH.csv'), index=False)
pd.DataFrame(subscore2).to_csv(os.path.join(directory, 'scoreLOW.csv'), index=False)
pd.DataFrame(subscore3).to_csv(os.path.join(directory, 'scoreMIDDLE.csv'), index=False)


# High confidence
move_and_rename_files(
    csv_file_path=os.path.join(directory, 'scoreHIGH.csv'),
    txt_source_dir=base_dir_txt,
    destination_dir=os.path.join(directory, 'good_txt'),
    suffix='.txt'
)

# low confidence
move_and_rename_files(
    csv_file_path=os.path.join(directory, 'scoreLOW.csv'),
    txt_source_dir=base_dir_txt,
    destination_dir=os.path.join(directory, 'bad_txt'),
    suffix='.txt'
)

# blur
move_and_rename_files(
    csv_file_path=os.path.join(directory, 'scoreMIDDLE.csv'),
    txt_source_dir=base_dir_txt,
    destination_dir=os.path.join(directory, 'vague_txt'),
    suffix='.txt'
)

# high confidence (png)
move_and_rename_files(
    csv_file_path=os.path.join(directory, 'scoreHIGH.csv'),
    txt_source_dir=base_dir_png,
    destination_dir=os.path.join(directory, 'good_图片'),
    suffix='.png'
)

# low confidence (png)
move_and_rename_files(
    csv_file_path=os.path.join(directory, 'scoreLOW.csv'),
    txt_source_dir=base_dir_png,
    destination_dir=os.path.join(directory, 'bad_图片'),
    suffix='.png'
)

# blur (png)
move_and_rename_files(
    csv_file_path=os.path.join(directory, 'scoreMIDDLE.csv'),
    txt_source_dir=base_dir_png,
    destination_dir=os.path.join(directory, 'vague_图片'),
    suffix='.png'
)


count_images_and_log(parse_config)
process_and_sort_images(parse_config)
print("分层 完成")











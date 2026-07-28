# # 1 Visually grasp the prediction errors and save them to fold{}/score.csv
# import pandas as pd

# # Read the original score.csv file
# score_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5fold/logs_test_5/fold3/score.csv'
# data = pd.read_csv(score_file_path)

# # Find rows where the label column and pre column are not equal
# wrong_data = data[data['label'] != data['pre']]

# # Save error lines to wrong.csv file
# wrong_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5fold/logs_test_5/fold3/wrong.csv'
# wrong_data.to_csv(wrong_file_path, index=False)




# # 2 Visualize the values less than 70 in fold{}/score.csv
# import pandas as pd

# # Read the original wrong.csv file
# wrong_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5fold/logs_test_5/fold3/wrong.csv'
# data = pd.read_csv(wrong_file_path)

# # Find the rows where the score column is less than or equal to 0.70
# wrong_070_data = data[data['score'] <= 0.70]

# # Save the rows where the score column is less than or equal to 0.70 to the wrong_0.7.csv file
# wrong_070_file_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5fold/logs_test_5/fold3/wrong_0.7.csv'
# wrong_070_data.to_csv(wrong_070_file_path, index=False)



# # # 3 Copy the incorrectly predicted .tiff image to /fold3/wrong_img
# import os
# import shutil
# import pandas as pd

# # Read wrong.csv file
# wrong_csv_path = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5fold/logs_test_5/fold3/wrong.csv'
# data = pd.read_csv(wrong_csv_path)

# # Original data directory and target directory
# source_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/data/F/fifth batch/original data'
# target_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5fold/logs_test_5/fold3/wrong_img'

# # Make sure the target directory exists
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)

# # Traverse each row in wrong.csv
# for index, row in data.iterrows():
#     # Get file name
#     file_name = row['name'].replace('.npz', '.tiff')
#     # Build original file path and target file path
#     source_file_path = os.path.join(source_dir, file_name)
#     target_file_path = os.path.join(target_dir, file_name)
#     # Copy the file and change the .npz file to a .tiff file
#     shutil.copy(source_file_path, target_file_path)



# #4 Remove the suffix name
import os
import shutil
target_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong_img'
filtered_target_dir = '/data2/linchen_data/linchen/DeepFRET-Model-master/LSTMF_v13.9.6.1_5折/logs_test_5/fold3/wrong_img筛选'

if not os.path.exists(filtered_target_dir):
    os.makedirs(filtered_target_dir)

# Traverse all files in the target directory
for filename in os.listdir(target_dir):
    if filename.endswith('.tiff'):
        # Get the path of the file
        file_path = os.path.join(target_dir, filename)
        # Remove the content after the last underscore in the file name
        new_filename = filename.rsplit('_', 1)[0] + '.tiff'
        # Path to new file
        new_file_path = os.path.join(filtered_target_dir, new_filename)
        # Copy files to new target directory
        shutil.copyfile(file_path, new_file_path)







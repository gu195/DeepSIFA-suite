import os

# Define directory path
# directory = '/home/node01/linchen/data/K10/training set/1'
# directory = '/home/node01/linchen/data/K10/training set/2'
# directory = '/home/node01/linchen/data/K10/validation set/1'
directory = '/home/node01/linchen/data/K10/验证集/2'


# Traverse files in the directory
for filename in os.listdir(directory):

    # Generate new file name
    new_filename = filename.replace('_good', '').replace('_bad', '')
    # Get the complete file path
    old_filepath = os.path.join(directory, filename)
    new_filepath = os.path.join(directory, new_filename)
    # Rename file
    os.rename(old_filepath, new_filepath)
    print(f'Renamed: {filename} -> {new_filename}')

import pandas as pd
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

FOLD = '5'
# definition file path
data_dir = '/home/node01/linchen/data/alphaK10/V2/alphak10-test/归一化插值后npz_1024_高斯平滑1/png1'
target_dir1 = f'./logs_test验证/fold{FOLD}/png0_0.3/'
target_dir2 = f'./logs_test验证/fold{FOLD}/png0.3_0.7/'
target_dir3 = f'./logs_test验证/fold{FOLD}/png0.7_1/'
input_file = f'./logs_test验证/fold{FOLD}/score.csv'
os.makedirs(target_dir1, exist_ok=True)
os.makedirs(target_dir2, exist_ok=True)
os.makedirs(target_dir3, exist_ok=True)

# Read CSV file
df = pd.read_csv(input_file)
# Filters out data with scores within the specified range
df = df[df['label'] == -1]
df_filtered1 = df[(df['score'] >= 0) & (df['score'] <= 0.3)]
df_filtered2 = df[(df['score'] > 0.3) & (df['score'] < 0.7)]
df_filtered3 = df[(df['score'] >= 0.7) & (df['score'] <= 1)]

# Get the list of files that meet the conditions
files_to_copy1 = df_filtered1[['name', 'score']]
files_to_copy2 = df_filtered2[['name', 'score']]
files_to_copy3 = df_filtered3[['name', 'score']]

def add_confidence_to_image(image_path, score, output_path):
    # Open picture
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Replace with the font path on your system
    font_size = 30  # Adjust font size
    font = ImageFont.truetype(font_path, font_size)

    # Add confidence text
    text = f'Confidence: {score:.4f}'
    text_position = (1050, 10)  # Text position, upper left corner
    text_color = (255, 0, 0)  # text color, red
    draw.text(text_position, text, fill=text_color, font=font)

    # Save image with text
    img.save(output_path)

# Copy the file and add confidence information to the image
for _, row in files_to_copy1.iterrows():
    file = row['name']
    score = row['score']
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    target_file = os.path.join(target_dir1, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        add_confidence_to_image(source_file, score, target_file)

for _, row in files_to_copy2.iterrows():
    file = row['name']
    score = row['score']
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    target_file = os.path.join(target_dir2, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        add_confidence_to_image(source_file, score, target_file)

for _, row in files_to_copy3.iterrows():
    file = row['name']
    score = row['score']
    source_file = os.path.join(data_dir, file.replace('.npz', '.png'))
    target_file = os.path.join(target_dir3, file.replace('.npz', '.png'))
    if os.path.isfile(source_file):
        add_confidence_to_image(source_file, score, target_file)



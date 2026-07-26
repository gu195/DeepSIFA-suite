import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FOLD = '5'
# definition file path
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/acc_vs_threshold_{}_good.png'.format(FOLD, FOLD)

# Read CSV file
df = pd.read_csv(input_file)

# filters data with label 0
df_filtered = df[df['label'] == 0]
scores = df_filtered['score']

# Define threshold interval
thresholds = np.linspace(0, 1, 101)
accuracies = []

# Calculate the accuracy of each threshold
for threshold in thresholds:
    correct_predictions = (scores < threshold).sum()
    accuracy = correct_predictions / len(scores)
    accuracies.append(accuracy)

# Find the first time the accuracy exceeds a certain percentage threshold
threshold_50 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.50), None)
threshold_60 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.60), None)
threshold_70 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.70), None)
threshold_75 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.75), None)
threshold_80 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.80), None)
threshold_85 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.85), None)
threshold_90 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.90), None)
threshold_95 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.95), None)

# Draw a histogram of threshold vs. accuracy
plt.figure(figsize=(8, 8))
plt.bar(thresholds, accuracies, width=0.01, edgecolor='black', alpha=0.7)

# Add title and tags
plt.title('Negative precision-Thresholds (Total: {})'.format(len(scores)))
plt.xlabel('Thresholds')
plt.ylabel('Negative precision')

# Add grid
plt.grid(axis='y', alpha=0.75)

# Set the range of the x-axis and the y-axis so that 0 on the x-axis coincides with 0 on the y-axis
plt.xlim(0, 1)
plt.ylim(0, 1)

# Calculate the percentage of each column and display it above the column
for threshold, accuracy in zip(thresholds, accuracies):
    percentage = f'{(accuracy * 100):.1f}%'
    plt.text(threshold, accuracy, percentage, ha='center', va='bottom', fontsize=2, color='black')

# Marks the first time the threshold exceeds a specific percentage and the corresponding accuracy value
def annotate_threshold(threshold, accuracy, color):
    if threshold is not None:
        plt.plot([threshold, threshold], [0, accuracy], color=color, linestyle='--')
        plt.plot([0, threshold], [accuracy, accuracy], color=color, linestyle='--')
        plt.text(threshold, accuracy, f'{threshold:.2f}, {accuracy:.2f}', color='red', ha='center', va='bottom')

annotate_threshold(threshold_50, 0.5, 'black')
annotate_threshold(threshold_60, 0.60, 'black')
annotate_threshold(threshold_70, 0.7, 'black')
annotate_threshold(threshold_75, 0.75, 'black')
annotate_threshold(threshold_80, 0.80, 'black')
annotate_threshold(threshold_85, 0.85, 'black')
annotate_threshold(threshold_90, 0.90, 'black')
annotate_threshold(threshold_95, 0.95, 'black')

# Save chart to file
plt.savefig(output_file, dpi=1000)

print(f"柱状图已保存到 {output_file}")






#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FOLD = '5'
# definition file path
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/acc_vs_threshold_{}_bad.png'.format(FOLD, FOLD)

# Read CSV file
df = pd.read_csv(input_file)

# filters data with label 0
df_filtered = df[df['label'] == 1]
scores = df_filtered['score']

# Define threshold interval
thresholds = np.linspace(0, 1, 101)
accuracies = []

# Calculate the accuracy of each threshold
for threshold in thresholds:
    correct_predictions = (scores > threshold).sum()
    accuracy = correct_predictions / len(scores)
    accuracies.append(accuracy)

# Find the first time the accuracy exceeds a certain percentage threshold
threshold_50 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.50), None)
threshold_60 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.60), None)
threshold_70 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.70), None)
threshold_75 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.75), None)
threshold_80 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.80), None)
threshold_85 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.85), None)
threshold_90 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.90), None)
threshold_95 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.95), None)

# Draw a histogram of threshold vs. accuracy
plt.figure(figsize=(8, 8))
plt.bar(thresholds, accuracies, width=0.01, edgecolor='black', alpha=0.7)

# Add title and tags
plt.title('Positive precision-Thresholds (Total: {})'.format(len(scores)))
plt.xlabel('Thresholds')
plt.ylabel('Positive precision')

# Add grid
plt.grid(axis='y', alpha=0.75)

# Set the range of the x-axis and the y-axis so that 0 on the x-axis coincides with 0 on the y-axis
plt.xlim(0, 1)
plt.ylim(0, 1)

# Calculate the percentage of each column and display it above the column
for threshold, accuracy in zip(thresholds, accuracies):
    percentage = f'{(accuracy * 100):.1f}%'
    plt.text(threshold, accuracy, percentage, ha='center', va='bottom', fontsize=2, color='black')

# Marks the first time the threshold exceeds a specific percentage and the corresponding accuracy value
def annotate_threshold(threshold, accuracy, color):
    if threshold is not None:
        plt.plot([threshold, threshold], [0, accuracy], color=color, linestyle='--')
        plt.plot([0, threshold], [accuracy, accuracy], color=color, linestyle='--')
        plt.text(threshold, accuracy, f'{threshold:.2f}, {accuracy:.2f}', color='red', ha='center', va='bottom')

annotate_threshold(threshold_50, 0.5, 'black')
annotate_threshold(threshold_60, 0.60, 'black')
annotate_threshold(threshold_70, 0.7, 'black')
annotate_threshold(threshold_75, 0.75, 'black')
annotate_threshold(threshold_80, 0.80, 'black')
annotate_threshold(threshold_85, 0.85, 'black')
annotate_threshold(threshold_90, 0.90, 'black')
annotate_threshold(threshold_95, 0.95, 'black')

# Save chart to file
plt.savefig(output_file, dpi=1000)

print(f"柱状图已保存到 {output_file}")


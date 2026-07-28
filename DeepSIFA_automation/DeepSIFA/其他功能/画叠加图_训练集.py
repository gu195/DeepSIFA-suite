import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FOLD = '5'
# definition file path
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/acc_vs_threshold_{}_line_adjusted.png'.format(FOLD, FOLD)


df = pd.read_csv(input_file)
# filters data with label 0
df_filtered0 = df[df['label'] == 0]
scores0 = df_filtered0['score']
# filters data with label 1
df_filtered1 = df[df['label'] == 1]
scores1 = df_filtered1['score']
# Define threshold interval
thresholds = np.linspace(0, 1, 101)
# Calculate the accuracy of each threshold
accuracies0 = []
accuracies1 = []

for threshold in thresholds:
    # Good accuracy
    correct_predictions0 = (scores0 < threshold).sum()
    accuracy0 = correct_predictions0 / len(scores0)
    accuracies0.append(accuracy0)
    
    # Bad accuracy
    correct_predictions1 = (scores1 >= threshold).sum()
    accuracy1 = correct_predictions1 / len(scores1)
    accuracies1.append(accuracy1)

# Draw a line chart of threshold vs. accuracy
plt.figure(figsize=(8, 8))
# Good accuracy line chart
plt.plot(thresholds, accuracies0, color='blue', label='Good Accuracy')
# Bad accuracy line chart
plt.plot(thresholds, accuracies1, color='red', label='Bad Accuracy')

def find_closest_accuracy(thresholds, accuracies1, target_accuracy):
    closest_accuracy = min(accuracies1, key=lambda x: abs(x - target_accuracy))
    index_closest_accuracy = accuracies1.index(closest_accuracy)
    threshold_at_closest_accuracy = thresholds[index_closest_accuracy]
    return closest_accuracy, threshold_at_closest_accuracy

# Define target accuracy
target_accuracies = [0.95, 0.9, 0.85]

# Traverse each target accuracy rate and find the closest bad accuracy rate and corresponding threshold
for target_accuracy in target_accuracies:
    closest_accuracy, threshold_at_closest_accuracy = find_closest_accuracy(thresholds, accuracies1, target_accuracy)
    print(f"Closest bad accuracy to {target_accuracy}: {closest_accuracy:.4f}")
    print(f"Threshold at closest accuracy: {threshold_at_closest_accuracy}")
    idx = int(threshold_at_closest_accuracy*100)
    print(f"Good Curve accuracy: {accuracies0[idx]}")
    GOOD_closest_accuracy = accuracies0[idx]

    # Find the coordinates of the intersection point
    idx_closest_accuracy = np.argmin(np.abs(np.array(accuracies1) - closest_accuracy))
    idx_threshold = np.argmin(np.abs(thresholds - threshold_at_closest_accuracy))
    
    # Draw a line from the intersection point to the threshold
    plt.plot([thresholds[idx_threshold], thresholds[idx_threshold]], [closest_accuracy, 0], color='gray', linestyle='--')
    plt.text(thresholds[idx_threshold], closest_accuracy, f'({thresholds[idx_threshold]:.2f}, {closest_accuracy:.3f})',fontsize=8, verticalalignment='bottom', horizontalalignment='center')
    plt.text(thresholds[idx_threshold], GOOD_closest_accuracy, f'({thresholds[idx_threshold]:.2f}, {GOOD_closest_accuracy:.3f})',fontsize=8, verticalalignment='top', horizontalalignment='center')
    # # Bad curve Draw a line from the intersection point to the y-axis
    # plt.plot([thresholds[idx_threshold], 0], [closest_accuracy, closest_accuracy], color='gray', linestyle='--')
    # # Good curve draws the line from the intersection point to closest_accuracy
    # plt.plot([thresholds[idx_threshold], 0], [GOOD_closest_accuracy, GOOD_closest_accuracy], color='gray', linestyle='--')




# Add title and tags
plt.title('Good and Bad Curve precision (Total Good: {}, Total Bad: {})'.format(len(scores0), len(scores1)))
plt.xlabel('Threshold')
plt.ylabel('Precision')


# Add legend
plt.legend()
# Add grid
plt.grid(False)

# Set the scale range and origin position of the x-axis and y-axis
# Set the starting position of the coordinate axis
plt.xlim(0, 1)
plt.ylim(0, 1)
# Set x-axis scale
plt.xticks(np.linspace(0, 1, 11))  # Set the scale to 0.0, 0.1, ..., 1.0
plt.yticks(np.linspace(0, 1, 11))  # Set the scale to 0.0, 0.1, ..., 1.0

# Save chart to file
plt.savefig(output_file, dpi=500)
print(f"折线图已保存到 {output_file}")



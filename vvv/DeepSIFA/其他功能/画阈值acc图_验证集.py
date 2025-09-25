import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FOLD = '5'
# 定义文件路径
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold{}/score.csv'.format(FOLD)
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold{}/acc_vs_threshold_{}_good.png'.format(FOLD, FOLD)

# 读取CSV文件
df = pd.read_csv(input_file)

# 过滤 label 为 0 的数据
df_filtered = df[df['label'] == 0]
scores = df_filtered['score']

# 定义阈值区间
thresholds = np.linspace(0, 1, 101)
accuracies = []

# 计算每个阈值的准确率
for threshold in thresholds:
    correct_predictions = (scores < threshold).sum()
    accuracy = correct_predictions / len(scores)
    accuracies.append(accuracy)

# 找出准确率首次超过特定百分比的阈值
threshold_50 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.50), None)
threshold_60 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.60), None)
threshold_70 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.70), None)
threshold_75 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.75), None)
threshold_80 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.80), None)
threshold_85 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.85), None)
threshold_90 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.90), None)
threshold_95 = next((t for t, a in zip(thresholds, accuracies) if a >= 0.95), None)

# 绘制阈值 vs 准确率的柱状图
plt.figure(figsize=(8, 8))
plt.bar(thresholds, accuracies, width=0.01, edgecolor='black', alpha=0.7)

# 添加标题和标签
plt.title('Negative precision-Thresholds (Total: {})'.format(len(scores)))
plt.xlabel('Thresholds')
plt.ylabel('Negative precision')

# 添加网格
plt.grid(axis='y', alpha=0.75)

# 设置x轴和y轴的范围，使得x轴的0和y轴的0重合
plt.xlim(0, 1)
plt.ylim(0, 1)

# 计算每个柱子的百分比并显示在柱子上方
for threshold, accuracy in zip(thresholds, accuracies):
    percentage = f'{(accuracy * 100):.1f}%'
    plt.text(threshold, accuracy, percentage, ha='center', va='bottom', fontsize=2, color='black')

# 标注首次超过特定百分比的阈值和对应的准确率值
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

# 保存图表到文件
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
# 定义文件路径
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold{}/score.csv'.format(FOLD)
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold{}/acc_vs_threshold_{}_bad.png'.format(FOLD, FOLD)

# 读取CSV文件
df = pd.read_csv(input_file)

# 过滤 label 为 0 的数据
df_filtered = df[df['label'] == 1]
scores = df_filtered['score']

# 定义阈值区间
thresholds = np.linspace(0, 1, 101)
accuracies = []

# 计算每个阈值的准确率
for threshold in thresholds:
    correct_predictions = (scores > threshold).sum()
    accuracy = correct_predictions / len(scores)
    accuracies.append(accuracy)

# 找出准确率首次超过特定百分比的阈值
threshold_50 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.50), None)
threshold_60 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.60), None)
threshold_70 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.70), None)
threshold_75 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.75), None)
threshold_80 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.80), None)
threshold_85 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.85), None)
threshold_90 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.90), None)
threshold_95 = next((t for t, a in zip(thresholds, accuracies) if a <= 0.95), None)

# 绘制阈值 vs 准确率的柱状图
plt.figure(figsize=(8, 8))
plt.bar(thresholds, accuracies, width=0.01, edgecolor='black', alpha=0.7)

# 添加标题和标签
plt.title('Positive precision-Thresholds (Total: {})'.format(len(scores)))
plt.xlabel('Thresholds')
plt.ylabel('Positive precision')

# 添加网格
plt.grid(axis='y', alpha=0.75)

# 设置x轴和y轴的范围，使得x轴的0和y轴的0重合
plt.xlim(0, 1)
plt.ylim(0, 1)

# 计算每个柱子的百分比并显示在柱子上方
for threshold, accuracy in zip(thresholds, accuracies):
    percentage = f'{(accuracy * 100):.1f}%'
    plt.text(threshold, accuracy, percentage, ha='center', va='bottom', fontsize=2, color='black')

# 标注首次超过特定百分比的阈值和对应的准确率值
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

# 保存图表到文件
plt.savefig(output_file, dpi=1000)

print(f"柱状图已保存到 {output_file}")


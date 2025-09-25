import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FOLD = '5'
# 定义文件路径
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_train验证/fold{}/acc_vs_threshold_{}_line_adjusted.png'.format(FOLD, FOLD)


df = pd.read_csv(input_file)
# 过滤 label 为 0 的数据
df_filtered0 = df[df['label'] == 0]
scores0 = df_filtered0['score']
# 过滤 label 为 1 的数据
df_filtered1 = df[df['label'] == 1]
scores1 = df_filtered1['score']
# 定义阈值区间
thresholds = np.linspace(0, 1, 101)
# 计算每个阈值的准确率
accuracies0 = []
accuracies1 = []

for threshold in thresholds:
    # 好的准确率
    correct_predictions0 = (scores0 < threshold).sum()
    accuracy0 = correct_predictions0 / len(scores0)
    accuracies0.append(accuracy0)
    
    # 坏的准确率
    correct_predictions1 = (scores1 >= threshold).sum()
    accuracy1 = correct_predictions1 / len(scores1)
    accuracies1.append(accuracy1)

# 绘制阈值 vs 准确率的折线图
plt.figure(figsize=(8, 8))
# 好的准确率折线图
plt.plot(thresholds, accuracies0, color='blue', label='Good Accuracy')
# 坏的准确率折线图
plt.plot(thresholds, accuracies1, color='red', label='Bad Accuracy')

def find_closest_accuracy(thresholds, accuracies1, target_accuracy):
    closest_accuracy = min(accuracies1, key=lambda x: abs(x - target_accuracy))
    index_closest_accuracy = accuracies1.index(closest_accuracy)
    threshold_at_closest_accuracy = thresholds[index_closest_accuracy]
    return closest_accuracy, threshold_at_closest_accuracy

# 定义目标准确率
target_accuracies = [0.95, 0.9, 0.85]

# 遍历每个目标准确率，找到最接近的坏准确率和对应的阈值
for target_accuracy in target_accuracies:
    closest_accuracy, threshold_at_closest_accuracy = find_closest_accuracy(thresholds, accuracies1, target_accuracy)
    print(f"Closest bad accuracy to {target_accuracy}: {closest_accuracy:.4f}")
    print(f"Threshold at closest accuracy: {threshold_at_closest_accuracy}")
    idx = int(threshold_at_closest_accuracy*100)
    print(f"Good Curve accuracy: {accuracies0[idx]}")
    GOOD_closest_accuracy = accuracies0[idx]

    # 找到交点的坐标
    idx_closest_accuracy = np.argmin(np.abs(np.array(accuracies1) - closest_accuracy))
    idx_threshold = np.argmin(np.abs(thresholds - threshold_at_closest_accuracy))
    
    # 绘制从交点到threshold的连线
    plt.plot([thresholds[idx_threshold], thresholds[idx_threshold]], [closest_accuracy, 0], color='gray', linestyle='--')
    plt.text(thresholds[idx_threshold], closest_accuracy, f'({thresholds[idx_threshold]:.2f}, {closest_accuracy:.3f})',fontsize=8, verticalalignment='bottom', horizontalalignment='center')
    plt.text(thresholds[idx_threshold], GOOD_closest_accuracy, f'({thresholds[idx_threshold]:.2f}, {GOOD_closest_accuracy:.3f})',fontsize=8, verticalalignment='top', horizontalalignment='center')
    # # 坏曲线            绘制从交点到y轴的连线
    # plt.plot([thresholds[idx_threshold], 0], [closest_accuracy, closest_accuracy], color='gray', linestyle='--')
    # # 好曲线        绘制从交点到closest_accuracy的连线
    # plt.plot([thresholds[idx_threshold], 0], [GOOD_closest_accuracy, GOOD_closest_accuracy], color='gray', linestyle='--')




# 添加标题和标签
plt.title('Good and Bad Curve precision (Total Good: {}, Total Bad: {})'.format(len(scores0), len(scores1)))
plt.xlabel('Threshold')
plt.ylabel('Precision')


# 添加图例
plt.legend()
# 添加网格
plt.grid(False)

# 设置x轴和y轴的刻度范围及原点位置
# 设置坐标轴的起始位置
plt.xlim(0, 1)
plt.ylim(0, 1)
# 设置 x 轴刻度
plt.xticks(np.linspace(0, 1, 11))  # 设置刻度为 0.0, 0.1, ..., 1.0
plt.yticks(np.linspace(0, 1, 11))  # 设置刻度为 0.0, 0.1, ..., 1.0

# 保存图表到文件
plt.savefig(output_file, dpi=500)
print(f"折线图已保存到 {output_file}")



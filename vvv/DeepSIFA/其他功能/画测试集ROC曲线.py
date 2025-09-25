import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os

# 读取 CSV 文件
csv_file_path = '/home/node01/实验数据/Vit+PRM/logs_test_5/fold3/score.csv'
data = pd.read_csv(csv_file_path)

# 提取标签和预测得分
y_test = data['label']
y_score = data['score']

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存图表
output_directory = '/home/node01/实验数据/Vit+PRM/logs_test_5/fold3'
image_filename = os.path.join(output_directory, 'roc_curve.png')
plt.savefig(image_filename, dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

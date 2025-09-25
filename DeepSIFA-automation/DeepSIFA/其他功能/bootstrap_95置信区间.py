import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

# 加载数据集
# output_file_path = './logs_train验证/fold5/95置信度.txt'
# data = pd.read_csv('./logs_train验证/fold5/score.csv')

# output_file_path = './logs_val验证/fold2/95置信度.txt'
# data = pd.read_csv('./logs_val验证/fold2/score.csv')

output_file_path = './logs_test_5/fold1/95置信度.txt'
data = pd.read_csv('./logs_test_5/fold1/score.csv')



import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score

# 假设 y_test, y_pred, y_pred_proba 以及 output_file_path 都已经定义
y_test = data['label']
y_pred = data['pre']    # 是1还是0
y_pred_proba = data['pre']  # 是1还是0的具体概率
auc = roc_auc_score(y_test, y_pred_proba)

# 自助法计算95%置信区间
n_bootstraps = 2000
auc_bootstrap = []
accuracy_bootstrap = []
specificity_bootstrap = []
sensitivity_bootstrap = []
f1_score_bootstrap = []
ppv_bootstrap = []
npv_bootstrap = []

for _ in range(n_bootstraps):
    indices = np.random.choice(len(y_test), len(y_test), replace=True)
    y_test_bootstrap = y_test.iloc[indices]
    y_pred_proba_bootstrap = y_pred_proba[indices]
    y_pred_bootstrap = (y_pred_proba_bootstrap > 0.5).astype(int)

    auc_bootstrap.append(roc_auc_score(y_test_bootstrap, y_pred_proba_bootstrap))
    accuracy_bootstrap.append(accuracy_score(y_test_bootstrap, y_pred_bootstrap))
    tn, fp, fn, tp = confusion_matrix(y_test_bootstrap, y_pred_bootstrap).ravel()
    specificity_bootstrap.append(tn / (tn + fp))
    sensitivity_bootstrap.append(tp / (tp + fn))
    ppv_bootstrap.append(tp / (tp + fp) if (tp + fp) != 0 else np.nan)
    npv_bootstrap.append(tn / (tn + fn) if (tn + fn) != 0 else np.nan)
    f1_score_bootstrap.append(f1_score(y_test_bootstrap, y_pred_bootstrap))

# 计算95%置信区间
confidence_level = 0.95
alpha = 1 - confidence_level
lower_percentile = alpha / 2 * 100
upper_percentile = (1 - alpha / 2) * 100

auc_ci = np.percentile(auc_bootstrap, [lower_percentile, upper_percentile])
accuracy_ci = np.percentile(accuracy_bootstrap, [lower_percentile, upper_percentile])
specificity_ci = np.percentile(specificity_bootstrap, [lower_percentile, upper_percentile])
sensitivity_ci = np.percentile(sensitivity_bootstrap, [lower_percentile, upper_percentile])
f1_score_ci = np.percentile(f1_score_bootstrap, [lower_percentile, upper_percentile])
ppv_ci = np.percentile(ppv_bootstrap, [lower_percentile, upper_percentile])
npv_ci = np.percentile(npv_bootstrap, [lower_percentile, upper_percentile])

# 计算总体的性能指标
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
ppv = tp / (tp + fp) if (tp + fp) != 0 else np.nan
npv = tn / (tn + fn) if (tn + fn) != 0 else np.nan
f1 = f1_score(y_test, y_pred)

# 保存结果到文件
with open(output_file_path, 'w') as file:
    file.write(f"Precision: {round(ppv * 100, 1)}({round(ppv_ci[0] * 100, 1)},{round(ppv_ci[1] * 100, 1)})\n")
    file.write(f"Recall: {round(sensitivity * 100, 1)}({round(sensitivity_ci[0] * 100, 1)},{round(sensitivity_ci[1] * 100, 1)})\n")
    file.write(f"F1 Score: {round(f1 * 100, 1)}({round(f1_score_ci[0] * 100, 1)},{round(f1_score_ci[1] * 100, 1)})\n")
    file.write(f"Accuracy: {round(accuracy * 100, 1)}({round(accuracy_ci[0] * 100, 1)},{round(accuracy_ci[1] * 100, 1)})\n")
    file.write(f"AUC: {round(auc * 100, 1)}({round(auc_ci[0] * 100, 1)},{round(auc_ci[1] * 100, 1)})\n")

    file.write(f"NPV: {round(npv * 100, 1)}({round(npv_ci[0] * 100, 1)},{round(npv_ci[1] * 100, 1)})\n")
    file.write(f"Specificity: {round(specificity * 100, 1)}({round(specificity_ci[0] * 100, 1)},{round(specificity_ci[1] * 100, 1)})\n")







# # 画概率图 全部
import pandas as pd
import matplotlib.pyplot as plt
FOLD = '3'
# 定义文件路径
input_file = './logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = './logs_train验证/fold{}/score_distribution_{}.png'.format(FOLD,FOLD)


# 读取CSV文件
df = pd.read_csv(input_file)

# 获取score列
scores = df['score']

# 绘制概率柱状图
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)

# 获取总条数
total = len(scores)
# 添加标题和标签，包括总条数
plt.title(f'Score Distribution (Total: {total})')
plt.xlabel('Score')
plt.ylabel('Frequency')

# 添加网格
plt.grid(axis='y', alpha=0.75)

# 计算每个柱子的百分比并显示在柱子上方
total = len(scores)
for count, bin_edge in zip(counts, bins):
    percentage = f'{(count / total * 100):.1f}%'
    plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, percentage,
             ha='center', va='bottom', fontsize=4, color='black')

# 保存图表到文件
plt.savefig(output_file,dpi=500)

print(f"柱状图已保存到 {output_file}")






#############################################################################################
#############################################################################################
#############################################################################################
#############画好曲线的概率图
import pandas as pd
import matplotlib.pyplot as plt
FOLD = '3'
# 定义文件路径
input_file = './logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = './logs_train验证/fold{}/score_distribution_{}_good.png'.format(FOLD,FOLD)

# 读取CSV文件
df = pd.read_csv(input_file)


df_filtered = df[df['label'] == 0]
scores = df_filtered['score']

# 绘制概率柱状图
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)

# 获取总条数
total = len(scores)
# 添加标题和标签
plt.title(f'Good Curve Score Distribution (Total: {total})')
plt.xlabel('Score')
plt.ylabel('Frequency')

# 添加网格
plt.grid(axis='y', alpha=0.75)

# 计算每个柱子的百分比并显示在柱子上方
total = len(scores)
for count, bin_edge in zip(counts, bins):
    percentage = f'{(count / total * 100):.1f}%'
    plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, percentage,
             ha='center', va='bottom', fontsize=4, color='black')

# 保存图表到文件
plt.savefig(output_file,dpi=500)

print(f"柱状图已保存到 {output_file}")









# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############画好曲线的概率图
import pandas as pd
import matplotlib.pyplot as plt
FOLD = '3'
# 定义文件路径
input_file = './logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = './logs_train验证/fold{}/score_distribution_{}_bad.png'.format(FOLD,FOLD)

# 读取CSV文件
df = pd.read_csv(input_file)


df_filtered = df[df['label'] == 1]
scores = df_filtered['score']

# 绘制概率柱状图
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)
total = len(scores)

# 添加标题和标签
plt.title(f'Bad Curve Score Distribution (Total: {total})')
plt.xlabel('Score')
plt.ylabel('Frequency')

# 添加网格
plt.grid(axis='y', alpha=0.75)

# 计算每个柱子的百分比并显示在柱子上方
total = len(scores)
for count, bin_edge in zip(counts, bins):
    percentage = f'{(count / total * 100):.1f}%'
    plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, percentage,
             ha='center', va='bottom', fontsize=4, color='black')

# 保存图表到文件
plt.savefig(output_file,dpi=500)

print(f"柱状图已保存到 {output_file}")

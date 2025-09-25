# import pandas as pd

# # 定义文件路径
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score.csv'
# output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score_大于0.8.csv'

# # 读取原始CSV文件
# df = pd.read_csv(input_file)

# # 筛选出score列中大于0.9的数据
# filtered_df = df[df['score'] > 0.8]

# # 将筛选后的数据保存到新的CSV文件
# filtered_df.to_csv(output_file, index=False)

# print(f"保存成功，路径为: {output_file}")


# #####################################################################################################################
# import pandas as pd

# # 定义文件路径
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score_大于0.8.csv'
# output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score_大于0.8_wrong.csv'
# # input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4/logs_val验证/fold4/score_小于0.9.csv'
# # output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4/logs_val验证/fold4/score_小于0.9_wrong.csv'
# # 读取CSV文件
# df = pd.read_csv(input_file)

# # 筛选出label和pre不相等的数据
# filtered_df = df[df['label'] != df['pre']]

# # 将筛选后的数据保存到新的CSV文件
# filtered_df.to_csv(output_file, index=False)

# print(f"保存成功，路径为: {output_file}")



# # ################################################################################################################
# import pandas as pd

# # 定义文件路径
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score_大于0.8.csv'

# # 读取CSV文件
# df = pd.read_csv(input_file)

# # 统计pre列中0和1的数量
# count_0 = (df['pre'] == 0).sum()
# count_1 = (df['pre'] == 1).sum()

# print(f"pre列中0的数量: {count_0}")
# print(f"pre列中1的数量: {count_1}")


# import pandas as pd
# import matplotlib.pyplot as plt

# # 定义文件路径
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score.csv'
# output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score_distribution.png'

# # 读取CSV文件
# df = pd.read_csv(input_file)

# # 获取score列
# scores = df['score']

# # 绘制概率柱状图
# plt.figure(figsize=(10, 6))
# plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)

# # 添加标题和标签
# plt.title('Score Distribution')
# plt.xlabel('Score')
# plt.ylabel('Frequency')

# # 显示图表
# plt.grid(axis='y', alpha=0.75)
# # 保存图表到文件
# plt.savefig(output_file)


import pandas as pd
import matplotlib.pyplot as plt

# 定义文件路径
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score.csv'
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score_distribution.png'

# 读取CSV文件
df = pd.read_csv(input_file)

# 获取score列
scores = df['score']

# 绘制概率柱状图
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)

# 添加标题和标签
plt.title('Score Distribution')
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


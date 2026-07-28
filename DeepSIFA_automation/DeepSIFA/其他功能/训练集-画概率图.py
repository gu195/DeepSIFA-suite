# # Draw probability graph All
import pandas as pd
import matplotlib.pyplot as plt
FOLD = '3'
# definition file path
input_file = './logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = './logs_train验证/fold{}/score_distribution_{}.png'.format(FOLD,FOLD)


# Read CSV file
df = pd.read_csv(input_file)

# Get the score column
scores = df['score']

# Draw probability histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)

# Get the total number of items
total = len(scores)
# Add title and tag, including total number of items
plt.title(f'Score Distribution (Total: {total})')
plt.xlabel('Score')
plt.ylabel('Frequency')

# Add grid
plt.grid(axis='y', alpha=0.75)

# Calculate the percentage of each column and display it above the column
total = len(scores)
for count, bin_edge in zip(counts, bins):
    percentage = f'{(count / total * 100):.1f}%'
    plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, percentage,
             ha='center', va='bottom', fontsize=4, color='black')

# Save chart to file
plt.savefig(output_file,dpi=500)

print(f"柱状图已保存到 {output_file}")






#############################################################################################
#############################################################################################
#############################################################################################
#############Draw the probability diagram of the curve
import pandas as pd
import matplotlib.pyplot as plt
FOLD = '3'
# definition file path
input_file = './logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = './logs_train验证/fold{}/score_distribution_{}_good.png'.format(FOLD,FOLD)

# Read CSV file
df = pd.read_csv(input_file)


df_filtered = df[df['label'] == 0]
scores = df_filtered['score']

# Draw probability histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)

# Get the total number of items
total = len(scores)
# Add title and tags
plt.title(f'Good Curve Score Distribution (Total: {total})')
plt.xlabel('Score')
plt.ylabel('Frequency')

# Add grid
plt.grid(axis='y', alpha=0.75)

# Calculate the percentage of each column and display it above the column
total = len(scores)
for count, bin_edge in zip(counts, bins):
    percentage = f'{(count / total * 100):.1f}%'
    plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, percentage,
             ha='center', va='bottom', fontsize=4, color='black')

# Save chart to file
plt.savefig(output_file,dpi=500)

print(f"柱状图已保存到 {output_file}")









# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############Draw the probability diagram of the curve
import pandas as pd
import matplotlib.pyplot as plt
FOLD = '3'
# definition file path
input_file = './logs_train验证/fold{}/score.csv'.format(FOLD)
output_file = './logs_train验证/fold{}/score_distribution_{}_bad.png'.format(FOLD,FOLD)

# Read CSV file
df = pd.read_csv(input_file)


df_filtered = df[df['label'] == 1]
scores = df_filtered['score']

# Draw probability histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)
total = len(scores)

# Add title and tags
plt.title(f'Bad Curve Score Distribution (Total: {total})')
plt.xlabel('Score')
plt.ylabel('Frequency')

# Add grid
plt.grid(axis='y', alpha=0.75)

# Calculate the percentage of each column and display it above the column
total = len(scores)
for count, bin_edge in zip(counts, bins):
    percentage = f'{(count / total * 100):.1f}%'
    plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, percentage,
             ha='center', va='bottom', fontsize=4, color='black')

# Save chart to file
plt.savefig(output_file,dpi=500)

print(f"柱状图已保存到 {output_file}")

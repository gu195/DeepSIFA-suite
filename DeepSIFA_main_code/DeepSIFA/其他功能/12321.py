# import pandas as pd

# # Define file path
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold1/score.csv'
# output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold1/score_greater than 0.8.csv'

# # Read the original CSV file
# df = pd.read_csv(input_file)

# # Filter out data greater than 0.9 in the score column
# filtered_df = df[df['score'] > 0.8]

# # Save filtered data to a new CSV file
# filtered_df.to_csv(output_file, index=False)

# print(f"Save successfully, path is: {output_file}")


# #####################################################################################################################
# import pandas as pd

# # Define file path
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold1/score_greater than 0.8.csv'
# output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold1/score_greater than 0.8_wrong.csv'
# # input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4/logs_val verification/fold4/score_less than 0.9.csv'
# # output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4/logs_valverification/fold4/score_less than 0.9_wrong.csv'
# # Read CSV file
# df = pd.read_csv(input_file)

# # Filter out data whose label and pre are not equal
# filtered_df = df[df['label'] != df['pre']]

# # Save filtered data to a new CSV file
# filtered_df.to_csv(output_file, index=False)

# print(f"Save successfully, path is: {output_file}")



# # ################################################################################################################
# import pandas as pd

# # Define file path
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold1/score_greater than 0.8.csv'

# # Read CSV file
# df = pd.read_csv(input_file)

# # Count the number of 0s and 1s in the pre column
# count_0 = (df['pre'] == 0).sum()
# count_1 = (df['pre'] == 1).sum()

# print(f"Number of 0s in column pre: {count_0}")
# print(f"Number of 1's in column pre: {count_1}")


# import pandas as pd
# import matplotlib.pyplot as plt

# # Define file path
# input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold1/score.csv'
# output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val verification/fold1/score_distribution.png'

# # Read CSV file
# df = pd.read_csv(input_file)

# # Get the score column
# scores = df['score']

# # Draw probability histogram
# plt.figure(figsize=(10, 6))
# plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)

# # Add title and tags
# plt.title('Score Distribution')
# plt.xlabel('Score')
# plt.ylabel('Frequency')

# # Display chart
# plt.grid(axis='y', alpha=0.75)
# # Save chart to file
# plt.savefig(output_file)


import pandas as pd
import matplotlib.pyplot as plt

# definition file path
input_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score.csv'
output_file = '/home/node01/linchen/Alphak10_trans_v3_new_v4.1/logs_val验证/fold1/score_distribution.png'

# Read CSV file
df = pd.read_csv(input_file)

# Get the score column
scores = df['score']

# Draw probability histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(scores, bins=40, edgecolor='black', alpha=0.7)

# Add title and tags
plt.title('Score Distribution')
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


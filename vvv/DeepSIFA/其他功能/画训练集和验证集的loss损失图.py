import matplotlib.pyplot as plt
import os

# 定义读取损失值的函数
def read_loss_from_file(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

# 指定损失值文件路径
train_loss_path = '/home/node01/实验数据/Vit+PRM/logs/fold4_v14/global_train_loss.txt'
val_loss_path = '/home/node01/实验数据/Vit+PRM/logs/fold4_v14/global_val_loss.txt'

# 从文件中读取损失值
global_train_loss = read_loss_from_file(train_loss_path)
global_val_loss = read_loss_from_file(val_loss_path)

# 绘制损失曲线
plt.figure(figsize=(5, 4))
# 确保 x 轴的值与数据点的数量相匹配
epochs = max(len(global_train_loss), len(global_val_loss))
x = range(epochs)

# 绘制训练损失曲线，使用红色粗实线
plt.plot(x[:len(global_train_loss)], global_train_loss, label='training loss', color='red', linewidth=3)
# 绘制验证损失曲线，使用蓝色粗实线
plt.plot(x[:len(global_val_loss)], global_val_loss, label='validation loss', color='#87CEFA', linewidth=3)

# 设置坐标轴的刻度和标签的参数
plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=4)
# plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=4)

# 设置坐标轴变粗
ax = plt.gca()  # 获取当前的轴
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
# 让 y 轴从 0 开始显示
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
# 保存图表
output_directory = '/home/node01/实验数据/Vit+PRM/logs/fold4_v14/'
image_filename = os.path.join(output_directory, 'train_val_loss.png')
plt.savefig(image_filename, dpi=1000, bbox_inches='tight')  # 设置 DPI 为 1000，提高图片清晰度，并移除留白
plt.show()

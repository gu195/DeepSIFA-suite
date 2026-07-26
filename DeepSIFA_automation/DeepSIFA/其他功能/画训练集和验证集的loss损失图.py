import matplotlib.pyplot as plt
import os

# defines the function to read the loss value
def read_loss_from_file(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

# Specifies the loss value file path
train_loss_path = '/home/node01/实验数据/Vit+PRM/logs/fold4_v14/global_train_loss.txt'
val_loss_path = '/home/node01/实验数据/Vit+PRM/logs/fold4_v14/global_val_loss.txt'

# Read loss value from file
global_train_loss = read_loss_from_file(train_loss_path)
global_val_loss = read_loss_from_file(val_loss_path)

# Draw the loss curve
plt.figure(figsize=(5, 4))
# Make sure the x-axis value matches the number of data points
epochs = max(len(global_train_loss), len(global_val_loss))
x = range(epochs)

# Draw the training loss curve, using a thick red solid line
plt.plot(x[:len(global_train_loss)], global_train_loss, label='training loss', color='red', linewidth=3)
# Draw the verification loss curve, using a thick blue solid line
plt.plot(x[:len(global_val_loss)], global_val_loss, label='validation loss', color='#87CEFA', linewidth=3)

# Set the parameters of the scale and label of the coordinate axis
plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=4)
# plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=4)

# Set the coordinate axis to be thicker
ax = plt.gca()  # Get the current axis
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
# Let the y-axis start displaying from 0
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
# Save chart
output_directory = '/home/node01/实验数据/Vit+PRM/logs/fold4_v14/'
image_filename = os.path.join(output_directory, 'train_val_loss.png')
plt.savefig(image_filename, dpi=1000, bbox_inches='tight')  # Set DPI to 1000 to improve picture clarity and remove white space
plt.show()

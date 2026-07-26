import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils.MYCAM import GradCAM, show_cam_on_image, center_crop_img
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_cfg(fold):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default='512', type=int)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--lr_seg', type=float, default=1e-4)  # 0.0003
    parser.add_argument('--n_epochs', type=int, default=1)  
    parser.add_argument('--bt_size', type=int, default=1)  # 36

    # github path
    parser.add_argument('--weight_path', type=str, default='./checkpoints/fold{}/best_acc.pth'.format(fold))
    parser.add_argument('--data_dir', type=str, default='/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3_数据增强/归一化插值后npz_1024_高斯平滑1/')
    parser.add_argument('--val_csv_dir', type=str, default='/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3_数据增强/1177_5折交叉验证_42/val_fold{}_多标签.csv'.format(fold))
    parser.add_argument('--directory', type=str, default='./logs_val验证_0.8/fold{}/'.format(fold))
    # parser.add_argument('--val_csv_dir', type=str, default='/data2/linchen_data/linchen/DeepFRET-Model-master/data/F_F0/v3/628_5-fold cross-validation_42/val_fold{}_multi-label.csv'.format(fold))

    parse_config = parser.parse_args()
    return parse_config



if __name__ == '__main__':
    for fold in range(1,2):
        print('第{}折:'.format(fold))
        # -------------------------- get args --------------------------#
        torch.cuda.empty_cache()
        parse_config = get_cfg(fold)

        from models.vit import vit_base_patch16_224
        model = vit_base_patch16_224(num_classes=2)
        print(model)
        pretrained = True
        if pretrained:#How to modify this paragraph
            model_dict = model.state_dict()
            model_weights = torch.load(parse_config.weight_path)
            pretrained_dict = model_weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#Remove the overlapping parts of the pre-trained model and the dict of the new model
            model_dict.update(pretrained_dict)#Update some parameters in new_model with pre-trained model parameters
            model.load_state_dict(model_dict) #Load the updated model_dict into the new model

        target_layers = [model.blocks[-1].norm1]
        npz_directory = parse_config.data_dir
        directory = parse_config.directory
        print(directory)
        # Get the names of all folders in the directory
        subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
        # Print the names of all folders
        for subdir in subdirectories:
            subdir_path = os.path.join(directory, subdir)
            print(subdir_path)
            png_files = glob.glob(os.path.join(subdir_path, '*.png'))
            for png_path in png_files:
                npz_filename = os.path.basename(png_path).replace('.png', '.npz')
                npz_path = npz_directory + npz_filename
                assert os.path.exists(npz_path), "file: '{}' dose not exist.".format(npz_path)
                npz_data = np.load(npz_path)

                # ------------------------It is not img and needs to be modified--------------------------------------------------------
                data = npz_data['data']
                data = data.reshape(1, -1)  # 1，1024
                data_tensor = torch.from_numpy(data.astype(np.float32)) # 1，1024
                # expand batch dimension
                # [C, W] -> [N, C, W]
                input_tensor = torch.unsqueeze(data_tensor, dim=0)
                # --------------------------------------------------------------


                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False) # initialization
                grayscale_cam = cam(input_tensor=input_tensor) # cam's __call__ method requires 2 inputs. There is no need to pass in target_category. It will automatically calculate and generate a heat map of the predicted value category!! !
                grayscale_cam = grayscale_cam[0, :] # 1 1024
                # np.savetxt('grayscale_cam.txt', grayscale_cam[0], fmt='%f', newline='\n')


                # --------------------------Visualization part-----------------
                # Create a custom color map
                from matplotlib.colors import LinearSegmentedColormap
                import matplotlib.cm as cm
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                norm = plt.Normalize(vmin=0, vmax=1)  # maps the gray value range to between 0-1
                norm_ = norm(grayscale_cam[0])
                norm_[norm_ < 0.1] = 0
                min_value = np.min(norm_)
                max_value = np.max(norm_)
                norm_ = 0.5 + 0.5 * (norm_ - min_value) / (max_value - min_value)
                norm_[norm_ < 0.6] = 0
                # plt.figure(figsize=(8, 6)) # Create a new graphic object and set the size to 8 inches wide and 6 inches high
                # plt.hist(norm_.flatten(), bins=100, range=(0.1, 1), color='blue', alpha=0.7)
                # plt.xlabel('Normalized Values')
                # plt.ylabel('Frequency')
                # plt.title('Distribution of Normalized Values')
                # plt.grid(True)
                # plt.savefig('./tiaos1.png')


                # Create a custom color map
                colors_blue_to_red = [(0, 0, 1), (1, 0, 0)]  # Blue to red transition
                cm = LinearSegmentedColormap.from_list('blue_to_red', colors_blue_to_red, N=256)
                colors = cm(norm_)
                # -----------------------------------------------------------------------------


                # Visual scatter points
                # Create x-axis coordinate
                data = npz_data['data'] # 1024,
                x = range(len(data))
                plt.figure(figsize=(24, 6))  # Set the size of the graphic to 10 inches wide and 5 inches high
                # Draw curves and scatter plots
                plt.plot(x, data, label='Data Line', linewidth=0.5)  # sets line width to 1

                # # -----------Set the threshold and set the background color, assuming it is white----------
                threshold = 0.2
                background_color = [1, 1, 1, 0]
                filtered_colors = [color if grayscale_cam[0][i] > threshold else background_color for i, color in enumerate(colors)]
                # ---------------------
                plt.scatter(x, data, c=filtered_colors, s=40, alpha=0.9, edgecolor='none') 


                for i, _ in enumerate(data):
                    # Add text labels only on points where the grayscale value is close to 1
                    if grayscale_cam[0][i] > threshold:
                        plt.text(x[i], data[i], str(grayscale_cam[0][i]), fontsize=1, ha='center', va='bottom')  
                # ------------------------------------------
                
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title('Data_{}'.format(npz_filename))
                plt.legend()  # Display legend

                # Add color bar next to image
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cb = plt.colorbar(cax=cax)
                cb.set_label('Color')

                save_path = subdir_path + '/热图threshold{}/'.format(threshold) 
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, npz_filename.replace('.npz','.png')), dpi=800)
                plt.clf()       # Clear the current graph so that the next file can draw a new graph
                plt.close()     # Close previous graph before drawing new graph


                # ##########################################################################
                # # Visualized heat map value size
                # # Create x-axis coordinate
                # data = grayscale_cam[0]
                # x = range(len(data))
                # plt.figure(figsize=(24, 6)) # Set the size of the figure to 10 inches wide and 5 inches high
                
                # # Draw curves and scatter plots
                # plt.plot(x, data, label='Data Line', linewidth=0.5) # Set the line width to 1
                # plt.scatter(x, data, label='Data Points', s=0.5) # Draw scatter points
                
                # plt.xlabel('Index')
                # plt.ylabel('Value')
                # plt.title('ReTu')
                # plt.legend() # Display legend

                # save_path = './heatmap/heatmap{}.png'.format(fold)
                # plt.savefig(save_path)
                # plt.clf() # Clear the current graph so that the next file can draw a new graph
                # ##########################################################################


                # ##########################################################################
                # # Visualized heat map color
                # # Create x-axis coordinate
                # values = grayscale_cam[0]
                # # Create a color map, 0 corresponds to blue, 1 corresponds to red
                # cmap = plt.cm.RdBu # Use red-blue color mapping, the corresponding value of red is 1, and the corresponding value of blue is 0
                # norm = plt.Normalize(vmin=0, vmax=1) # Map the value range to between 0-1
                # colors = cmap(norm(values)) # Map the values of the array to colors 1024, 4

                # # Draw color bar icon
                # plt.figure(figsize=(8, 2))
                # plt.imshow([values], cmap=cmap, aspect='auto')
                # plt.colorbar(label='Values')
                # plt.title('Color Map')
                # save_path = './heatmap/color{}.png'.format(fold)
                # plt.savefig(save_path)
                # plt.clf() # Clear the current graph so that the next file can draw a new graph
                # ##########################################################################

                # # Based on the above visual scatter points, attach the heat map information of each point, that is, the size of each position point, and change it to color.
                # visualization = show_cam_on_image(data.astype(dtype=np.float32) / 255.,# attention
                #                                 grayscale_cam,
                #                                 use_rgb=True)
                # plt.imshow(visualization)
                # plt.savefig('./heatmap/png{}.png'.format(fold))
                # --------------------------------------------------------------

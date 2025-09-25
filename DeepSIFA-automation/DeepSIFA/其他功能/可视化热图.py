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

    # github上路径
    parser.add_argument('--weight_path', type=str, default='./checkpoints/fold{}/best_acc.pth'.format(fold))
    parser.add_argument('--data_dir', type=str, default='/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3_数据增强/归一化插值后npz_1024_高斯平滑1/')
    parser.add_argument('--val_csv_dir', type=str, default='/home/node01/linchen/data/alphaK10/V2/汇总/1+2+3_数据增强/1177_5折交叉验证_42/val_fold{}_多标签.csv'.format(fold))
    parser.add_argument('--directory', type=str, default='./logs_val验证_0.8/fold{}/'.format(fold))
    # parser.add_argument('--val_csv_dir', type=str, default='/data2/linchen_data/linchen/DeepFRET-Model-master/data/F_F0/v3/628_5折交叉验证_42/val_fold{}_多标签.csv'.format(fold))

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
        if pretrained:#这一段要如何修改
            model_dict = model.state_dict()
            model_weights = torch.load(parse_config.weight_path)
            pretrained_dict = model_weights
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#取出预训练模型中与新模型的dict中重合的部分
            model_dict.update(pretrained_dict)#用预训练模型参数更新new_model中的部分参数
            model.load_state_dict(model_dict) #将更新后的model_dict加载进new model中 

        target_layers = [model.blocks[-1].norm1]
        npz_directory = parse_config.data_dir
        directory = parse_config.directory
        print(directory)
        # 获取目录下所有文件夹的名称
        subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
        # 打印所有文件夹的名称
        for subdir in subdirectories:
            subdir_path = os.path.join(directory, subdir)
            print(subdir_path)
            png_files = glob.glob(os.path.join(subdir_path, '*.png'))
            for png_path in png_files:
                npz_filename = os.path.basename(png_path).replace('.png', '.npz')
                npz_path = npz_directory + npz_filename
                assert os.path.exists(npz_path), "file: '{}' dose not exist.".format(npz_path)
                npz_data = np.load(npz_path)

                # ---------------------都不是img，需要修改--------------------------------------------------------
                data = npz_data['data']
                data = data.reshape(1, -1)  # 1，1024
                data_tensor = torch.from_numpy(data.astype(np.float32)) # 1，1024
                # expand batch dimension
                # [C, W] -> [N, C, W]
                input_tensor = torch.unsqueeze(data_tensor, dim=0)
                # --------------------------------------------------------------


                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False) # 初始化
                grayscale_cam = cam(input_tensor=input_tensor) # cam的__call__方法需要2个输入，不用传入target_category，自动计算，生成预测值那一类的热图!！！
                grayscale_cam = grayscale_cam[0, :] # 1 1024
                # np.savetxt('grayscale_cam.txt', grayscale_cam[0], fmt='%f', newline='\n')


                # ---------------------可视化部分-----------------
                # 创建自定义的颜色映射
                from matplotlib.colors import LinearSegmentedColormap
                import matplotlib.cm as cm
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                norm = plt.Normalize(vmin=0, vmax=1)  # 将灰度值范围映射到0-1之间
                norm_ = norm(grayscale_cam[0])
                norm_[norm_ < 0.1] = 0
                min_value = np.min(norm_)
                max_value = np.max(norm_)
                norm_ = 0.5 + 0.5 * (norm_ - min_value) / (max_value - min_value)
                norm_[norm_ < 0.6] = 0
                # plt.figure(figsize=(8, 6))  # 创建一个新的图形对象，设置大小为宽8英寸，高6英寸
                # plt.hist(norm_.flatten(), bins=100, range=(0.1, 1), color='blue', alpha=0.7)
                # plt.xlabel('Normalized Values')
                # plt.ylabel('Frequency')
                # plt.title('Distribution of Normalized Values')
                # plt.grid(True)
                # plt.savefig('./tiaos1.png')


                # 创建自定义的颜色映射
                colors_blue_to_red = [(0, 0, 1), (1, 0, 0)]  # 蓝色到红色的过渡
                cm = LinearSegmentedColormap.from_list('blue_to_red', colors_blue_to_red, N=256)
                colors = cm(norm_)
                # -----------------------------------------------------------------------------


                # 可视化散点
                # 创建x轴坐标
                data = npz_data['data'] # 1024,
                x = range(len(data))
                plt.figure(figsize=(24, 6))  # 设置图形的大小为宽10英寸，高5英寸
                # 绘制曲线和散点图
                plt.plot(x, data, label='Data Line', linewidth=0.5)  # 将线条宽度设置为1

                # # -----------设定阈值 设置背景颜色，假设为白色----------
                threshold = 0.2
                background_color = [1, 1, 1, 0]
                filtered_colors = [color if grayscale_cam[0][i] > threshold else background_color for i, color in enumerate(colors)]
                # ---------------------
                plt.scatter(x, data, c=filtered_colors, s=40, alpha=0.9, edgecolor='none') 


                for i, _ in enumerate(data):
                    # 仅在灰度值接近1的点上添加文本标签
                    if grayscale_cam[0][i] > threshold:
                        plt.text(x[i], data[i], str(grayscale_cam[0][i]), fontsize=1, ha='center', va='bottom')  
                # ------------------------------------------
                
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title('Data_{}'.format(npz_filename))
                plt.legend()  # 显示图例

                # 在图像旁边添加颜色条
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cb = plt.colorbar(cax=cax)
                cb.set_label('Color')

                save_path = subdir_path + '/热图threshold{}/'.format(threshold) 
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, npz_filename.replace('.npz','.png')), dpi=800)
                plt.clf()       # 清除当前图形，以便下一个文件可以绘制新图
                plt.close()     # 在绘制新图之前关闭之前的图形


                # ##########################################################################
                # # 可视化热图值大小
                # # 创建x轴坐标
                # data = grayscale_cam[0]
                # x = range(len(data))
                # plt.figure(figsize=(24, 6))  # 设置图形的大小为宽10英寸，高5英寸
                
                # # 绘制曲线和散点图
                # plt.plot(x, data, label='Data Line', linewidth=0.5)  # 将线条宽度设置为1
                # plt.scatter(x, data, label='Data Points', s=0.5)  # 绘制散点
                
                # plt.xlabel('Index')
                # plt.ylabel('Value')
                # plt.title('ReTu')
                # plt.legend()  # 显示图例

                # save_path = './热图/热图{}.png'.format(fold)
                # plt.savefig(save_path)
                # plt.clf()  # 清除当前图形，以便下一个文件可以绘制新图
                # ##########################################################################


                # ##########################################################################
                # # 可视化热图颜色
                # # 创建x轴坐标
                # values = grayscale_cam[0]
                # # 创建颜色映射，0 对应蓝色，1 对应红色
                # cmap = plt.cm.RdBu  # 使用红-蓝颜色映射，红色对应值为 1，蓝色对应值为 0
                # norm = plt.Normalize(vmin=0, vmax=1)  # 将值范围映射到 0-1 之间
                # colors = cmap(norm(values))  # 将数组的值映射到颜色 1024, 4

                # # 绘制色彩条图示
                # plt.figure(figsize=(8, 2))
                # plt.imshow([values], cmap=cmap, aspect='auto')
                # plt.colorbar(label='Values')
                # plt.title('Color Map')
                # save_path = './热图/color{}.png'.format(fold)
                # plt.savefig(save_path)
                # plt.clf()  # 清除当前图形，以便下一个文件可以绘制新图
                # ##########################################################################

                # # 要在上面这个可视化散点的基础上，附带上每一个点的热图信息，也就是每一个位置点附带的大小，变为颜色即可
                # visualization = show_cam_on_image(data.astype(dtype=np.float32) / 255.,# attention
                #                                 grayscale_cam,
                #                                 use_rgb=True)
                # plt.imshow(visualization)
                # plt.savefig('./热图/png{}.png'.format(fold))
                # --------------------------------------------------------------

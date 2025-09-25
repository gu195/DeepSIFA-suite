import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    print(np.sum(intersection))
    print(np.sum(image1))
    iou = np.sum(intersection) / np.sum(image1)
    return iou

# 设置两个目录的路径
# directory1 = "/data2/linchen_data/row_data/xmu1/85张dcm/jpg_output"
# directory2 = "/data2/linchen_data/jing/bert_v16五折/logs/fold5/val_images/pre"
# directory1 = "/data2/linchen_data/row_data/xmu1/test_2023-10-20jpg_output"
# directory2 = "/data2/linchen_data/jing/bert_v16/logs_xmu1_test验证q20/fold5/val_images/pre"
directory1 = "/data2/linchen_data/row_data/jl2/精细化113张/jpg_output"
directory2 = "/data2/linchen_data/jing/bert_v16/log_jlu2外部验证_q25/fold5/val_images/pre"

iou_values = []

# 获取directory1中的所有文件
for filename1 in os.listdir(directory1):
    if filename1.endswith(".jpg"):
        # 构建图像的完整路径
        path1 = os.path.join(directory1, filename1)
        
        # 构建相应的mask文件路径，假设mask文件与图像文件名相同，只是位于directory2目录下
        filename2 = filename1.replace("_fracture_jpg_Label.nii.jpg", ".jpg") #适合xmu1val和jl2
        # filename2 = filename1.split('-')[0] + ".jpg" # 适合xmu1第二批50张
        path2 = os.path.join(directory2, filename2)
        print(path2)
        
        # 加载图像
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.resize(image1, (512, 512))
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # 二值化处理
        _, image1 = cv2.threshold(image1, 128, 1, cv2.THRESH_BINARY)
        _, image2 = cv2.threshold(image2, 128, 1, cv2.THRESH_BINARY)

        # # 保存二值化后的图像
        # cv2.imwrite('image1.jpg', image1)
        # cv2.imwrite('image2.jpg', image2)
        
        # 计算IoU值
        iou = compute_iou(image1, image2)
        
        # 将IoU值添加到列表中
        iou_values.append(iou)

# 计算平均IoU值
average_iou = np.mean(iou_values)
print("Average IoU:", average_iou)

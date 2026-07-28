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

# Set the paths of two directories
# directory1 = "/data2/linchen_data/row_data/xmu1/85dcm/jpg_output"
# directory2 = "/data2/linchen_data/jing/bert_v1650% off/logs/fold5/val_images/pre"
# directory1 = "/data2/linchen_data/row_data/xmu1/test_2023-10-20jpg_output"
# directory2 = "/data2/linchen_data/jing/bert_v16/logs_xmu1_testverificationq20/fold5/val_images/pre"
directory1 = "/data2/linchen_data/row_data/jl2/精细化113张/jpg_output"
directory2 = "/data2/linchen_data/jing/bert_v16/log_jlu2外部验证_q25/fold5/val_images/pre"

iou_values = []

# Get all files in directory1
for filename1 in os.listdir(directory1):
    if filename1.endswith(".jpg"):
        # Full path to build image
        path1 = os.path.join(directory1, filename1)
        
        # Construct the corresponding mask file path, assuming that the mask file has the same name as the image file, but is located in the directory2 directory
        filename2 = filename1.replace("_fracture_jpg_Label.nii.jpg", ".jpg") #suitable for xmu1val and jl2
        # filename2 = filename1.split('-')[0] + ".jpg" # Suitable for the second batch of 50 photos of xmu1
        path2 = os.path.join(directory2, filename2)
        print(path2)
        
        # Load image
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.resize(image1, (512, 512))
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # Binary processing
        _, image1 = cv2.threshold(image1, 128, 1, cv2.THRESH_BINARY)
        _, image2 = cv2.threshold(image2, 128, 1, cv2.THRESH_BINARY)

        # # Save the binarized image
        # cv2.imwrite('image1.jpg', image1)
        # cv2.imwrite('image2.jpg', image2)
        
        # Calculate IoU value
        iou = compute_iou(image1, image2)
        
        # Add IoU value to list
        iou_values.append(iou)

# Calculate the average IoU value
average_iou = np.mean(iou_values)
print("Average IoU:", average_iou)

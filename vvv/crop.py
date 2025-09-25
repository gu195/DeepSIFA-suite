from PIL import Image
import tifffile

# 打开 TIFF 图像
tif_path = "D:\DeepSIFA_main\data\wt0117测试\I\S55C_label_alex55_10nM_1.tif"
img = tifffile.imread(tif_path)

# 获取指定区域的图像（注意：Python中的索引是从0开始）
x_start, x_end = 100, 140  # x坐标的范围
y_start, y_end = 190, 230  # y坐标的范围

# 图像裁剪
cropped_img = img[:, y_start:y_end, x_start:x_end]  # 因为tif图像是 (frames, height, width) 

# 保存裁剪后的图像
output_path = "D:\DeepSIFA_main\data\wt0117测试\I\S55C_label_alex55_10nM_1_cropped_image.tif"
tifffile.imwrite(output_path, cropped_img)

print(f"Cropped image saved to {output_path}")
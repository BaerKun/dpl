import cv2
import os

# 定义输入和输出文件夹路径
input_folder = './img'  # 输入文件夹路径
output_folder = './grey_img'  # 输出文件夹路径

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 只处理图像文件（可以根据需要调整文件扩展名）
    if filename.endswith('.png'):
        # 读取图像
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # 检查图像是否成功读取
        if image is not None:
            # 转换为灰度图
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 保存灰度图到输出文件夹
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, gray_image)

            print(f"Processed {filename}")

print("所有图像已成功转换为灰度图。")
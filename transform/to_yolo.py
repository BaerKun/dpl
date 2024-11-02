import os
import cv2

# 定义路径
images_folder = './grey_img'  # 图像文件夹路径
labels_folder = './txt'  # 标签文件夹路径
output_labels_folder = './yolo'  # 输出 YOLO 标签文件夹路径

# 创建输出文件夹
os.makedirs(output_labels_folder, exist_ok=True)

# 遍历图像文件夹中的所有图像
for img_filename in os.listdir(images_folder):
    if img_filename.endswith('.png'):
        # 读取图像
        img_path = os.path.join(images_folder, img_filename)
        image = cv2.imread(img_path)
        img_height, img_width, _ = image.shape

        # 对应的标签文件
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_folder, label_filename)

        if os.path.exists(label_path):
            # 创建 YOLO 格式的标签文件
            yolo_label_path = os.path.join(output_labels_folder, label_filename)

            with open(label_path, 'r') as f, open(yolo_label_path, 'w') as yolo_file:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(' ')
                    label = int(parts[0])  # 标签
                    coords = [0.0] * 10
                    min_x = min_y = 1.0
                    max_x = max_y = 0.0

                    for i in range(1, 11, 2):
                        x = float(parts[i]) / img_width
                        y = float(parts[i + 1]) / img_height

                        if x < min_x:
                            min_x = x
                        if x > max_x:
                            max_x = x
                        if y < min_y:
                            min_y = y
                        if y > max_y:
                            max_y = y

                        coords[i - 1] = x
                        coords[i] = y

                    box_center_x = (min_x + max_x) / 2
                    box_center_y = (min_y + max_y) / 2
                    box_width = max_x - min_x
                    box_height = max_y - min_y

                    # 写入 YOLO 格式
                    yolo_file.write(f"{label} {box_center_x} {box_center_y} {box_width} {box_height} "
                                    f"{' '.join([str(coord) for coord in coords[:]])}\n")

print("已成功将图像和关键点标签转换为 YOLO 格式。")

import cv2
from preprocess import process_image


def preprocess(img_path):
    # 读取图像
    image = cv2.imread(img_path)
    # 检查图像是否成功读取
    if image is not None:
        # 转换为灰度图
        processed_image, matrix = process_image(image)

        # 保存灰度图到输出文件夹
        output_path = "prepro-test.png"
        cv2.imwrite(output_path, processed_image)
        print("done")
    else:
        print("Error: Failed to read image.")

if __name__ == "__main__":
    path = input("Enter image path: ")
    preprocess(path)

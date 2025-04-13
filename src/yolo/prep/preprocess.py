import cv2
import os
import shutil
import random
from datetime import datetime
import utils.preprocess as prep
from utils.basedata import KeyPoints

# LEFT, UPPER, RIGHT, R_CENTER, LOWER = 0, 1, 2, 3, 4
# NUM_CLASSES = 3
INPUT_SIZE = 416


class YoloPoseLabel:
    def __init__(self, cls, key_points):
        self.cls = cls
        self.key_points = KeyPoints(key_points)


def yolo_load_label(label_path):
    labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            cls = int(parts[0])
            key_points = [float(_) for _ in parts[1:]]
            labels.append(YoloPoseLabel(cls, key_points))

    return labels


def yolo_save_label(labels: list[YoloPoseLabel], save_path: str):
    with open(save_path, 'w') as f:
        for label in labels:
            box = label.key_points.bounding_box()

            box_width = box[2]
            box_height = box[3]
            box_center_x = box[0] + box_width / 2
            box_center_y = box[1] + box_height / 2

            f.write(f"{label.cls} {box_center_x} {box_center_y} {box_width} {box_height} "
                    f"{' '.join([str(coord) for coord in label.key_points.np.flatten()])}\n")


def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = prep.add_noise(gray_image, (-32, 4)) # 防止对背景过拟合 !!!

    cut_image, cut_matrix = prep.cut(gray_image, (0, 50), (0, 50), (0, 100), (0, 100))
    resized_image, affine_matrix = prep.letterbox(cut_image, (INPUT_SIZE, INPUT_SIZE))
    rotated_image, rotate_matrix = prep.affine(resized_image, rotate=(0., 360.))

    m = rotate_matrix @ affine_matrix @ cut_matrix

    return rotated_image, m


def process_and_save(image, labels, image_save_path, label_save_path):
    processed_image, matrix = process_image(image)
    cv2.imwrite(image_save_path, processed_image)

    clone_labels = []
    for label in labels:
        kps = label.key_points.transform(matrix, scale=1. / INPUT_SIZE)
        clone_labels.append(YoloPoseLabel(label.cls, kps))

    yolo_save_label(clone_labels, label_save_path)


def resample(image, labels):
    repeat = [1, 3, 2, 2, 1, 1, 1]
    multiple = 1
    return [(image, labels)] * (repeat[len(labels)] * multiple)


def preprocess(mode: str = "train", max_count: int = None, *, use_backup: bool = True, move_backup: bool = True):
    if use_backup:
        images_input_folder = os.path.join('backup/images', mode)
        labels_input_folder = os.path.join('backup/labels', mode)
        move_backup = False
    else:
        images_input_folder = 'img'
        labels_input_folder = 'txt'

    images_list = os.listdir(images_input_folder)
    if max_count is not None:
        random.shuffle(images_list)

    images_output_folder = os.path.join('../datasets/images', mode)  # 输出文件夹路径
    labels_output_folder = os.path.join('../datasets/labels', mode)
    images_backup_folder = os.path.join('backup/images', mode)
    labels_backup_folder = os.path.join('backup/labels', mode)

    if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)
    if not os.path.exists(labels_output_folder):
        os.makedirs(labels_output_folder)

    if move_backup:
        if not os.path.exists(images_backup_folder):
            os.makedirs(images_backup_folder)
        if not os.path.exists(labels_backup_folder):
            os.makedirs(labels_backup_folder)

    counter = 0
    for image_filename in images_list:
        if not image_filename.endswith('.png'):
            continue

        img_path = os.path.join(images_input_folder, image_filename)
        label_filename = image_filename.replace('.png', '.txt')
        label_path = os.path.join(labels_input_folder, label_filename)

        image = cv2.imread(img_path)
        if image is None:
            continue
        labels = yolo_load_label(label_path)

        for _image, _labels in resample(image, labels):
            str_time = datetime.now().strftime("%H%M%S%f")
            images_output_path = os.path.join(images_output_folder, str_time + '.png')
            label_output_path = os.path.join(labels_output_folder, str_time + '.txt')

            process_and_save(_image, _labels, images_output_path, label_output_path)
            counter += 1

            print(f"\rprocessed: {counter}", end="")
            if max_count is not None and counter == max_count:
                break

        if move_backup:
            # 移动文件到 backup 文件夹
            shutil.move(label_path, os.path.join(labels_backup_folder, label_filename))
            shutil.move(img_path, os.path.join(images_backup_folder, image_filename))

        if max_count is not None and counter == max_count:
            break

    print(f"\ndone.\n{counter} images/labels processed in total.")


def preprocess_test(image_folder):
    output_folder = "../datasets/images/test"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)

        image = cv2.imread(image_path)
        if image is None:
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image //= 4
        processed_image = prep.add_noise(gray_image, (-8, 8))

        cv2.imwrite(os.path.join(output_folder, image_filename), processed_image)


if __name__ == '__main__':
    _mode = input("images to train[0] / val[1] / test[2]? (default: train) ")
    if not _mode:
        _mode = "train"
    elif _mode.isdigit():
        if int(_mode) == 0:
            _mode = "train"
        elif int(_mode) == 1:
            _mode = "val"
        elif int(_mode) == 2:
            test_folder = input("test folder? (default: backup/test) ")
            if not test_folder:
                test_folder = "backup/test"
            preprocess_test(test_folder)
            exit()
        else:
            raise ValueError("mode must be 0 or 1")
    else:
        raise ValueError("mode must be 0 or 1")

    use_backup_ = input("use backup? false[0] or true[1] (default: true) ")
    if not use_backup_:
        use_backup_ = True
    elif use_backup_.isdigit():
        use_backup_ = int(use_backup_) != 0
    else:
        raise ValueError("use_backup must be int")

    if use_backup_:
        move_backup_ = False
    else:
        move_backup_ = input("move backup? false[0] or true[1] (default: true) ")
        if not move_backup_:
            move_backup_ = True
        elif move_backup_.isdigit():
            move_backup_ = int(move_backup_) != 0
        else:
            raise ValueError("move_backup must be int")

    _max_count = input("max count? (default: None) ")
    if not _max_count:
        _max_count = None
    elif _max_count.isdigit():
        _max_count = int(_max_count)
    else:
        raise ValueError("max_count must be int")

    preprocess(_mode, _max_count, use_backup=use_backup_, move_backup=move_backup_)

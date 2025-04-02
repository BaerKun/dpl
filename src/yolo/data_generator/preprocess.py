import numpy as np
import cv2
import os
import shutil
import random
from datetime import datetime

LEFT, UPPER, RIGHT, R_CENTER, LOWER = 0, 1, 2, 3, 4
NUM_CLASSES = 3
INPUT_SIZE = 416
ASPECT_RATIO = 1440 / 1080

class Label:
    def __init__(self, cls:int, key_points:np.ndarray):
        self.cls = cls
        self.key_points = key_points.reshape(-1, 2)

    def transform(self, matrix:np.ndarray, normalize:bool = True):
        if matrix is None:
            return self.key_points.copy()

        _coords = self.key_points.T
        w = matrix[:2, :2]
        b = matrix[:2, 2:]
        _coords = w @ _coords + b
        if normalize:
            _coords /= INPUT_SIZE
        return _coords.T


def bounding_box(coords: np.ndarray) -> tuple:
    x = coords[:, 0].min()
    y = coords[:, 1].min()
    width = coords[:, 0].max() - x
    height = coords[:, 1].max() - y
    return x, y, width, height


def box_union(box, r_box):
    alpha = 0.5
    r_tl_x = r_box[0] + r_box[2] * (1 - alpha) / 2
    r_tl_y = r_box[1] + r_box[3] * (1 - alpha) / 2
    r_br_x = r_box[0] + r_box[2] * (1 + alpha) / 2
    r_br_y = r_box[1] + r_box[3] * (1 + alpha) / 2
    x = min(box[0], r_tl_x)
    y = min(box[1], r_tl_y)
    w = max(box[0] + box[2], r_br_x) - x
    h = max(box[1] + box[3], r_br_y) - y
    return x, y, w, h


def cut(image):
    w, h = image.shape[1], image.shape[0]
    if w / h >= ASPECT_RATIO:
        cut_x = (w - int(ASPECT_RATIO * h)) // 2
        cut_y = 0
    else:
        cut_x = 0
        cut_y = (h - int(w / ASPECT_RATIO)) // 2

    # 超出范围的数据,ultralytics会忽略
    left = cut_x + random.randint(-100, 100)
    if left < 0:
        left = 0
    right = cut_x + random.randint(-100, 100)
    if right >= w:
        right = w - 1
    top = cut_y + random.randint(-50, 50)
    if top < 0:
        top = 0
    bottom = cut_y + random.randint(-50, 50)
    if bottom >= h:
        bottom = h - 1

    m = np.array(((1., 0., -left),
                  (0., 1., -top),
                  (0., 0., 1.)), dtype=np.float32)
    return image[top: h - bottom, left: w - right], m


def letterbox(src):
    width, height = src.shape[1], src.shape[0]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_width, new_height = int(width * scale), int(height * scale)

    if new_width % 2 == 1:
        new_width += 1
    if new_height % 2 == 1:
        new_height += 1

    padding_x = (INPUT_SIZE - new_width) // 2
    padding_y = (INPUT_SIZE - new_height) // 2
    dst = cv2.resize(src, (new_width, new_height))
    dst = cv2.copyMakeBorder(dst, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

    m = np.array(((scale, 0, padding_x),
                  (0, scale, padding_y),
                  (0, 0, 1.)), dtype=np.float32)
    return dst, m


def rotate(src):
    h, w = src.shape[0], src.shape[1]
    angle = random.uniform(0., 360.)
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.)  # (2 x 3)
    dst = cv2.warpAffine(src, m, (w, h))
    m = np.vstack((m, (0, 0, 1)))
    return dst, m


def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    random_noise = np.array(random.randint(-32, 4), dtype=np.float64) # cv2.add规定必须是float64
    gray_image = cv2.add(gray_image, random_noise) # 防止对背景过拟合 !!!

    cut_image, cut_matrix = cut(gray_image)
    resized_image, affine_matrix = letterbox(cut_image)
    rotated_image, rotate_matrix = rotate(resized_image)

    m = rotate_matrix @ affine_matrix @ cut_matrix

    return rotated_image, m


def load_label(label_path):
    _labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            cls = int(parts[0])
            coords = np.array([float(_) for _ in parts[1:]], dtype=np.float32)
            _labels.append(Label(cls, coords))

    return _labels


def process_label2(labels : list[Label], matrix=None):
    str_labels = ""

    r = labels[-1]
    key_points = r.transform(matrix)
    r_box = bounding_box(key_points)
    r_center = r_box[0] + r_box[2] / 2, r_box[1] + r_box[3] / 2

    for _label in labels:
        if _label.cls == 0:
            break

        key_points = _label.transform(matrix)
        box = bounding_box(key_points)
        box = box_union(box, r_box)
        key_points[R_CENTER] = r_center

        box_width = box[2]
        box_height = box[3]
        box_center_x = box[0] + box_width / 2
        box_center_y = box[1] + box_height / 2

        str_labels += (
            f"{_label.cls - 1} {box_center_x} {box_center_y} {box_width} {box_height} "
            f"{' '.join([str(coord) for coord in key_points.flatten()])}\n")

    return str_labels


def process_label3(labels : list[Label], matrix=None):
    str_labels = ""

    for _label in labels:
        key_points = _label.transform(matrix)
        box = bounding_box(key_points)

        box_width = box[2]
        box_height = box[3]
        box_center_x = box[0] + box_width / 2
        box_center_y = box[1] + box_height / 2

        str_labels += (
            f"{_label.cls} {box_center_x} {box_center_y} {box_width} {box_height} "
            f"{' '.join([str(coord) for coord in key_points.flatten()])}\n")

    return str_labels


def process_label(labels : list[Label], matrix=None, num_classes:int=NUM_CLASSES):
    if num_classes == 3:
        return process_label3(labels, matrix)
    return process_label2(labels, matrix)


def process_and_save(image, labels, image_save_path, label_save_path):
    processed_image, matrix = process_image(image)
    str_labels = process_label(labels, matrix)
    cv2.imwrite(image_save_path, processed_image)

    with open(label_save_path, 'w') as label_output_file:
        for _label in str_labels:
            label_output_file.write(_label)


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
        labels = load_label(label_path)

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

        noise = np.array(random.randint(-8, 8), dtype=np.float64)
        image //= 4
        processed_image = cv2.add(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), noise)

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

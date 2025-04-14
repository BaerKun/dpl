import cv2
import os
import random
from datetime import datetime
from dpl.basedata import KeyPoints


class YoloPoseLabel:
    def __init__(self, cls, key_points):
        self.cls = cls
        self.key_points = KeyPoints(key_points)


def load_label(label_path):
    labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            cls = int(parts[0])
            key_points = [float(_) for _ in parts[1:]]
            labels.append(YoloPoseLabel(cls, key_points))

    return labels


def save_label(labels: list[YoloPoseLabel], save_path: str):
    with open(save_path, 'w') as f:
        for label in labels:
            box = label.key_points.bounding_box()

            box_width = box[2]
            box_height = box[3]
            box_center_x = box[0] + box_width / 2
            box_center_y = box[1] + box_height / 2

            f.write(f"{label.cls} {box_center_x} {box_center_y} {box_width} {box_height} "
                    f"{' '.join([str(coord) for coord in label.key_points.np.flatten()])}\n")


def preprocess_dir(images_dir, labels_dir, datasets_dir, prep_image, mode:str, max_count: int = None):
    if mode not in ['train', 'val']:
        raise ValueError("mode must be one of 'train', 'val'")

    if not os.path.exists(os.path.join(datasets_dir, 'images')):
        os.makedirs(os.path.join(datasets_dir, 'images'))
    if not os.path.exists(os.path.join(datasets_dir, 'labels')):
        os.makedirs(os.path.join(datasets_dir, 'labels'))

    output_images_dir = os.path.join(datasets_dir, 'images', mode)
    output_labels_dir = os.path.join(datasets_dir, 'labels', mode)

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    images_list = os.listdir(images_dir)
    if max_count is not None:
        random.shuffle(images_list)

    counter = 0
    for image_filename in images_list:
        img_path = os.path.join(images_dir, image_filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        labels = load_label(label_path)

        str_time = datetime.now().strftime("%H%M%S%f")
        image_save_path = os.path.join(output_images_dir, str_time + '.png')
        label_save_path = os.path.join(output_labels_dir, str_time + '.txt')

        processed_image, matrix = prep_image(image)
        cv2.imwrite(image_save_path, processed_image)

        processed_labels = []
        for label in labels:
            label.key_points.transform_local(matrix)
            processed_labels.append(label)
        save_label(processed_labels, label_save_path)

        counter += 1
        if max_count is not None and counter == max_count:
            break
        print(f"\rprocessed: {counter}", end="")
    print(f"\ndone.\n{counter} images/labels processed in total.")
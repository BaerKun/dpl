import os
from preprocess import Label, load_label


def remove(mode: str = "train"):
    images_folder = os.path.join("backup/images", mode)
    labels_folder = os.path.join("backup/labels", mode)

    for label in os.listdir(labels_folder):
        label_path = os.path.join(labels_folder, label)
        label_lines = load_label(label_path)
        if len(label_lines) > 2:
            image_path = os.path.join(images_folder, label.replace(".txt", ".png"))
            os.remove(image_path)
            os.remove(label_path)


def statistics(labels_folder: str):
    class_map = ["R", "INACTIVATED", "ACTIVATE"]
    class_num = [0, 0, 0]
    freq_num = [0, 0, 0, 0, 0, 0, 0]

    for label_filename in os.listdir(labels_folder):
        label_path = os.path.join(labels_folder, label_filename)
        labels = load_label(label_path)
        freq_num[len(labels)] += 1
        for label in labels:
            class_num[label.cls] += 1

    for i in range(len(class_num)):
        print(class_map[i], ": ", class_num[i])

    for i in range(len(freq_num)):
        print(i, ": ", freq_num[i])

    print("total: ", sum(freq_num))


if __name__ == '__main__':
    folder = "backup/labels/train"
    statistics(folder)
import numpy as np


def to_one_hot(datas):
    labels = datas["train_labels"]
    label_num = labels.max()
    eye = np.eye(label_num + 1)

    datas["train_labels"] = eye[labels]

    labels = datas["test_labels"]
    datas["test_labels"] = eye[labels]

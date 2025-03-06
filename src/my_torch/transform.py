import numpy as np


def to_one_hot(datas):
    labels = datas["train_labels"]
    label_num = labels.max()
    eye = np.eye(label_num + 1)

    datas["train_labels"] = eye[labels]

    labels = datas["test_labels"]
    datas["test_labels"] = eye[labels]


def to_float32(datas):
    train = datas["train_datas"]
    test = datas["test_datas"]
    train = train.astype(np.float32)
    train /= 255.0
    test = test.astype(np.float32)
    test /= 255.0

    datas["train_datas"] = train
    datas["test_datas"] = test
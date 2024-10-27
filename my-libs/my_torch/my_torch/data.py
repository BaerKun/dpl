import pickle
import numpy as np


class Dataset:
    def __init__(self, data_path, transform=None):
        self.datas = dict(zip(("train_datas", "train_labels", "test_datas", "test_labels"), 
                          pickle.load(open(data_path, "rb")).values()))

        if transform is not None:
            for tf in transform:
                tf(self.datas)

    def get_train(self):
        return self.datas["train_datas"], self.datas["train_labels"]
    
    def get_test(self):
        return self.datas["test_datas"], self.datas["test_labels"]

    def save(self, path):
        pickle.dump(self.datas, open(path, "wb"))


class DataLoader:
    __index: int
    __datas: np.ndarray
    __labels: np.ndarray
    __number: int

    def __init__(self, dataset, batch_size, train=True):
        self.dataset = dataset
        self.batch_size = batch_size

        if train:
            self.to_train()

        else:
            self.to_test()

    def to_train(self):
        self.__datas, self.__labels = self.dataset.get_train()
        self.__number = self.__datas.shape[0]
    
    def to_test(self):
        self.__datas, self.__labels = self.dataset.get_test()
        self.__number = self.__datas.shape[0]

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index + self.batch_size > self.__number:
            raise StopIteration

        datas = self.__datas[self.__index:self.__index + self.batch_size]
        labels = self.__labels[self.__index:self.__index + self.batch_size]

        self.__index += self.batch_size
        return datas, labels

    def __len__(self):
        return self.__datas.shape[0]

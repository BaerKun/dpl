import torch
import torch.utils.data
from random import randint


# 将序列分段打包，方便DataLoader加载
class SeqPacker(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.TensorDataset, num_steps, random_offset=True):
        self.tensors = tuple(t.contiguous() for t in dataset.tensors)
        self.num_steps = num_steps
        self.random_offset = random_offset

        self.__update(randint(0, self.num_steps - 1) if random_offset else 0)

    def __len__(self):
        return self.__num_sub_seqs

    def __getitem__(self, item):
        return tuple(t[item] for t in self.__packed_tensors)

    def update(self):
        if self.random_offset:
            self.__update(randint(0, self.num_steps - 1))

    def __new_shape(self, shape):
        if len(shape) == 1:
            return self.__num_sub_seqs, self.num_steps
        return self.__num_sub_seqs, self.num_steps, *shape[1:]

    def __update(self, offset):
        self.__num_sub_seqs = (len(self.tensors[0]) - offset) // self.num_steps
        invalid_end = self.__num_sub_seqs * self.num_steps + offset

        self.__packed_tensors = tuple(t[offset: invalid_end].view(self.__new_shape(t.shape)) for t in self.tensors)


# 作为参数加入DataLoader，可实现数据集的动态调整，而无需重新创建DataLoader
# 但此时DataLoader.__len__()可能将不再是常数
# shuffle集成到DynamicSampler中，不要再传入DataLoader
class DynamicSampler(torch.utils.data.Sampler):
    # update：当DataLoader调用__iter__时，调用dataset.update方法（如果有）
    def __init__(self, dataset, shuffle=True, update=True):
        super().__init__()

        self.dataset = dataset
        self.shuffle = shuffle
        self.update = update

        if shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.__generator = torch.Generator()
            self.__generator.manual_seed(seed)
        else:
            self.__generator = None

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.update:
            self.dataset.update()

        if self.shuffle:
            yield from torch.randperm(len(self.dataset), generator=self.__generator).tolist()
        else:
            yield from range(len(self.dataset))

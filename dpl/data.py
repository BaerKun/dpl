import torch


class SeqPacker(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.TensorDataset, num_steps, predict_steps=1, random_offset=True):
        self.tensors = tuple(t.contiguous() for t in dataset.tensors)
        self.num_steps = num_steps
        self.predict_steps = predict_steps
        self.random_offset = random_offset

        self.__refresh(0)

    def __len__(self):
        return self.__num_sub_seqs

    def __getitem__(self, item):
        return tuple(t[item] for t in self.__packed_tensors)

    def refresh(self):
        from random import randint

        if self.random_offset:
            self.__refresh(randint(0, self.num_steps - 1))

    def __new_shape(self, shape):
        if len(shape) == 1:
            return self.__num_sub_seqs, self.num_steps
        return self.__num_sub_seqs, self.num_steps, *shape[1:]

    def __refresh(self, offset):
        self.__num_sub_seqs = (len(self.tensors[0]) - offset - self.predict_steps) // self.num_steps
        invalid_end = self.__num_sub_seqs * self.num_steps + offset

        self.__packed_tensors = tuple(t[offset: invalid_end].view(self.__new_shape(t.shape)) for t in self.tensors)


class SeqSampler(torch.utils.data.Sampler):
    def __init__(self, packer, shuffle):
        super().__init__()

        self.packer = packer
        self.shuffle = shuffle

        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self):
        return len(self.packer)

    def __iter__(self):
        self.packer.refresh()
        yield from torch.randperm(len(self.packer), generator=self.generator).tolist()


class SeqDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.TensorDataset,
                 batch_size, num_steps: int,
                 predict_steps=1, random_offset=True,
                 shuffle=True, drop_last=False, num_workers=0):
        packer = SeqPacker(dataset, num_steps, predict_steps, random_offset)
        super().__init__(packer, batch_size, sampler=SeqSampler(packer, shuffle),
                         num_workers=num_workers, drop_last=drop_last)

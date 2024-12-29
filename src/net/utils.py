import torch
import numpy as np
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelManager:
    def __init__(self, model: torch.nn.Module | str):
        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, str):
            self.model = torch.load(model)
        else:
            raise TypeError("model must be a torch.nn.Module or a path to a model.")

    def train(self, train_loader: torch.utils.data.DataLoader, loss_f, epochs: int, lr: float=0.001,
              device: torch.device = None):
        device = self.__try_cuda(device)

        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        num_iter_per_epoch = str(len(train_loader))
        print_cuda_memory = device.type == "cuda"
        print("start training.")
        print(f"device: {device.type}.")

        for epoch in range(1, epochs + 1):
            total_loss = 0.
            curr_iter = 0
            if not print_cuda_memory:
                print(f"epoch {epoch}...")
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                y_hat = self.model(x)
                loss = loss_f(y_hat, y)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                curr_iter += 1

                if print_cuda_memory:
                    print(f"CUDA memory usage: {torch.cuda.memory_reserved() // 1024 // 1024} MB.")
                    print("epoch 1...")
                    print_cuda_memory = False
                if curr_iter % 5 == 0:
                    print(f"\r{curr_iter}/{num_iter_per_epoch}. sum of loss: {total_loss:.8f}", end="")
            print(f"\r{num_iter_per_epoch}/{num_iter_per_epoch}.")
            print(f"sum of loss: {total_loss}.")

    def test(self, test_loader: torch.utils.data.DataLoader, device: torch.device = None, mode: str = "acc",
             loss_f=None):
        device = self.__try_cuda(device)
        self.model.to(device)
        self.model.eval()

        result = 0.
        for x, l in test_loader:
            x = x.to(device)
            l = l.to(device)
            y = self.model(x)

            if mode == 'acc':
                y_pred = torch.argmax(y, dim=1)
                result += torch.sum(y_pred == l).item() / len(l)
            elif mode == 'loss':
                if loss_f is None:
                    raise ValueError("loss_f must be specified when mode is 'loss'.")
                result += loss_f(y, l).item()
            else:
                pass

        print(f"{mode}: {result / len(test_loader)}")

    def predict(self, x: torch.Tensor, device: torch.device = None):
        device = self.__try_cuda(device)

        self.model.to(device)
        self.model.eval()
        x = x.to(device)
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model, path)

    @staticmethod
    def __try_cuda(device: torch.device):
        if device is not None:
            return device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PackedSeqDataset(torch.utils.data.Dataset):
    def __init__(self, seq: torch.Tensor, num_steps, offset):
        self.__num_sub_seqs = (len(seq) - offset - 1) // num_steps
        invalid_end = self.__num_sub_seqs * num_steps + offset
        self.data = seq[offset: invalid_end].reshape((self.__num_sub_seqs, num_steps))
        self.label = seq[offset + 1: invalid_end + 1].reshape((self.__num_sub_seqs, num_steps))

    def __len__(self):
        return self.__num_sub_seqs

    def __getitem__(self, item):
        return self.data[item], self.label[item]


class Vocab:
    def __init__(self, corpus, min_count=0):
        import collections

        token_counter = collections.Counter(corpus)
        sorted_counter = sorted(token_counter.items(), key=lambda x: x[1])

        self.token2idx = {'<unk>': 0}
        self.idx2token = ['<unk>']

        i = 1
        for token, count in sorted_counter:
            if count < min_count:
                self.token2idx[token] = 0
            else:
                self.idx2token.append(token)
                self.token2idx[token] = i
                i += 1

    def decode(self, indices) -> list[str]:
        return [self.idx2token[idx] for idx in indices]

    def encode(self, tokens) -> list[int]:
        return [self.token2idx[token] for token in tokens]

    def __len__(self):
        return len(self.idx2token)


class Corpus:
    vocab: Vocab
    corpus_idx: torch.Tensor

    def __init__(self, root: str):
        with open(root, 'r') as f:
            self.corpus = f.read().split()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]

    def build_vocab(self, min_count=0):
        self.vocab = Vocab(self.corpus, min_count)
        self.corpus_idx = torch.tensor(self.vocab.encode(self.corpus))

    def get_loader(self, batch_size, num_steps, random_offset=True, random_sample=False):
        import random
        packed_seq = PackedSeqDataset(self.corpus_idx, num_steps,
                                      random.randint(0, num_steps - 1) if random_offset else 0)
        return torch.utils.data.DataLoader(packed_seq, batch_size=batch_size, shuffle=random_sample, drop_last=True)


def load_fashion_mnist(batch_size, size=28, train=True) -> torch.utils.data.DataLoader:
    import torchvision

    # str_label = ["T-shit", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    mnist_train = torchvision.datasets.FashionMNIST(root=os.path.join(project_root, "data"),
                                                    transform=torchvision.transforms.ToTensor() if size == 28 else
                                                    torchvision.transforms.Compose(
                                                        (torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize(size)))
                                                    , train=train, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    return mnist_loader


def load_wiki_text(mode: str = "train") -> Corpus:
    import os

    path = os.path.join(project_root, "data", "wikitext-2", f"wiki.{mode}.tokens")
    return Corpus(path)


def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    img_np = tensor.permute(1, 2, 0).numpy()
    img_np *= 255
    return img_np.astype(np.uint8)


def show_image(image: np.ndarray, label: str = None, prob: float = None):
    import cv2

    if label is not None:
        cv2.putText(image, f"{label}" if prob is None else f"{label} {prob:.2f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    cv2.imshow("test", image)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        return False
    return True

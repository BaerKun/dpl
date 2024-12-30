import torch
import numpy as np
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelManager:
    def __init__(self, model: torch.nn.Module | str, seq2seq=False, output_state=False):
        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, str):
            self.model = torch.load(model)
        else:
            raise TypeError("model must be a torch.nn.Module or a path to a model.")

        self.__seq2seq = seq2seq
        self.__output_state = output_state

    def train(self, loader: torch.utils.data.DataLoader, loss_f, epochs: int, lr: float = 0.001,
              warmup_steps=0, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        num_iter_per_epoch = str(len(loader))
        print_cuda_memory = device.type == "cuda"
        print("start training.")
        print(f"device: {device.type}.")

        for epoch in range(1, epochs + 1):
            total_loss = 0.
            curr_iter = 0
            if not print_cuda_memory:
                print(f"epoch {epoch}/{epochs}...")
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                y_hat = self.model(x)
                if self.__seq2seq:
                    if self.__output_state:
                        y_hat = y_hat[0]
                    if warmup_steps != 0:
                        y_hat = y_hat[:, warmup_steps:]
                        y = y[:, warmup_steps:]
                    y_hat = y_hat.flatten(0, 1)
                    y = y.flatten(0, 1)

                loss = loss_f(y_hat, y)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                curr_iter += 1

                if print_cuda_memory:
                    print(f"CUDA memory usage: {torch.cuda.memory_reserved() // 1024 // 1024} MB.")
                    print(f"epoch 1/{epochs}...")
                    print_cuda_memory = False
                if curr_iter % 5 == 0:
                    print(f"\r{curr_iter}/{num_iter_per_epoch}. sum of loss: {total_loss:.4f}", end="")
            print(f"\r{num_iter_per_epoch}/{num_iter_per_epoch}.")
            print(f"sum of loss: {total_loss}.")

    def test(self, loader: torch.utils.data.DataLoader, score_f, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()

        num_iter_per_epoch = str(len(loader))
        curr_iter = 0
        score = 0.
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = self.model(x)
            if self.__output_state:
                y_hat = y_hat[0]

            score += score_f(y_hat, y)

            curr_iter += 1
            if curr_iter % 5 == 0:
                print(f"\r{curr_iter}/{num_iter_per_epoch}.", end="")
        print(f"\r{num_iter_per_epoch}/{num_iter_per_epoch}.")
        print(f"mean score: {score / len(loader)}")

    def predict(self, x: torch.Tensor, device: torch.device = try_cuda):
        self.model.to(device)
        self.model.eval()
        x = x.to(device)
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model, path)

    # return (batch_size, predict_steps)
    def predict_seq(self, x: torch.Tensor, predict_steps, device: torch.device = try_cuda) -> torch.Tensor:
        self.model.to(device)
        self.model.eval()
        x = x.to(device)

        y_hat, state = self.model(x)
        y_prev = y_hat[:, -1].argmax(dim=1, keepdims=True)
        y_pred = [y_prev]

        for i in range(predict_steps - 1):
            y_hat, state = self.model(y_prev, state)
            y_prev = y_hat.argmax(dim=2)
            y_pred.append(y_prev)

        return torch.cat(y_pred, dim=1)

    def predict_with_image(self, x: torch.Tensor, str_labels=None, device: torch.device = try_cuda) -> bool:
        self.model.to(device)
        self.model.eval()

        x = x.to(device)
        y_hat = self.model(x)
        for x_b, y_hat_b in zip(x, y_hat):
            img = tensor2image(x_b)
            prob = y_hat_b.softmax(dim=0).max()
            label = y_hat_b.argmax(dim=0)
            label = str(label) if str_labels is None else str_labels[label]
            if show_image(img, label, prob) is False:
                return False


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
    def __init__(self, corpus: list[str], min_freq=0, max_vocab_size=None):
        self.token2idx = self.__build_vocab(corpus, min_freq, max_vocab_size)
        self.idx2token = list(self.token2idx.keys())
        self.vocab_size = len(self.idx2token)

    def decode(self, indices) -> list[str]:
        return [self.idx2token[idx] if idx < self.vocab_size else '<unk>' for idx in indices]

    def encode(self, tokens) -> list[int]:
        return [self.token2idx[token] if token in self.token2idx else self.vocab_size - 1 for token in tokens]

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def __build_vocab(corpus, min_frq, max_vocab_size):
        import collections

        token_counter = collections.Counter(corpus)
        token_counter.pop('<unk>', None)
        sorted_counter = sorted(token_counter.items(), key=lambda x: x[1])

        i = 0
        for token, count in sorted_counter:
            if count >= min_frq:
                break
            i += 1

        vocab_size_no_unk = len(sorted_counter) - i
        if max_vocab_size is not None and vocab_size_no_unk >= max_vocab_size:
            vocab_size_no_unk = max_vocab_size - 1
            i = len(sorted_counter) - vocab_size_no_unk

        vocab = {token: i for i, (token, _) in enumerate(sorted_counter[i:])}
        vocab['<unk>'] = vocab_size_no_unk
        return vocab


# 初次使用先运行 download_nltk
class Corpus:
    tensor_dataset: torch.Tensor

    def __init__(self, root: str, lower=True, filter_stopwords=False, filter_punctuation=False, user_filter:set[str]=None):
        with open(root, 'r') as f:
            corpus = f.read()
        try:
            self.corpus = self.__preprocess_text(corpus, lower, filter_stopwords, filter_punctuation, user_filter)
        except LookupError:
            self.__download_nltk()
            self.corpus = self.__preprocess_text(corpus, lower, filter_stopwords, filter_punctuation, user_filter)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]

    def build_vocab(self, min_freq=0, max_vocab_size=None) -> Vocab:
        vocab = Vocab(self.corpus, min_freq, max_vocab_size)
        self.tensor_dataset = torch.tensor(vocab.encode(self.corpus))
        print("vocab size: ", len(vocab))
        return vocab

    def get_loader(self, batch_size, num_steps, random_offset=True, random_sample=False):
        import random
        packed_seq = PackedSeqDataset(self.tensor_dataset, num_steps,
                                      random.randint(0, num_steps - 1) if random_offset else 0)
        return torch.utils.data.DataLoader(packed_seq, batch_size=batch_size, shuffle=random_sample, drop_last=True)

    @staticmethod
    def __preprocess_text(corpus, lower, filter_stopwords, filter_punctuation, user_filter):
        import nltk

        filter_set = set() if user_filter is None else user_filter
        if filter_stopwords:
            stopwords = set(nltk.corpus.stopwords.words('english'))
            filter_set |= stopwords
        if filter_punctuation:
            import string
            punctuation = set(string.punctuation)
            filter_set |= punctuation
        if lower:
            corpus = corpus.lower()

        corpus = nltk.word_tokenize(corpus) # 主要的性能消耗
        if len(filter_set) != 0:
            corpus = [token for token in corpus if token not in filter_set]

        return corpus

    @staticmethod
    def __download_nltk():
        import nltk

        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('stopwords')



def load_fashion_mnist(batch_size, size=28, train=True, get_labels=False):
    import torchvision

    str_labels = ["T-shit", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    mnist_train = torchvision.datasets.FashionMNIST(root=os.path.join(project_root, "data"),
                                                    transform=torchvision.transforms.ToTensor() if size == 28 else
                                                    torchvision.transforms.Compose(
                                                        (torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize(size)))
                                                    , train=train, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    if get_labels:
        return mnist_loader, str_labels
    return mnist_loader


def load_wiki_text(mode: str = "train") -> Corpus:
    import os

    path = os.path.join(project_root, "data", "wikitext-2", f"wiki.{mode}.tokens")
    return Corpus(path, user_filter={'=', '<', '>', 'unk'})


def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    img_np = tensor.permute(1, 2, 0).numpy()
    img_np *= 255
    return img_np.astype(np.uint8)


def show_image(image: np.ndarray, label: str, prob: float = None) -> bool:
    import cv2

    if prob is not None:
        label += f" {prob:.2f}"
    cv2.putText(image, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("test", image)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        return False
    return True


def score_acc(y_hat: torch.Tensor, y: torch.Tensor):
    y_hat = y_hat.argmax(dim=1)
    return (y_hat == y).float().mean().item()

import torch
import numpy as np


class ModelManager:
    def __init__(self, model: torch.nn.Module | str):
        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, str):
            self.model = torch.load(model)
        else:
            raise TypeError("model must be a torch.nn.Module or a path to a model.")

    def train(self, train_loader: torch.utils.data.DataLoader, loss_function, epochs: int, lr: float = 0.001,
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
            for img, label in train_loader:
                img = img.to(device)
                label = label.to(device)

                y = self.model(img)
                loss = loss_function(y, label)
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


class Corpus(torch.utils.data.Dataset):
    def __init__(self, root: str, num_steps: int, warm_up: int = 1):
        with open(root, 'r') as f:
            self.corpus = f.read().split()
        self.__count_corpus(self.corpus)
        self.num_steps = num_steps
        self.warm_up = warm_up

    def __count_corpus(self, corpus):
        import collections

        counter = collections.Counter(corpus)
        self.vocab = {token: i for i, (token, count) in enumerate(counter.items())}
        self.idx2token = tuple(self.vocab.keys())
        self.corpus_idx = [self.vocab[token] for token in corpus]

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus_idx[item: item + self.num_steps], self.corpus_idx[
                                                             item + self.warm_up: item + self.num_steps + 1]


def load_fashion_mnist(batch_size, size=28, train=True) -> tuple[torch.utils.data.DataLoader, list[str]]:
    import torchvision

    str_label = ["T-shit", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    mnist_train = torchvision.datasets.FashionMNIST(root='../../data',
                                                    transform=torchvision.transforms.ToTensor() if size == 28 else
                                                    torchvision.transforms.Compose(
                                                        (torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Resize(size)))
                                                    , train=train, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    return mnist_loader, str_label


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

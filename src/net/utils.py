import torch


class ModelManager:
    def __init__(self, model: torch.nn.Module | str):
        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, str):
            self.model = torch.load(model)
        else:
            raise TypeError("model must be a torch.nn.Module or a path to a model.")

    def train(self, train_loader: torch.utils.data.DataLoader, loss_function, epochs: int, lr: float=0.001,
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
                    print(f"\r{curr_iter}/{num_iter_per_epoch}. sum of loss: {total_loss:.2f}", end="")
            print(f"\r{num_iter_per_epoch}/{num_iter_per_epoch}.")
            print(f"sum of loss: {total_loss}.")

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

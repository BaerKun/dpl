import os.path
import torch
from .text import Corpus

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
fashion_mnist_labels = ["T-shit", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                        "Ankle Boot"]


def load_fashion_mnist(batch_size, size=28, train=True):
    import torchvision

    mnist = torchvision.datasets.FashionMNIST(root=os.path.join(project_root, "data"),
                                              transform=torchvision.transforms.ToTensor() if size == 28 else
                                              torchvision.transforms.Compose(
                                                  (torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize(size)))
                                              , train=train, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

    return mnist_loader


def load_wiki_text(mode: str = "train") -> Corpus:
    import os

    path = os.path.join(project_root, "data", "wikitext-2", f"wiki.{mode}.tokens")
    return Corpus(path, user_filter={'=', '<', '>', 'unk'})

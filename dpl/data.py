import os
import torch
from .sequence import Corpus


def load_fashion_mnist(dataset_dir, batch_size, size=28, train=True, return_str_labels=False):
    import torchvision

    fashion_mnist_labels = ["T-shit", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                            "Ankle Boot"]

    mnist = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                              transform=torchvision.transforms.ToTensor() if size == 28 else
                                              torchvision.transforms.Compose(
                                                  (torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize(size)))
                                              , train=train, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

    if return_str_labels:
        return mnist_loader, fashion_mnist_labels
    return mnist_loader


def load_wiki_text(dataset_dir, mode: str = "train") -> Corpus:
    path = os.path.join(dataset_dir, f"wikitext-2/wiki.{mode}.tokens")
    return Corpus(path, user_filter={'=', '<', '>', 'unk'})

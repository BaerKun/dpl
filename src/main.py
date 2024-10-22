import torch
import torchvision
import cv2


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(784, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


str_label = ("T-shit", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

mnist_test = torchvision.datasets.FashionMNIST(root='../data', transform=torchvision.transforms.ToTensor(),
                                               train=False, download=True, )

mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)

loss_func = torch.nn.CrossEntropyLoss()
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

for img, label in mnist_loader:
    y = net(img)
    loss = loss_func(y, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())

import torch
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


net = torch.nn.Sequential(
    torch.nn.Conv2d(1, 6, 5, 1, 2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(6, 16, 5, 1, 0),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(400, 120),
    torch.nn.ReLU(),
    torch.nn.Linear(120, 84),
    torch.nn.ReLU(),
    torch.nn.Linear(84, 10),
    torch.nn.Softmax(1)
).to(device)


str_label = ("T-shit", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

mnist_train = torchvision.datasets.FashionMNIST(root='../data', transform=torchvision.transforms.ToTensor(),
                                               train=True, download=True)

mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

for _ in range(10):
    total_loss = 0
    for img, label in mnist_loader:
        img = img.to(device)
        label = label.to(device)
        y = net(img)
        loss = loss_func(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)

import torch
import utils

num_classes = 10
net = torch.nn.Sequential(
    torch.nn.Conv2d(1, 6, 5, 1, 2),
    torch.nn.MaxPool2d(2, 2), torch.nn.ReLU(),
    torch.nn.Conv2d(6, 16, 5, 1, 0),
    torch.nn.MaxPool2d(2, 2), torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(400, 120), torch.nn.ReLU(),
    torch.nn.Linear(120, 84), torch.nn.ReLU(),
    torch.nn.Linear(84, num_classes)
)

loader = utils.load_fashion_mnist(128)
loss_func = torch.nn.CrossEntropyLoss()
mm = utils.ModelManager(net, "../../weights/lenet.pt")
mm.train(loader, loss_func, 50)
mm.test(loader, utils.score_acc)

loader = utils.load_fashion_mnist(4096, train=False)
mm.test(loader, utils.score_acc)

mm.save("../../weights/lenet.pt")

import os
import numpy as np
import optim, trainer, data, network, loss, transform, mnist
from dpl.data import project_root

mnist_dir = os.path.join(project_root, "datasets/mnist")
mnist.init_mnist(mnist_dir)
dataset = data.Dataset(os.path.join(mnist_dir, "mnist.pkl"), (transform.to_one_hot, transform.to_float32))

dataloader = data.DataLoader(dataset, 512)

net = network.LeNet()

loss_func = loss.CrossEntropyLoss()

optimizer = optim.AdaGrad(0.00001, 0.0001)
optimizer.push_params(net.get_params())

_trainer = trainer.Trainer(dataset, dataloader, net, loss_func, optimizer)

for loss in _trainer.train(10):
    print(loss)

accuracy = 0
dataloader.to_test()
for _data, _target in dataloader:
    output = net(_data)
    accuracy += np.sum(np.argmax(output, axis=1) == np.argmax(_target, axis=1))

accuracy /= len(dataloader)
print(accuracy)

net.save(os.path.join(project_root, "m1.pkl"))

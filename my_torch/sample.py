import numpy as np

from my_torch import optimizer, trainer, data, network, loss, transform

dataset = data.Dataset("mnist.pkl", (transform.to_one_hot, transform.to_float32))
dataloader = data.DataLoader(dataset, 512)
net = network.load("m1.pkl")
loss_func = loss.CrossEntropyLoss()
optimizer = optimizer.SGD(0.00001, 0.0001)
optimizer.push_params(net.get_params())

_trainer = trainer.Trainer(dataset, dataloader, net, loss_func, optimizer)

for loss in _trainer.train(20):
    print(loss)

accuracy = 0
dataloader.to_test()
for _data, _target in dataloader:
    output = net.forward(_data)
    accuracy += np.sum(np.argmax(output, axis=1) == np.argmax(_target, axis=1))

accuracy /= len(dataloader)
print(accuracy)

net.save("m1.pkl")

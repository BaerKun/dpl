import numpy as np

from my_torch import optimizer, trainer, data, layer, network, loss, transform

layers = [
    layer.Convolution2d(1, 32, 3, 1, 1),
    layer.ReLU(),
    layer.MaxPool2d(2),
    layer.Flatten(),
    layer.Affine(32 * 14 * 14, 10),
    layer.Normalization(),
    layer.Softmax()
]

dataset = data.Dataset("../../data/mnist.pkl", (transform.to_one_hot,))
dataloader = data.DataLoader(dataset, 512)
net = network.NetWork(layers)
loss_func = loss.CrossEntropyLoss()
optimizer = optimizer.Momentum(0.01)
optimizer.push_params(net.get_params())

_trainer = trainer.Trainer(dataset, dataloader, net, loss_func, optimizer)

for loss in _trainer.train(30):
    print(loss)

accuracy = 0
dataloader.to_test()
for _data, _target in dataloader:
    output = net.forward(_data)
    accuracy += np.sum(np.argmax(output, axis=1) == np.argmax(_target, axis=1))

accuracy /= len(dataloader)
print(accuracy)

net.save("../../model/m1.pkl")

import numpy as np

import loss
import optimizer
import network
import data
import layer
import transform
import trainer

layers = [layer.Convolution2d(1, 32, 3, 1),
          layer.ReLU(),
          layer.Convolution2d(32, 64, 3, 1),
          layer.ReLU(),
          layer.MaxPool2d(2),
          layer.Flatten(),
          layer.Affine(64 * 12 * 12, 128),
          layer.ReLU(),
          layer.Affine(128, 10),
          layer.Softmax()
          ]

dataset = data.Dataset("../dataset/mnist.pkl", (transform.to_one_hot,))
dataloader = data.DataLoader(dataset, 64)
net = network.NetWork(layers)
loss_func = loss.CrossEntropyLoss()
optimizer = optimizer.AdaGrad(0.01)
optimizer.push_params(net.get_params())

_trainer = trainer.Trainer(dataset, dataloader, net, loss_func, optimizer)

for loss in _trainer.train(1, True):
    print(loss)

accuracy = 0
dataloader.to_test()
for _data, _target in dataloader:
    output = net.forward(_data)
    accuracy += np.sum(np.argmax(output, axis=1) == np.argmax(_target, axis=1)) / 64

print(accuracy)

net.save("../model/m1.pkl")

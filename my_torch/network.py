import pickle
from my_torch import layer


def load(load_path):
    cls = NetWork.__new__(NetWork)

    _load = pickle.load(open(load_path, 'rb'))
    for layer_t, params in _load:
        cls.layers.append(layer_t.restore(params))

    return cls


class NetWork:
    param_list: list = None
    layers: list = []

    def __init__(self, layer_list):
        self.layers = layer_list

    def forward(self, tensor):
        for _layer in self.layers:
            tensor = _layer.forward(tensor)

        return tensor

    def __call__(self, tensor):
        return self.forward(tensor)

    def backward(self, grad):
        for _layer in reversed(self.layers):
            grad = _layer.backward(grad)

    def get_params(self):
        if self.param_list is None:
            self.param_list = []
            for _layer in self.layers:
                _layer.get_params(self.param_list)

        return self.param_list

    def print_params(self):
        for _param in self.param_list:
            print(_param)

    def save(self, save_path):
        _save = []
        for _layer in self.layers:
            _params = _layer.get_restore_params()
            _save.append((type(_layer), _params))

        pickle.dump(_save, open(save_path, 'wb'))


class LeNet(NetWork):
    def __init__(self):
        super().__init__([
            layer.Convolution2d(1, 6, 5, 1, 2),
            layer.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Convolution2d(6, 16, 5, 1, 0),
            layer.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
            layer.Affine(400, 120),
            layer.ReLU(),
            layer.Affine(120, 84),
            layer.ReLU(),
            layer.Affine(84, 10),
            layer.Softmax()
        ])

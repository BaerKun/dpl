import pickle


class NetWork:
    param_list = None

    def __init__(self, layer_list):
        self.layers = layer_list
        
    def forward(self, tensor):
        for _layer in self.layers:
            tensor = _layer.forward(tensor)

        return tensor

    def backward(self, loss):
        grad = loss.backward()

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
            _save.append((type(_layer), _layer.get_restore_params()))

        pickle.dump(_save, open(save_path, 'wb'))

    @staticmethod
    def load(load_path):
        cls = NetWork.__new__(NetWork)

        _load = pickle.load(open(load_path, 'rb'))
        for layer_t, params in _load:
            cls.layers.append(layer_t.restore(params))

        return cls

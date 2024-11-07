import numpy as np

from my_torch import param


# 各层输入tensor的数据不要修改
# 各层之间grad的shape不要修改


def _to_tuple2(arg):
    arg_type = type(arg)
    if arg_type is int:
        return arg, arg
    elif arg_type is tuple and len(arg) == 2:
        return arg
    raise ValueError("Unsupported argument type, only int and tuple[int, int] are allowed")


class Layer:
    def __init__(self):
        pass

    def forward(self, tensor):
        pass

    def backward(self, grad):
        pass

    def get_params(self, param_list):
        pass

    def get_restore_params(self):
        pass

    @staticmethod
    def restore(param_dict):
        pass


class ReLU(Layer):
    __mask: np.ndarray = None
    __out_tensor: np.ndarray = None

    def forward(self, tensor: np.ndarray):
        if self.__mask is None:
            self.__mask = np.ndarray(tensor.shape, dtype=np.bool_)
            self.__out_tensor = np.ndarray(tensor.shape, dtype=np.float32)

        self.__mask[:] = tensor < 0
        self.__out_tensor[:] = tensor
        self.__out_tensor[self.__mask] = 0

        return self.__out_tensor

    def backward(self, grad):
        grad[self.__mask] = 0

        return grad

    @staticmethod
    def restore(none):
        return ReLU.__new__(ReLU)


class Sigmoid(Layer):
    __out_tensor: np.ndarray = None
    __out_grad: np.ndarray = None

    def forward(self, tensor):
        if self.__out_tensor is None:
            self.__out_tensor = np.ndarray(tensor.shape, dtype=np.float32)
            self.__out_grad = np.ndarray(tensor.shape, dtype=np.float32)

        np.negative(tensor, out=self.__out_tensor)
        np.exp(self.__out_tensor, out=self.__out_tensor)
        self.__out_tensor += 1.0
        np.reciprocal(self.__out_tensor, out=self.__out_tensor)

        return self.__out_tensor

    def backward(self, grad):
        np.subtract(1., self.__out_tensor, out=self.__out_grad)
        np.multiply(self.__out_grad, self.__out_tensor, grad, out=self.__out_grad)

        return self.__out_grad

    @staticmethod
    def restore(param_dict):
        return Sigmoid.__new__(Sigmoid)


class Affine(Layer):
    __in_tensor: np.ndarray
    __out_tensor: np.ndarray = None
    __out_grad: np.ndarray = None

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.wight = param.xavier((in_dims, out_dims), in_dims + out_dims)
        self.bias = param.xavier(out_dims, out_dims)

    def forward(self, tensor):
        if self.__out_tensor is None:
            self.__out_tensor = np.ndarray((tensor.shape[0], self.wight.shape[1]), dtype=np.float32)
            self.__out_grad = np.ndarray(tensor.shape, dtype=np.float32)

        self.__in_tensor = tensor
        np.matmul(tensor, self.wight.data, out=self.__out_tensor)
        self.__out_tensor += self.bias.data

        return self.__out_tensor

    def backward(self, grad: np.ndarray):
        np.mean(grad, axis=0, out=self.bias.grad)
        np.matmul(self.__in_tensor.T, grad, out=self.wight.grad)

        np.matmul(grad, self.wight.data.T, out=self.__out_grad)
        return self.__out_grad

    def get_params(self, params_list):
        params_list.append(self.wight)
        params_list.append(self.bias)

    def get_restore_params(self):
        param_dict = dict(wight=self.wight.data, bias=self.bias.data)
        return param_dict

    @staticmethod
    def restore(param_dict):
        cls = Affine.__new__(Affine)
        cls.wight = param.Param(param_dict["wight"])
        cls.bias = param.Param(param_dict["bias"])

        return cls


# 依赖 out_tensor
class Softmax(Layer):
    __out_tensor: np.ndarray = None
    __out_grad: np.ndarray = None
    __max_sum: np.ndarray = None

    def forward(self, tensor):
        if self.__out_tensor is None:
            self.__out_tensor = np.ndarray(tensor.shape, dtype=np.float32)
            self.__out_grad = np.ndarray(tensor.shape, dtype=np.float32)
            self.__max_sum = self.__out_grad[..., 0][..., np.newaxis]  # 借用内存

        tensor.max(axis=-1, keepdims=True, out=self.__max_sum)
        np.subtract(tensor, self.__max_sum, out=self.__out_tensor)
        np.exp(self.__out_tensor, out=self.__out_tensor)

        self.__out_tensor.sum(axis=-1, keepdims=True, out=self.__max_sum)
        self.__out_tensor /= self.__max_sum

        return self.__out_tensor

    def backward(self, in_grad):
        np.subtract(1., self.__out_tensor, out=self.__out_grad)
        self.__out_grad *= self.__out_tensor
        self.__out_grad *= in_grad

        return self.__out_grad

    @staticmethod
    def restore(none):
        return Softmax.__new__(Softmax)


class Flatten(Layer):
    start_dim: int
    end_dim: int
    __out_shape = None
    __original_shape = None

    def __calc_out_shape(self):
        length = len(self.__original_shape)
        if self.end_dim == -1:
            self.end_dim = length - 1

        self.__out_shape = []
        dim = 0
        dim_size = 1
        while dim < length:
            dim_size *= self.__original_shape[dim]
            if dim < self.start_dim or dim >= self.end_dim:
                self.__out_shape.append(dim_size)
                dim_size = 1
            dim += 1

        self.__out_shape = tuple(self.__out_shape)

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, tensor):
        if self.__out_shape is None:
            self.__original_shape = tensor.shape
            self.__calc_out_shape()

        return tensor.reshape(self.__out_shape)

    def backward(self, grad):
        return grad.reshape(self.__original_shape)

    def get_restore_params(self):
        param_dict = dict(start_dim=self.start_dim, end_dim=self.end_dim)
        return param_dict

    @staticmethod
    def restore(param_dict):
        return Flatten(param_dict["start_dim"], param_dict["end_dim"])


class Normalization(Layer):
    __out_tensor: np.ndarray = None
    __mean_std: np.ndarray = None
    __N: int = None
    __out_grad: np.ndarray = None

    def __init__(self):
        super().__init__()
        self.wight = param.Param(np.array(1., dtype=np.float32))
        self.bias = param.Param(np.array(0., dtype=np.float32))

    def forward(self, tensor):
        if self.__N is None:
            self.__N = tensor.shape[-1]
            self.__mean_std = np.ndarray((*tensor.shape[:-1], 1), dtype=np.float32)
            self.__out_tensor = np.ndarray(tensor.shape, dtype=np.float32)
            self.__out_grad = np.ndarray(tensor.shape, dtype=np.float32)

        tensor.mean(axis=-1, keepdims=True, out=self.__mean_std)
        np.subtract(tensor, self.__mean_std, out=self.__out_tensor)

        tensor.std(axis=-1, keepdims=True, out=self.__mean_std)
        self.__mean_std += 1e-7

        self.__out_tensor *= self.wight.data
        self.__out_tensor /= self.__mean_std
        self.__out_tensor += self.bias.data

        return self.__out_tensor

    def backward(self, grad):
        np.square(self.__out_tensor, out=self.__out_grad)
        np.subtract(self.__N - 1, self.__out_grad, out=self.__out_grad)
        self.__out_grad *= self.wight.data
        self.__out_grad /= self.__N
        self.__out_grad /= self.__mean_std
        self.__out_grad *= grad

        grad.mean(out=self.bias.grad)

        grad *= self.__out_tensor
        grad.mean(out=self.wight.grad)

        return self.__out_grad

    def get_params(self, params_list):
        params_list.append(self.wight)
        params_list.append(self.bias)

    def get_restore_params(self):
        param_dict = dict(wight=self.wight.data, bias=self.bias.data)
        return param_dict

    @staticmethod
    def restore(param_dict):
        cls = Normalization.__new__(Normalization)
        cls.wight = param.Param(param_dict["wight"])
        cls.bias = param.Param(param_dict["bias"])

        return cls


class Dropout(Layer):
    __mask: np.ndarray = None
    __out_tensor: np.ndarray = None

    def __init__(self, dropout_rate=0.2, train=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.train = train

    def forward(self, tensor):
        if self.train:
            if self.__mask is None:
                self.__mask = np.ndarray(tensor.shape, dtype=np.bool_)
                self.__out_tensor = np.ndarray(tensor.shape, dtype=np.float32)

            self.__mask[:] = np.random.rand(*tensor.shape) < self.dropout_rate
            self.__out_tensor[:] = tensor
            self.__out_tensor[self.__mask] = 0
        else:
            np.multiply(tensor, 1. - self.dropout_rate, out=self.__out_tensor)

        return self.__out_tensor

    def backward(self, grad):
        grad[self.__mask] = 0
        return grad

    def get_restore_params(self):
        param_dict = dict(dropout_rate=self.dropout_rate)
        return param_dict

    @staticmethod
    def restore(param_dict):
        return Dropout(param_dict["dropout_rate"])


class ConvPoolLayer(Layer):
    filter_shape: tuple = None
    stride: tuple = None
    padding: tuple = None
    _batch_size: int = None
    _in_channels: int = None
    _out_tensor_col: np.ndarray = None
    _output_shape: tuple[int, int] = None
    __input_shape: tuple[int, int] = None
    __im2col_static: np.ndarray = None
    __col2mg_static: np.ndarray = None
    __tensor_pad: np.ndarray = None

    def _first_init(self, input_shape, out_channels):
        batch_size, in_channels, input_h, input_w = input_shape
        self.__input_shape = (input_h, input_w)
        self._batch_size = batch_size
        self._in_channels = in_channels

        padded_shape = input_h + 2 * self.padding[0], input_w + 2 * self.padding[1]

        self._output_shape = ((padded_shape[0] - self.filter_shape[0]) // self.stride[0] + 1,
                              (padded_shape[1] - self.filter_shape[1]) // self.stride[1] + 1)

        self._out_tensor_col = np.ndarray(
            (batch_size * self._output_shape[0] * self._output_shape[1],
             (out_channels if out_channels > 0 else in_channels)), dtype=np.float32)

        self.__tensor_pad = np.zeros((batch_size, in_channels, *padded_shape), dtype=np.float32)

        self.__im2col_static = np.ndarray(
            (batch_size, in_channels, *self.filter_shape, *self._output_shape), dtype=np.float32)

        self.__col2mg_static = np.ndarray(
            (batch_size, in_channels, padded_shape[0] + self.stride[0] - 1, padded_shape[1] + self.stride[1] - 1),
            dtype=np.float32)

    def _im2col(self, tensor):
        self.__tensor_pad[:, :, self.padding[0]:self.padding[0] + self.__input_shape[0],
        self.padding[1]:self.padding[1] + self.__input_shape[1]] = tensor

        for y in range(self.filter_shape[0]):
            y_max = y + self.stride[0] * self._output_shape[0]

            for x in range(self.filter_shape[1]):
                x_max = x + self.stride[1] * self._output_shape[1]
                self.__im2col_static[:, :, y, x, :, :] = self.__tensor_pad[:, :, y:y_max:self.stride[0],
                                                         x:x_max:self.stride[1]]

        return self.__im2col_static.transpose(0, 4, 5, 1, 2, 3).reshape(
            self._batch_size * self._output_shape[0] * self._output_shape[1], -1)

    def _col2im(self, col):
        col = (col.reshape(self._batch_size, *self._output_shape, self._in_channels, *self.filter_shape)
               .transpose(0, 3, 4, 5, 1, 2))

        self.__col2mg_static.fill(0)
        for y in range(self.filter_shape[0]):
            y_max = y + self.stride[0] * self._output_shape[0]

            for x in range(self.filter_shape[1]):
                x_max = x + self.stride[1] * self._output_shape[1]
                self.__col2mg_static[:, :, y:y_max:self.stride[0], x:x_max:self.stride[1]] += col[:, :, y, x, :, :]

        return (self.__col2mg_static[:, :, self.padding[0]:self.__input_shape[0] + self.padding[0],
                self.padding[1]:self.__input_shape[1] + self.padding[1]])

    def __init__(self, filter_shape, stride, padding):
        super().__init__()
        self.filter_shape = _to_tuple2(filter_shape)
        self.stride = _to_tuple2(stride)
        self.padding = _to_tuple2(padding)

    def get_restore_params(self):
        param_dict = dict(filter_shape=self.filter_shape, stride=self.stride, padding=self.padding)
        return param_dict

    def _restore(self, param_dict):
        self.filter_shape = param_dict["filter_shape"]
        self.stride = param_dict["stride"]
        self.padding = param_dict["padding"]


class Convolution2d(ConvPoolLayer):
    __tensor_col: np.ndarray = None
    __kernel_col: np.ndarray = None
    kernel: param.Param = None
    bias: param.Param = None

    def __init__(self, in_channels, out_channels, kernel_shape, stride=(1, 1), padding=(0, 0)):
        super().__init__(kernel_shape, stride, padding)
        self.out_channels = out_channels

        self.kernel = param.xavier((out_channels, in_channels, *self.filter_shape),
                                   self.filter_shape[0] * self.filter_shape[1])
        self.bias = param.Param(np.zeros(out_channels, dtype=np.float32))

    def forward(self, tensor):
        if self._batch_size is None:
            self._first_init(tensor.shape, self.out_channels)
            self.__kernel_col = self.kernel.data.reshape(self.out_channels, -1).T

        self.__tensor_col = self._im2col(tensor)

        np.matmul(self.__tensor_col, self.__kernel_col, out=self._out_tensor_col)
        self._out_tensor_col += self.bias.data

        return self._out_tensor_col.reshape(self._batch_size, *self._output_shape, -1).transpose(0, 3, 1, 2)

    def backward(self, grad):
        grad = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        np.mean(grad, axis=0, out=self.bias.grad)

        np.matmul(grad.T, self.__tensor_col, out=self.kernel.grad.reshape(self.out_channels, -1))
        self.kernel.grad.resize(self.out_channels, self._in_channels, *self.filter_shape)

        out_grad = self._col2im(np.matmul(grad, self.__kernel_col.T))
        return out_grad

    def get_params(self, param_list):
        param_list.append(self.kernel)
        param_list.append(self.bias)

    def get_restore_params(self):
        param_dict = super().get_restore_params()
        param_dict["kernel"] = self.kernel.data
        param_dict["bias"] = self.bias.data

        return param_dict

    @staticmethod
    def restore(param_dict):
        cls = Convolution2d.__new__(Convolution2d)
        ConvPoolLayer._restore(cls, param_dict)
        cls.kernel = param.Param(param_dict["kernel"])
        cls.bias = param.Param(param_dict["bias"])
        cls.out_channels, cls._in_channels = cls.kernel.shape[0:2]

        return cls


class MaxPool2d(ConvPoolLayer):
    __arg_max: np.ndarray = None
    __out_grad_col: np.ndarray = None
    __pool_area: int

    def __init__(self, pool_shape, stride=None, padding=(0, 0)):
        super().__init__(pool_shape, pool_shape if stride is None else stride, padding)

    def __first_init(self, input_shape):
        super()._first_init(input_shape, -1)

        output_size = self._batch_size * self._in_channels * self._output_shape[0] * self._output_shape[1]

        self._out_tensor_col.resize(output_size, 1)
        self.__pool_area = self.filter_shape[0] * self.filter_shape[1]
        self.__out_grad_col = np.ndarray((output_size, self.__pool_area), dtype=np.float32)
        self.__arg_max = np.ndarray((output_size, 1), dtype=np.uint16)

    def forward(self, tensor):
        if self._batch_size is None:
            self.__first_init(tensor.shape)

        tensor_col = self._im2col(tensor).reshape(-1, self.__pool_area)

        np.argmax(tensor_col, axis=1, keepdims=True, out=self.__arg_max)
        self._out_tensor_col[:] = np.take_along_axis(tensor_col, self.__arg_max, axis=1)

        return self._out_tensor_col.reshape(self._batch_size, *self._output_shape, self._in_channels).transpose(0, 3, 1, 2)

    def backward(self, grad):
        grad = grad.transpose(0, 2, 3, 1)

        self.__out_grad_col.fill(0)
        np.put_along_axis(self.__out_grad_col, self.__arg_max, grad.reshape(-1, 1), axis=1)
        out_grad = self._col2im(self.__out_grad_col)

        return out_grad

    @staticmethod
    def restore(param_dict):
        cls = MaxPool2d.__new__(MaxPool2d)
        ConvPoolLayer._restore(cls, param_dict)

        return cls

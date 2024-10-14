import numpy as np
import param

# 各层输入tensor的数据不要修改


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
    __mask: np.ndarray

    def forward(self, tensor):
        self.__mask = tensor > 0.

        return tensor * self.__mask

    def backward(self, grad):
        grad[self.__mask] = 0.

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
        self.__out_grad *= self.__out_tensor
        self.__out_grad *= grad

        return self.__out_grad

    @staticmethod
    def restore(param_dict):
        return Sigmoid.__new__(Sigmoid)


class Affine(Layer):
    in_tensor: np.ndarray

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.wight = param.he(in_dims, out_dims)
        self.bias = param.he(out_dims)

    def forward(self, tensor):
        self.in_tensor = tensor

        out_tensor = np.matmul(tensor, self.wight.data)
        out_tensor += self.bias.data

        return out_tensor

    def backward(self, grad: np.ndarray):
        np.mean(grad, axis=0, out=self.bias.grad)
        np.matmul(self.in_tensor.T, grad, out=self.wight.grad)

        grad = np.matmul(grad, self.wight.data.T)
        return grad

    def get_params(self, params_list):
        params_list.append(self.wight)
        params_list.append(self.bias)

    def get_restore_params(self):
        param_dict = dict(wight=self.wight, bias=self.bias)
        return param_dict

    @staticmethod
    def restore(param_dict):
        cls = Affine.__new__(Affine)
        cls.wight = param_dict["wight"]
        cls.bias = param_dict["bias"]

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
    out_shape = None
    original_shape = None

    def __calc_out_shape(self):
        length = len(self.original_shape)
        if self.end_dim == -1:
            self.end_dim = length - 1

        self.out_shape = []
        dim = 0
        dim_size = 1
        while dim < length:
            dim_size *= self.original_shape[dim]
            if dim < self.start_dim or dim >= self.end_dim:
                self.out_shape.append(dim_size)
                dim_size = 1
            dim += 1

        self.out_shape = tuple(self.out_shape)

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, tensor):
        if self.out_shape is None:
            self.original_shape = tensor.shape
            self.__calc_out_shape()

        return tensor.reshape(self.out_shape)

    def backward(self, grad):
        grad.resize(self.original_shape)
        return grad

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
    __out_grad: np.ndarray

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
        param_dict = dict(wight=self.wight, bias=self.bias)
        return param_dict

    @staticmethod
    def restore(param_dict):
        cls = Normalization.__new__(Normalization)
        cls.wight = param_dict["wight"]
        cls.bias = param_dict["bias"]

        return cls


class Dropout(Layer):
    __mask: np.ndarray = None

    def __init__(self, dropout_rate=0.2, train=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.train = train

    def forward(self, tensor):
        if self.train:
            self.__mask = np.random.rand(*tensor.shape) > self.dropout_rate
            return tensor * self.__mask

        return tensor * (1 - self.dropout_rate)

    def backward(self, grad):
        grad[self.__mask] = 0.
        return grad

    def get_restore_params(self):
        param_dict = dict(dropout_rate=self.dropout_rate)
        return param_dict

    @staticmethod
    def restore(param_dict):
        return Dropout(param_dict["dropout_rate"])


class ConvPoolLayer(Layer):
    filter_size: tuple
    stride: tuple
    padding: tuple
    bitch: int = None
    in_channels: int = None
    input_size: tuple[int, int] = None
    output_size: tuple[int, int] = None
    __im2col_static: np.ndarray = None
    __col2mg_static: np.ndarray = None
    __tensor_pad: np.ndarray = None

    def _get_output_size(self):
        return (self.input_size[0] + 2 * self.padding[0] - self.filter_size[0]) // self.stride[0] + 1, \
               (self.input_size[1] + 2 * self.padding[1] - self.filter_size[1]) // self.stride[1] + 1

    def _first_init(self, input_shape):
        bitch, in_channels, input_h, input_w = input_shape
        self.input_size = (input_h, input_w)
        self.bitch = bitch
        self.in_channels = in_channels
        self.output_size = self._get_output_size()
        self.__tensor_pad = np.zeros(
            (bitch, in_channels, input_h + 2 * self.padding[0], input_w + 2 * self.padding[1]), dtype=np.float32)
        self.__im2col_static = np.ndarray((bitch, in_channels, *self.filter_size, *self.output_size), dtype=np.float32)
        self.__col2mg_static = np.ndarray((bitch, in_channels,
                                           input_h + 2 * self.padding[0] + self.stride[0] - 1,
                                           input_w + 2 * self.padding[1] + self.stride[1] - 1), dtype=np.float32)

    def _im2col(self, tensor):
        self.__tensor_pad[:, :, self.padding[0]:self.padding[0] + self.input_size[0],
        self.padding[1]:self.padding[1] + self.input_size[1]] = tensor

        for y in range(self.filter_size[0]):
            y_max = y + self.stride[0] * self.output_size[0]

            for x in range(self.filter_size[1]):
                x_max = x + self.stride[1] * self.output_size[1]
                self.__im2col_static[:, :, y, x, :, :] = self.__tensor_pad[:, :, y:y_max:self.stride[0],
                                                         x:x_max:self.stride[1]]

        return self.__im2col_static.transpose(0, 4, 5, 1, 2, 3).reshape(
            self.bitch * self.output_size[0] * self.output_size[1], -1)

    def _col2im(self, col):
        col = (col.reshape(self.bitch, *self.output_size, self.in_channels, *self.filter_size)
               .transpose(0, 3, 4, 5, 1, 2))

        self.__col2mg_static.fill(0.0)
        for y in range(self.filter_size[0]):
            y_max = y + self.stride[0] * self.output_size[0]

            for x in range(self.filter_size[1]):
                x_max = x + self.stride[1] * self.output_size[1]
                self.__col2mg_static[:, :, y:y_max:self.stride[0], x:x_max:self.stride[1]] += col[:, :, y, x, :, :]

        return (self.__col2mg_static[:, :, self.padding[0]:self.input_size[0] + self.padding[0],
                self.padding[1]:self.input_size[1] + self.padding[1]])

    def __init__(self, filter_size, stride, padding):
        super().__init__()
        self.filter_size = _to_tuple2(filter_size)
        self.stride = _to_tuple2(stride)
        self.padding = _to_tuple2(padding)

    def get_restore_params(self):
        param_dict = dict(filter_size=self.filter_size, stride=self.stride, padding=self.padding,
                          input_shape=(self.bitch, self.in_channels, *self.input_size))
        return param_dict

    def _restore(self, param_dict):
        self.filter_size = param_dict["filter_size"]
        self.stride = param_dict["stride"]
        self.padding = param_dict["padding"]


class Convolution2d(ConvPoolLayer):
    __tensor_col = None
    __kernel_col = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super().__init__(kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel = param.he(out_channels, in_channels, *self.filter_size)
        self.kernel.data /= np.sqrt(self.filter_size[0] * self.filter_size[1])
        self.bias = param.Param(np.zeros(out_channels, dtype=np.float32))

    def forward(self, tensor):
        if self.bitch is None:
            self._first_init(tensor.shape)

        self.__tensor_col = self._im2col(tensor)
        self.__kernel_col = self.kernel.data.reshape(self.out_channels, -1).T

        out_tensor = np.matmul(self.__tensor_col, self.__kernel_col)
        out_tensor += self.bias.data
        out_tensor = out_tensor.reshape(self.bitch, *self.output_size, -1).transpose(0, 3, 1, 2)

        return out_tensor

    def backward(self, grad):
        grad = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        np.mean(grad, axis=0, out=self.bias.grad)

        self.kernel.grad = np.matmul(self.__tensor_col.T, grad).transpose(1, 0)
        self.kernel.grad.resize(self.out_channels, self.in_channels, *self.filter_size)

        out_grad = self._col2im(np.matmul(grad, self.__kernel_col.T))
        return out_grad

    def get_params(self, param_list):
        param_list.append(self.kernel)
        param_list.append(self.bias)

    def get_restore_params(self):
        param_dict = super().get_restore_params()
        param_dict["out_channels"] = self.out_channels
        param_dict["kernel"] = self.kernel
        param_dict["bias"] = self.bias

        return param_dict

    @staticmethod
    def restore(param_dict):
        cls = Convolution2d.__new__(Convolution2d)
        ConvPoolLayer._restore(cls, param_dict)
        cls.out_channels = param_dict["out_channels"]
        cls.kernel = param_dict["kernel"]
        cls.bias = param_dict["bias"]
        cls._first_init(param_dict["input_shape"])

        return cls


class MaxPool2d(ConvPoolLayer):
    grad_col_shape = None
    arg_max = None
    grad_col = None
    pool_area: int

    def __init__(self, pool_size, stride=None, padding=(0, 0)):
        super().__init__(pool_size, pool_size if stride is None else stride, padding)

    def _first_init(self, input_shape):
        super()._first_init(input_shape)

        rows_num = self.bitch * self.in_channels * self.output_size[0]
        out_grad_size = rows_num * self.output_size[1]

        self.pool_area = self.filter_size[0] * self.filter_size[1]
        self.grad_col = np.ndarray(out_grad_size * self.pool_area, dtype=np.float32)
        self.grad_col_shape = (rows_num, self.output_size[1] * self.pool_area)
        self.arg_max = np.ndarray(out_grad_size, dtype=np.uint32)

    def forward(self, tensor):
        if self.bitch is None:
            self._first_init(tensor.shape)

        tensor_col = self._im2col(tensor).reshape(-1, self.pool_area)

        np.argmax(tensor_col, axis=1, out=self.arg_max)

        out_tensor = np.max(tensor_col, axis=1)
        out_tensor.resize(self.bitch, *self.output_size, self.in_channels)
        out_tensor = out_tensor.transpose(0, 3, 1, 2)

        return out_tensor

    def backward(self, grad):
        grad = grad.transpose(0, 2, 3, 1)

        self.grad_col.fill(0.0)
        self.grad_col.resize(grad.size, self.pool_area)
        self.grad_col[np.arange(self.arg_max.size), self.arg_max.flatten()] = grad.flatten()

        self.grad_col.resize(*self.grad_col_shape)
        out_grad = self._col2im(self.grad_col)

        return out_grad

    @staticmethod
    def restore(param_dict):
        cls = MaxPool2d.__new__(MaxPool2d)
        ConvPoolLayer._restore(cls, param_dict)
        cls._first_init(param_dict["input_shape"])

        return cls

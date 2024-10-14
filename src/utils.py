import numpy as np


def to_tuple2(arg):
    arg_type = type(arg)
    if arg_type is int:
        return arg, arg
    elif arg_type is tuple and len(arg) == 2:
        return arg
    raise ValueError("Unsupported argument type, only int and tuple[int, int] are allowed")


def conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1, \
           (input_size[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1


def im2col(tensor, col, kernel_size, stride, padding):
    n, c, h, w = tensor.shape
    out_h, out_w = conv_output_size((h, w), kernel_size, stride, padding)

    img = np.pad(tensor, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))

    for y in range(kernel_size[0]):
        y_max = y + stride[0] * out_h
        for x in range(kernel_size[1]):
            x_max = x + stride[1] * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride[0], x:x_max:stride[1]]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)


def col2im(col, col_img, input_size, kernel_size, stride, padding):
    h, w = input_size
    n, c = col_img.shape[0:2]
    out_h, out_w = conv_output_size((h, w), kernel_size, stride, padding)
    col = col.reshape(n, out_h, out_w, c, *kernel_size).transpose(0, 3, 4, 5, 1, 2)

    col_img.fill(0)
    for y in range(kernel_size[0]):
        y_max = y + stride[0] * out_h
        for x in range(kernel_size[1]):
            x_max = x + stride[1] * out_w
            col_img[:, :, y:y_max:stride[0], x:x_max:stride[1]] += col[:, :, y, x, :, :]

    return col_img[:, :, padding[0]:h + padding[0], padding[1]:w + padding[1]]

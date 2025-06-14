import numpy as np
import cv2
import random
from . import datatype

TYPE_NP_TO_CV = {
    np.dtype('uint8'): cv2.CV_8U,
    np.dtype('int8'): cv2.CV_8S,
    np.dtype('uint16'): cv2.CV_16U,
    np.dtype('int16'): cv2.CV_16S,
    np.dtype('int32'): cv2.CV_32S,
    np.dtype('float32'): cv2.CV_32F,
    np.dtype('float64'): cv2.CV_64F,
}

TYPE_CV_TO_NP = {
    cv2.CV_8U: np.uint8,
    cv2.CV_8S: np.int8,
    cv2.CV_16U: np.uint16,
    cv2.CV_16S: np.int16,
    cv2.CV_32S: np.int32,
    cv2.CV_32F: np.float32,
    cv2.CV_64F: np.float64,
}


def _random_int_range(param, depth=0):
    if depth == 0:
        if isinstance(param, (tuple, list)):
            return random.randint(*param)
        return param
    return tuple(_random_int_range(p, depth - 1) for p in param)


def _random_float_range(param, depth=0):
    if depth == 0:
        if isinstance(param, (tuple, list)):
            return random.uniform(*param)
        return param
    return tuple(_random_float_range(p, depth - 1) for p in param)


def cut(image, rect: datatype.Rect):
    tf = datatype.Affine(2).translate(-rect.x, -rect.y)
    return image[rect.slice()], tf


def letterbox(src, dst_shape):
    width, height = src.shape[1], src.shape[0]
    scale = min(dst_shape[0] / width, dst_shape[1] / height)
    new_width, new_height = int(width * scale), int(height * scale)

    if new_width % 2 == 1:
        new_width += 1
    if new_height % 2 == 1:
        new_height += 1

    padding_x = (dst_shape[0] - new_width) // 2
    padding_y = (dst_shape[1] - new_height) // 2
    dst = cv2.resize(src, (new_width, new_height))
    dst = cv2.copyMakeBorder(dst, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT)

    tf = datatype.Affine(2).scale(scale).translate(padding_x, padding_y)
    return dst, tf


# 绕 center(默认图片中心) 放缩 + 旋转(角度；逆时针) -> 平移
# scale默认同时作用于 x, y；若scale_y is not None，则 scale, scale_y 分别作用于 x, y
def affine(src, scale=1., rotate=0., translate=(0, 0), scale_y=None, center=None, dst_shape=None):
    h, w = src.shape[0], src.shape[1]

    translate = _random_int_range(translate, depth=1)
    scale_x, scale_y, rotate = _random_float_range((scale, scale_y, rotate), depth=1)
    if scale_y is None:
        scale_y = scale_x
    if center is None:
        center = (w // 2, h // 2)
    if dst_shape is None:
        dst_shape = (int(w * scale_x), int(h * scale_y))

    tf = (datatype.Affine(2)
          .translate(-center[0], -center[1])
          .scale(scale_x, scale_y)
          .rotate(-rotate)
          .translate(center[0] * scale_x + translate[0], center[1] * scale_y + translate[1]))

    dst = cv2.warpAffine(src, tf.get_matrix(), dst_shape)
    return dst, tf


def add_noise(src, mean, std=0.):
    mean, std = _random_float_range((mean, std), depth=1)
    if std == 0.:
        return cv2.add(src, mean)

    n = np.random.normal(mean, std, src.shape)
    return cv2.add(src, n, dtype=TYPE_NP_TO_CV[src.dtype])

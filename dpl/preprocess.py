import numpy as np
import cv2
import random

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


def cut(image, top=0, bottom=0, left=0, right=0):
    w, h = image.shape[1], image.shape[0]

    if isinstance(top, tuple):
        top = random.randint(*top)
    if isinstance(bottom, tuple):
        bottom = random.randint(*bottom)
    if isinstance(left, tuple):
        left = random.randint(*left)
    if isinstance(right, tuple):
        right = random.randint(*right)

    m = np.array(((1., 0., -left),
                  (0., 1., -top),
                  (0., 0., 1.)), dtype=np.float32)
    return image[top: h - bottom, left: w - right], m


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

    m = np.array(((scale, 0, padding_x),
                  (0, scale, padding_y),
                  (0, 0, 1.)), dtype=np.float32)
    return dst, m


# 绕 图像中心 放缩 + 旋转(角度) -> 平移
def affine(src, scale=1., rotate=0., translate=(0, 0), dst_shape=None):
    h, w = src.shape[0], src.shape[1]

    if isinstance(scale, tuple):
        scale = random.uniform(*scale)
    if isinstance(rotate, tuple):
        rotate = random.uniform(*rotate)

    m = cv2.getRotationMatrix2D((w // 2, h // 2), rotate, scale)  # (2 x 3)

    if isinstance(translate[0], tuple):
        m[0, 2] += random.randint(*translate[0])
        m[1, 2] += random.randint(*translate[1])
    else:
        m[0, 2] += translate[0]
        m[1, 2] += translate[1]
    if dst_shape is None:
        dst_shape = (w, h)

    dst = cv2.warpAffine(src, m, dst_shape)
    m = np.vstack((m, (0, 0, 1)))
    return dst, m.astype(np.float32)


def add_noise(src, mean, std=0.):
    if isinstance(mean, tuple):
        mean = random.uniform(*mean)
    if std == 0.:
        return cv2.add(src, mean)

    n = np.random.normal(mean, std, src.shape)
    return cv2.add(src, n, dtype=TYPE_NP_TO_CV[src.dtype])


# pts: (N, 2)
def transform(pts, matrix, out=None, ):
    w = matrix[:2, :2].T
    b = matrix[:2, 2].T
    if out is None:
        pts = pts @ w
        pts += b
        return pts

    out[:] = pts @ w
    out += b
    return out


# return -> (top_left_x, top_left_y, width, height)
def bounding_box(pts):
    x = pts[:, 0].min()
    y = pts[:, 1].min()
    width = pts[:, 0].max() - x
    height = pts[:, 1].max() - y
    return x, y, width, height

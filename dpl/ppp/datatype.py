import numpy as np


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def top_left(self):
        return self.x, self.y

    def bottom_right(self):
        return self.x + self.w, self.y + self.h

    def size(self):
        return self.w, self.h

    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def slice(self):
        return slice(int(self.y), int(self.y + self.h)), slice(int(self.x), int(self.x + self.w))

    def __and__(self, other):
        tl_x = max(self.x, other.x)
        tl_y = max(self.y, other.y)
        br_x = min(self.x + self.w, other.x + other.w)
        br_y = min(self.y + self.h, other.y + other.h)
        return Rect(tl_x, tl_y, max(br_x - tl_x, 0), max(br_y - tl_y, 0))

    def __or__(self, other):
        tl_x = min(self.x, other.x)
        tl_y = min(self.y, other.y)
        br_x = max(self.x + self.w, other.x + other.w)
        br_y = max(self.y + self.h, other.y + other.h)
        return Rect(tl_x, tl_y, br_x - tl_x, br_y - tl_y)


def bounding_box(pts: np.ndarray, col_major=False):
    x_coords = pts[0] if col_major else pts[:, 0]
    y_coords = pts[1] if col_major else pts[:, 1]

    x = x_coords.min()
    y = y_coords.min()
    width = x_coords.max() - x
    height = y_coords.max() - y
    return Rect(x, y, width, height)


class Affine:
    def __init__(self, dims):
        self.dims = dims

        self._m = np.eye(dims, dims + 1, dtype=np.float32)
        self._linear = self._m[:, :dims]  # rotate + scale
        self._translate = self._m[:, dims].reshape(dims, 1)

    def from_matrix(self, matrix: np.ndarray):
        self._m[:] = matrix[:self.dims, :]
        return self

    def get_matrix(self):
        return self._m

    def transform(self, pts: np.ndarray, col_major=False):
        if col_major:
            return self._linear @ pts + self._translate
        return pts @ self._linear.T + self._translate.T

    def transform_rect(self, rect: Rect):
        left = rect.x
        right = rect.x + rect.w
        top = rect.y
        bottom = rect.y + rect.h
        corners = np.array((
            (left, left, right, right),
            (top, bottom, bottom, top)), dtype=np.float32)
        transformed_corners = self.transform(corners, True)
        return bounding_box(transformed_corners, True)

    def scale(self, *scale):
        self._m *= np.array(scale, dtype=np.float32).reshape(-1, 1)
        return self

    def rotate(self, rad, axis=None):
        cos = np.cos(rad)
        sin = np.sin(rad)

        if self.dims == 2:
            rotation = np.array((
                (cos, -sin),
                (sin, cos)), dtype=np.float32)
        else:  # 3d
            axis = np.array(axis, dtype=np.float32).flatten()
            axis /= np.linalg.norm(axis)
            k = np.array((
                (0, -axis[2], axis[1]),
                (axis[2], 0, -axis[0]),
                (-axis[1], axis[0], 0)), dtype=np.float32)
            i = np.eye(3, dtype=np.float32)
            rotation = i + k @ (i * sin + k * (1 - cos))

        self._m[:] = rotation @ self._m
        return self

    def translate(self, *t):
        self._translate += np.array(t, dtype=np.float32).reshape(self.dims, 1)
        return self

    def link(self, other):
        m2 = other.get_matrix()
        self._m[:] = m2[:, :self.dims] @ self._m
        self._translate += m2[:, self.dims].reshape(self.dims, 1)

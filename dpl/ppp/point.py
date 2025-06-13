import numpy as np


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
        return self._m.copy()

    def transform(self, pts: np.ndarray, col_major=False):
        if col_major:
            return self._linear @ pts + self._translate
        return pts @ self._linear.T + self._translate.T

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


# return -> (top_left_x, top_left_y, width, height)
def bounding_box(pts: np.ndarray):
    x = pts[:, 0].min()
    y = pts[:, 1].min()
    width = pts[:, 0].max() - x
    height = pts[:, 1].max() - y
    return x, y, width, height

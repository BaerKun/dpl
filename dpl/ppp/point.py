import numpy as np


class Affine:
    def __init__(self, dims):
        self.dims = dims
        self.m = np.eye(dims, dims + 1, dtype=np.float32)

        self.linear = self.m[:, :dims]  # rotate + scale
        self.translation = self.m[:, dims].reshape(dims, 1)

    def from_matrix(self, matrix: np.ndarray):
        self.m[:] = matrix[:self.dims, :]
        return self

    def to_matrix(self):
        return self.m

    def transform(self, pts: np.ndarray, col_major=False):
        if col_major:
            return self.linear @ pts + self.translation
        return pts @ self.linear.T + self.translation.T

    def scale(self, *scale):
        self.m *= np.array(scale, dtype=np.float32).reshape(-1, 1)
        return self

    def rotate(self, rad, axis=None):
        if self.dims == 2:
            rotation = np.array(((np.cos(rad), -np.sin(rad)), (np.sin(rad), np.cos(rad))), dtype=np.float32)
            self.m[:] = rotation @ self.m
            return self

        # 3d
        return self

    def translate(self, *t):
        self.translation += np.array(t, dtype=np.float32).reshape(self.dims, 1)
        return self


# return -> (top_left_x, top_left_y, width, height)
def bounding_box(pts: np.ndarray):
    x = pts[:, 0].min()
    y = pts[:, 1].min()
    width = pts[:, 0].max() - x
    height = pts[:, 1].max() - y
    return x, y, width, height

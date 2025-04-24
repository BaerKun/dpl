import numpy as np


class KeyPoints:
    def __init__(self, coords, dtype=np.float32):
        self.np = np.array(coords, dtype=dtype).reshape(-1, 2)

    def transform(self, matrix):
        w = matrix[:2, :2].T
        b = matrix[:2, 2:].T
        out = self.np @ w
        out += b
        return out

    def transform_local(self, matrix):
        w = matrix[:2, :2].T
        b = matrix[:2, 2:].T
        self.np = self.np @ w
        self.np += b

    # return -> (top_left_x, top_left_y, width, height)
    def bounding_box(self):
        x = self.np[:, 0].min()
        y = self.np[:, 1].min()
        width = self.np[:, 0].max() - x
        height = self.np[:, 1].max() - y
        return x, y, width, height

    def __iter__(self):
        return self.np.__iter__()

    def __getitem__(self, item):
        return self.np[item]

    def __len__(self):
        return len(self.np)

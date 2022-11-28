import numpy as np


class Interpolator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idxy = None
        self.idxx = None
        self.grid = None
        self.weights = None

    def make_grid(self, point):
        idxx = np.where(self.x == self.x[self.x <= point[0]].max())[0].item()
        idxy = np.where(self.y == self.y[self.y <= point[1]].max())[0].item()
        self.idxx = idxx
        self.idxy = idxy
        grid = np.array([self.x[idxx:idxx + 2], self.y[idxy:idxy + 2]])
        self.grid = grid
        return grid

    def get_bilinear_weights(self, point):
        grid = self.make_grid(point)
        y2 = grid[..., 1, 1]
        y1 = grid[..., 1, 0]
        x2 = grid[..., 0, 1]
        x1 = grid[..., 0, 0]
        x = point[0]
        y = point[1]
        dx = x2 - x1
        dy = y2 - y1
        c1 = (x2 - x) * (y2 - y) / dx / dy
        c2 = (x - x1) * (y2 - y) / dx / dy
        c3 = (x2 - x) * (y - y1) / dx / dy
        c4 = (x - x1) * (y - y1) / dx / dy
        self.weights = [c1, c2, c3, c4]
        return c1, c2, c3, c4

    def interpolate(self, grid_data, coefs=None):
        f11 = grid_data[..., 0, 0]
        f21 = grid_data[..., 0, 1]
        f12 = grid_data[..., 1, 0]
        f22 = grid_data[..., 1, 1]
        if coefs:
            return f11 * coefs[0] + f21 * coefs[1] + f12 * coefs[2] + f22 * coefs[3]
        else:
            return f11 * self.weights[0] + f21 * self.weights[1] + f12 * self.weights[2] + f22 * self.weights[3]

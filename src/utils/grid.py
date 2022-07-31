import numpy as np


def create_grid(grid_size: int):
    x_range = y_range = z_range = np.linspace(-1.0, 1.0, grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

    return grid_x, grid_y, grid_z

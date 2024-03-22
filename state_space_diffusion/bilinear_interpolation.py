import torch

def bilinear_interpolation(features: torch.Tensor, xy: torch.Tensor, grid_square_size: float, max_grid_square_inclusive: int):
    x = xy[..., 0]
    y = xy[..., 1]

    lower_ind = torch.floor((xy - (grid_square_size / 2)) / grid_square_size).to(torch.long)
    lower_ind[lower_ind < 0] = 0
    lower_ind[lower_ind > (max_grid_square_inclusive - 1)] = max_grid_square_inclusive - 1
    lower_x_ind = lower_ind[..., 0]
    lower_y_ind = lower_ind[..., 1]
    upper_x_ind = lower_x_ind + 1
    upper_y_ind = lower_y_ind + 1
    lower_x = (lower_x_ind + 0.5) * grid_square_size
    lower_y = (lower_y_ind + 0.5) * grid_square_size
    upper_x = (upper_x_ind + 0.5) * grid_square_size
    upper_y = (upper_y_ind + 0.5) * grid_square_size

    # interpolated_value = 1/((upper_x - lower_x) * (upper_y - lower_y)) * (
    interpolated_value = 1/(grid_square_size ** 2) * (
        features[lower_x_ind, lower_y_ind] * (upper_x - x) * (upper_y - y) +
        features[upper_x_ind, lower_y_ind] * (x - lower_x) * (upper_y - y) +
        features[lower_x_ind, upper_y_ind] * (upper_x - x) * (y - lower_y) +
        features[upper_x_ind, upper_y_ind] * (x - lower_x) * (y - lower_y)
    )

    return interpolated_value, (lower_x_ind, lower_y_ind)
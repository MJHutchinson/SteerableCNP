import torch

from einops import rearrange


def grid_2d(min_x, max_x, n_xaxis, min_y=None, max_y=None, n_yaxis=None, flatten=True):
    """
    Input:
        min_x,max_x,min_y,max_y: float - range of x-axis/y-axis
        n_x_axis,n_y_axis: int - number of points per axis
        flatten: Boolean - determines shape of output
    Output:
        torch.tensor - if flatten is True: shape (n_y_axis*n_x_axis,2)
                                          (element i*n_x_axis+j gives i-th element in y-grid
                                           and j-th element in  x-grid.
                                           In other words: x is periodic counter and y the stable counter)
                       if flatten is not True: shape (n_y_axis,n_x_axis,2)
    """
    if min_y is None:
        min_y = min_x
    if max_y is None:
        max_y = max_x
    if n_yaxis is None:
        n_yaxis = n_xaxis

    x = torch.linspace(min_x, max_x, n_xaxis)
    y = torch.linspace(min_y, max_y, n_yaxis)
    # TODO: reorder the grid - has repercussions later on
    Y, X = torch.meshgrid(y, x)
    grid = torch.stack((X, Y), 2)

    if flatten:
        grid = rearrange(grid, "x y d -> (x y) d")

    return grid


def grid_3d(
    min_x,
    max_x,
    n_xaxis,
    min_y=None,
    max_y=None,
    min_z=None,
    max_z=None,
    n_yaxis=None,
    n_zaxis=None,
    flatten=True,
):
    if min_y is None:
        min_y = min_x
    if max_y is None:
        max_y = max_x
    if n_yaxis is None:
        n_yaxis = n_xaxis
    if min_z is None:
        min_z = min_x
    if max_z is None:
        max_z = max_x
    if n_zaxis is None:
        n_zaxis = n_xaxis

    x = torch.linspace(min_x, max_x, n_xaxis)
    y = torch.linspace(min_y, max_y, n_yaxis)
    z = torch.linspace(min_z, max_z, n_zaxis)
    # TODO: reorder the grid - has repercussions later on
    X, Y, Z = torch.meshgrid(x, y, z)
    grid = torch.stack((X, Y, Z), -1)

    if flatten:
        grid = rearrange(grid, "x y z d -> (x y z) d")

    return grid


def radial_grid_2d(max_r, n_axis):
    """
    Input:
        min_r: float - maximum radius from origin
        n_axis: float - number of points across the x diameter
    Output:
        torch.tensor - if flatten is True: shape (n_y_axis*n_x_axis,2)
    """
    grid = grid_2d(min_x=-max_r, max_x=max_r, n_xaxis=n_axis, flatten=True)

    in_radius_indices = grid.norm(dim=-1) <= max_r

    grid = grid[in_radius_indices]

    return grid


def expand_with_ones(Y):
    """
    Input: Y - torch.Tensor - shape (batch_size,n,C)
    Output: torch.Tensor -shape (batch_size,n,C+1) - added a column of ones to Y (at the start) Y[i,j]<--[1,Y[i,j]]
    """
    return torch.cat([torch.ones([Y.size(0), Y.size(1), 1], device=Y.device), Y], dim=2)

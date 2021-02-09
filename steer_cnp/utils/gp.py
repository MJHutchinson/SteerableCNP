import torch

from einops import rearrange

from steer_cnp.utils.grid import grid_2d, radial_grid_2d
from steer_cnp.gp import sample_gp_prior


def sample_gp_grid_2d(kernel, samples=1, min_x=-2, max_x=2, n_axis=10, obs_noise=1e-4):
    """
    Input:
    kernel: steer_cnp.kernel.Kernel
    min_x,max_x: scalar - left-right/lower-upper limit of grid
    n_grid_points: int - number of grid points per axis
    Output:
    X: torch.tensor
       Shape (n_grid_points**2,2)
    Y: torch.tensor
       Shape (n_grid_points**2,D), D...dimension of label space
    """

    X = grid_2d(min_x, max_x, n_axis, flatten=True)
    Y = sample_gp_prior(X, kernel, samples=samples, obs_noise=obs_noise)

    if samples != 1:
        X = X.unsqueeze(0).repeat_interleave(samples, 0)

    #   bs, n, d = X.shape

    #   batch_inds = torch.arange(bs)[:, None, None]
    #   perm_inds = torch.randn(X.shape).argsort(dim=1)
    #   d_inds = torch.arange(d)[None, None, :]

    #   X = X[batch_inds, perm_inds, d_inds]
    #   Y = Y[batch_inds, perm_inds, d_inds]

    return X, Y


def sample_gp_radial_grid_2d(kernel, samples=1, max_r=2, n_axis=10, obs_noise=1e-4):
    """
    Input:
    kernel: steer_cnp.kernel.Kernel
    max_r: scalar - radius of the grid
    n_grid_points: int - number of grid points per axis
    Output:
    X: torch.tensor
       Shape (n_grid_points**2,2)
    Y: torch.tensor
       Shape (n_grid_points**2,D), D...dimension of label space
    """

    X = radial_grid_2d(max_r, n_axis)
    Y = sample_gp_prior(X, kernel, samples=samples, obs_noise=obs_noise)

    if samples != 1:
        X = X.unsqueeze(0).repeat_interleave(samples, 0)

    # bs, n, d = X.shape

    # batch_inds = torch.arange(bs)[:, None, None]
    # perm_inds = torch.randn(X.shape).argsort(dim=1)
    # d_inds = torch.arange(d)[None, None, :]

    # X = X[batch_inds, perm_inds, d_inds]
    # Y = Y[batch_inds, perm_inds, d_inds]

    return X, Y

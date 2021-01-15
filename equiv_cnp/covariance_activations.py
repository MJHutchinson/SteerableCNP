import math

import torch
import torch.nn.functional as F

from einops import rearrange


def quadratic_covariance_activation(pre_activation, epsilon=1e-6):
    """Computes the quadratic covariance activation function

    Parameters
    ----------
    pre_activation : torch.Tensor
        A (bs, n, sigma_dim**2) tensor
    epsilon : Numerical tolerance parameter to add, optional, by default 1e-6
    """

    _, _, d2 = pre_activation.shape
    d = int(math.sqrt(d2))

    pre_activation = rearrange(pre_activation, "b n (d1 d2) -> b n d2 d1", d1=d, d2=d)

    covariance = pre_activation @ pre_activation.transpose(-1, -2)

    # covariance = covariance + (torch.eye(d) * epsilon)

    return rearrange(covariance, "b n d2 d1 -> b n (d1 d2)")


def diagonal_quadratic_covariance_activation(pre_activation):
    b, n, _ = pre_activation.shape
    return torch.diag_embed(pre_activation.pow(2)).view(b, n, -1)


def diagonal_softplus_covariance_activation(pre_activation, min_var=0.1):
    b, n, _ = pre_activation.shape
    return torch.diag_embed(min_var + F.softplus(pre_activation)).view(b, n, -1)


def diagonal_quadratic_softplus_covariance_activation(pre_activation, min_sigma=0.1):
    b, n, _ = pre_activation.shape
    return torch.diag_embed((min_sigma + F.softplus(pre_activation)).pow(2)).view(
        b, n, -1
    )

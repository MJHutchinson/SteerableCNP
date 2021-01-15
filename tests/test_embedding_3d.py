# %%
import torch

from equiv_cnp.kernel import (
    RBFKernel,
    SeparableKernel,
    DotProductKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
    kernel_smooth,
)

from equiv_cnp.rkhs_embedding import DiscretisedRKHSEmbedding

from equiv_cnp.utils import expand_with_ones, plot_embedding, plot_vector_field

import matplotlib.pyplot as plt

# %%
rbf = SeparableKernel(3, 4, RBFKernel(3, 3.0))

grid_range = [-4, 4]
n_axes = 33

embedder = DiscretisedRKHSEmbedding(
    grid_range, n_axes, dim=3, kernel=rbf, normalise=True
)


X_grid = embedder.grid

X_context = torch.Tensor([[1, 2, 1], [2, 1, 3], [-1, -1, -2]]).unsqueeze(0)
Y_context = torch.Tensor([[1, 1, 1], [1, -2, 3], [-4, 3, -1]]).unsqueeze(0)

X_grid, Y_grid = embedder(X_context, Y_context)
# %%

for x3 in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:

    plot_vector_field(
        X_grid[X_grid[:, :, 2] == x3][:, :2].squeeze(),
        Y_grid[X_grid[:, :, 2] == x3][:, :2].squeeze(),
    )
    plt.title(x3)
# %%

X_context = torch.randn((10, 100, 2))
Y_context = torch.randn((10, 100, 2))

Y_grid = embedder(X_context, Y_context)
# %%

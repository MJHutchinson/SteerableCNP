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

from equiv_cnp.utils import expand_with_ones, plot_embedding

# %%
rbf = SeparableKernel(2, 3, RBFKernel(2, 3.0))

grid_range = [-4, 4]
n_axes = 20

embedder = DiscretisedRKHSEmbedding(
    grid_range, n_axes, dim=2, kernel=rbf, normalise=True
)


X_grid = embedder.grid

X_context = torch.Tensor([[1, 2], [2, 1], [-1, -1]]).unsqueeze(0)
Y_context = torch.Tensor([[1, 1], [1, -2], [-4, 3]]).unsqueeze(0)

Y_grid = embedder(X_context, Y_context)
# %%

plot_embedding(
    X_context.squeeze(), Y_context.squeeze(), X_grid.squeeze(), Y_grid.squeeze()
)
# %%

X_context = torch.randn((10, 100, 2))
Y_context = torch.randn((10, 100, 2))

Y_grid = embedder(X_context, Y_context)
# %%

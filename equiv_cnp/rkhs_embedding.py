import torch
import torch.nn as nn

from equiv_cnp.kernel import kernel_smooth
from equiv_cnp.utils import grid_2d, grid_3d, expand_with_ones


class DiscretisedRKHSEmbedding(nn.Module):
    def __init__(self, grid_ranges, n_axes, dim, kernel, normalise):
        super().__init__()

        assert dim == kernel.input_dim

        self.kernel = kernel
        self.dim = dim
        self.normalise = normalise
        self.set_grid(grid_ranges, n_axes)

    def set_grid(self, grid_ranges, n_axes):
        # if passed one range, expand the grid ranges to cover all dims
        if not isinstance(grid_ranges[0], list):
            grid_ranges = [grid_ranges] * self.dim

        if not isinstance(n_axes, list):
            n_axes = [n_axes] * self.dim

        assert len(grid_ranges) == self.dim
        assert len(n_axes) == self.dim

        self.grid_ranges = grid_ranges
        self.n_axes = n_axes

        if self.dim == 2:
            self.register_buffer(
                "grid",
                grid_2d(
                    min_x=grid_ranges[0][0],
                    max_x=grid_ranges[0][1],
                    min_y=grid_ranges[1][0],
                    max_y=grid_ranges[1][1],
                    n_xaxis=n_axes[0],
                    n_yaxis=n_axes[1],
                    flatten=True,
                ),
                persistent=True,
            )
        elif self.dim == 3:
            self.register_buffer(
                "grid",
                grid_3d(
                    min_x=grid_ranges[0][0],
                    max_x=grid_ranges[0][1],
                    min_y=grid_ranges[1][0],
                    max_y=grid_ranges[1][1],
                    min_z=grid_ranges[1][0],
                    max_z=grid_ranges[1][1],
                    n_xaxis=n_axes[0],
                    n_yaxis=n_axes[1],
                    n_zaxis=n_axes[1],
                    flatten=True,
                ),
                persistent=True,
            )
        else:
            raise NotImplementedError(f"Not implemented for dimension {self.dim}")

    def forward(self, X, Y):
        # make sure grid is on the right device.
        self.grid = self.grid.to(X.device)

        bs, n, _ = X.shape

        Y = expand_with_ones(Y)

        # print("RKHS: ", X.device, Y.device, self.grid.device)

        Y_grid = kernel_smooth(X, Y, self.grid.unsqueeze(0), self.kernel)

        if self.normalise:
            # Annoying method to avoid in place operation errors in backward pass
            Y_features = torch.zeros_like(Y_grid[:, :, 1:])
            Y_features = Y_grid[:, :, 1:] / (Y_grid[:, :, 0].unsqueeze(-1) + 1e-6)

            Y_grid = torch.cat([Y_grid[:, :, 0].unsqueeze(-1), Y_features], dim=-1)

        return self.grid.unsqueeze(0), Y_grid

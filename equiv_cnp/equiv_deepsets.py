import torch.nn as nn

from einops import rearrange

from equiv_cnp.rkhs_embedding import DiscretisedRKHSEmbedding
from equiv_cnp.kernel import kernel_smooth
from equiv_cnp.utils import Expression, Pass


class EquivDeepSet(nn.Module):
    def __init__(
        self,
        grid_ranges,
        n_axes,
        embedding_kernel,
        normalise_embedding,
        normalise_output,
        cnn,
        output_kernel,
        dim,
    ):
        super().__init__()

        self.dim = dim

        self.cnn = cnn
        self.output_kernel = output_kernel
        self.normalise_output = normalise_output

        self.discrete_rkhs_embedder = DiscretisedRKHSEmbedding(
            grid_ranges, n_axes, dim, embedding_kernel, normalise_embedding
        )

        self.encoder = nn.Sequential(
            Expression(lambda inpt: self.discrete_rkhs_embedder(*inpt)),
        )

        if dim == 2:
            self.stack_to_grid = lambda Y: rearrange(
                Y,
                "b (m1 m2) d -> b d m2 m1",
                m1=self.discrete_rkhs_embedder.n_axes[0],
                m2=self.discrete_rkhs_embedder.n_axes[1],
            )
            self.grid_to_stack = lambda Y: rearrange(Y, "b d m2 m1 -> b (m1 m2) d")
        elif dim == 3:
            self.stack_to_grid = lambda Y: rearrange(
                Y,
                "b (m1 m2 m3) d -> b m1 m2 m3 d",
                m1=self.discrete_rkhs_embedder.n_axes[0],
                m2=self.discrete_rkhs_embedder.n_axes[1],
                m3=self.discrete_rkhs_embedder.n_axes[2],
            )
            self.grid_to_stack = lambda Y: rearrange(
                Y, "b m1 m2 m3 d -> b (m1 m2 m3) d"
            )
        else:
            raise NotImplementedError(f"Not implemented for dim = {dim}")

    def encode(self, X_context, Y_context):
        # Embed the context set into a discrete gridded RKHS
        return self.discrete_rkhs_embedder(X_context, Y_context)

    def decode(self, X_grid, Y_grid, X_target):
        # reshape Y_grid to go through the CNN
        Y_grid = self.stack_to_grid(Y_grid)
        # pass Y's through the CNN
        Y_grid = self.cnn(Y_grid)
        # reshape Y's back to a stack
        Y_grid = self.grid_to_stack(Y_grid)
        # kernel smooth the outputs to the target points
        return kernel_smooth(
            X_grid, Y_grid, X_target, self.output_kernel, self.normalise_output
        )

    def forward(self, X_context, Y_context, X_target):
        return self.decode(*self.encode(X_context, Y_context), X_target)


class OnGridEquivDeepSet(nn.Module):
    def __init__(
        self,
        grid_ranges,
        n_axes,
        embedding_kernel,
        normalise_embedding,
        normalise_output,
        cnn,
        output_kernel,
        dim,
    ):
        super().__init__()

        self.dim = dim

        self.cnn = cnn
        self.output_kernel = output_kernel
        self.normalise_output = normalise_output

        self.discrete_rkhs_embedder = DiscretisedRKHSEmbedding(
            grid_ranges, n_axes, dim, embedding_kernel, normalise_embedding
        )

        self.encoder = nn.Sequential(
            Expression(lambda inpt: self.discrete_rkhs_embedder(*inpt)),
        )

        if dim == 2:
            self.stack_to_grid = lambda Y: rearrange(
                Y,
                "b (m1 m2) d -> b d m2 m1",
                m1=self.discrete_rkhs_embedder.n_axes[0],
                m2=self.discrete_rkhs_embedder.n_axes[1],
            )
            self.grid_to_stack = lambda Y: rearrange(Y, "b d m2 m1 -> b (m1 m2) d")
        elif dim == 3:
            self.stack_to_grid = lambda Y: rearrange(
                Y,
                "b (m1 m2 m3) d -> b m1 m2 m3 d",
                m1=self.discrete_rkhs_embedder.n_axes[0],
                m2=self.discrete_rkhs_embedder.n_axes[1],
                m3=self.discrete_rkhs_embedder.n_axes[2],
            )
            self.grid_to_stack = lambda Y: rearrange(
                Y, "b m1 m2 m3 d -> b (m1 m2 m3) d"
            )
        else:
            raise NotImplementedError(f"Not implemented for dim = {dim}")

    def encode(self, X_context, Y_context):
        # Embed the context set into a discrete gridded RKHS
        return self.discrete_rkhs_embedder(X_context, Y_context)

    def decode(self, X_grid, Y_grid, X_target):
        # reshape Y_grid to go through the CNN
        Y_grid = self.stack_to_grid(Y_grid)
        # pass Y's through the CNN
        Y_grid = self.cnn(Y_grid)
        # reshape Y's back to a stack
        Y_grid = self.grid_to_stack(Y_grid)
        # kernel smooth the outputs to the target points
        return kernel_smooth(
            X_grid, Y_grid, X_target, self.output_kernel, self.normalise_output
        )

    def forward(self, X_context, Y_context, X_target):
        return self.decode(*self.encode(X_context, Y_context), X_target)
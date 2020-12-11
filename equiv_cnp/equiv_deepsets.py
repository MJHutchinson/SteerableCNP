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

        if dim == 2:
            self.discrete_rkhs_embedder = DiscretisedRKHSEmbedding(
                grid_ranges, n_axes, dim, embedding_kernel, normalise_embedding
            )
            self.n_axes = self.discrete_rkhs_embedder.n_axes

            self.encoder = nn.Sequential(
                Expression(lambda inpt: self.discrete_rkhs_embedder(*inpt)),
            )

            self.decoder = nn.Sequential(
                Pass(
                    Expression(
                        lambda Y: rearrange(
                            Y,
                            "b (m1 m2) d -> b d m2 m1",
                            m1=self.n_axes[0],
                            m2=self.n_axes[1],
                        )
                    ),
                    dim=1,
                ),  # Reshape data to a grid for applying CNN
                Pass(self.cnn, dim=1),  # apply CNN to the Y embedding
                Pass(
                    Expression(lambda Y: rearrange(Y, "b d m2 m1 -> b (m1 m2) d")),
                    dim=1,
                ),  # reshape Y predictions back from grid
                Expression(
                    lambda inpt: kernel_smooth(
                        *inpt, self.output_kernel, normalise=self.normalise_output
                    )
                ),  # smooth the outputs to the target set
            )
        else:
            raise NotImplementedError(f"Not implemented for dim = {dim}")

    def forward(self, X_context, Y_context, X_target):
        return self.decoder((*self.encoder((X_context, Y_context)), X_target))
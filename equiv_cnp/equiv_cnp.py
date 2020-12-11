import torch
import torch.nn as nn

from einops import rearrange

from equiv_cnp.equiv_deepsets import EquivDeepSet
from equiv_cnp.kernel import SeparableKernel, RBFKernelReparametrised, kernel_smooth
from equiv_cnp.utils import Expression, Pass


class EquivCNP(EquivDeepSet):
    def __init__(
        self,
        prediction_dim,
        covariance_activation_function,
        **kwargs,
    ):

        super(EquivCNP, self).__init__(**kwargs)

        self.prediction_dim = prediction_dim
        self.covariance_activation_function = covariance_activation_function

        if self.dim == 2:
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
                Pass(
                    Expression(
                        lambda Y: torch.cat(
                            [
                                Y[:, :, : self.prediction_dim],
                                self.covariance_activation_function(
                                    Y[:, :, self.prediction_dim :]
                                ),
                            ],
                            dim=2,
                        )
                    ),
                    dim=1,
                ),  # apply covariance activation function
                Expression(
                    lambda inpt: kernel_smooth(
                        *inpt,
                        self.output_kernel,
                        normalise=self.normalise_output,
                    )
                ),  # smooth the outputs to the target set
                Expression(
                    lambda Y_target: (
                        Y_target[:, :, : self.prediction_dim],
                        rearrange(
                            Y_target[:, :, self.prediction_dim :],
                            "b m (d1 d2) -> b m d1 d2",
                            d1=self.prediction_dim,
                            d2=self.prediction_dim,
                        ),
                    ),
                ),  # return the mean and covariance matrices of each point
            )
        else:
            raise NotImplementedError(f"Not implemented for dim = {self.dim}")
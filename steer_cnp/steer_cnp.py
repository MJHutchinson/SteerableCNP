import torch
import torch.nn as nn

from einops import rearrange

from steer_cnp.equiv_deepsets import EquivDeepSet
from steer_cnp.kernel import kernel_smooth


class SteerCNP(EquivDeepSet):
    def __init__(
        self,
        prediction_dim,
        covariance_activation_function,
        min_cov=0.0,
        **kwargs,
    ):

        super(SteerCNP, self).__init__(**kwargs)

        self.prediction_dim = prediction_dim
        self.covariance_activation_function = covariance_activation_function
        self.min_cov = min_cov

    def decode(self, X_grid, Y_grid, X_target):
        # reshape Y_grid to go through the CNN
        Y_grid = self.stack_to_grid(Y_grid)
        # pass Y's through the CNN
        Y_grid = self.cnn(Y_grid)
        # reshape Y's back to a stack
        Y_grid = self.grid_to_stack(Y_grid)
        # apply the covariance activation function to the covariances
        Y_grid = torch.cat(
            [
                Y_grid[:, :, : self.prediction_dim],
                self.covariance_activation_function(
                    Y_grid[:, :, self.prediction_dim :]
                ),
            ],
            dim=-1,
        )
        # kernel smooth the outputs to the target points
        Y_target = kernel_smooth(
            X_grid,
            Y_grid,
            X_target,
            self.output_kernel,
            normalise=self.normalise_output,
        )
        # split the output into mean and covariance, and reshape the covariances.
        Y_mean = Y_target[:, :, : self.prediction_dim]
        Y_cov = rearrange(
            Y_target[:, :, self.prediction_dim :],
            "b m (d1 d2) -> b m d1 d2",
            d1=self.prediction_dim,
            d2=self.prediction_dim,
        )
        # add on a minimum covariance
        Y_cov = Y_cov + (
            torch.eye(self.prediction_dim).to(Y_target.device) * self.min_cov
        )
        return Y_mean, Y_cov

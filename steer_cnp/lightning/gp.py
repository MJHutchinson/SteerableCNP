import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Subset

from steer_cnp.gp import conditional_gp_posterior
from steer_cnp.utils import multivariate_log_likelihood
from steer_cnp.lightning import Mean
from .utils import parse_kernel


class LightningGP(pl.LightningModule):
    def __init__(
        self,
        kernel_type="rbf",
        kernel_length_scale=1.0,
        kernel_sigma_var=1.0,
        kernel_learnable=False,
        x_dim=2,
        prediction_dim=2,
        chol_noise=1e-8,
        lr=5e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # parse the kernel to use
        self.kernel = parse_kernel(
            kernel_type=kernel_type,
            x_dim=x_dim,
            rkhs_dim=prediction_dim,
            length_scale=kernel_length_scale,
            sigma_var=kernel_sigma_var,
            learnable_length_scale=kernel_learnable,
        )

        self.optimise = False
        self.chol_noise = chol_noise

        self.lr = lr

        self.val_ll = Mean()
        self.test_ll = Mean()

        self.name = f"GP_{kernel_type}"

    def forward(self, X_context, Y_context, X_target):
        mean, covariance, variances = conditional_gp_posterior(
            X_context,
            Y_context,
            X_target,
            self.kernel,
            obs_noise=0,
            chol_noise=self.chol_noise,
        )

        return mean, variances

    def compute_batch_log_loss(self, batch, context_in_target=False):
        X_context, Y_context, X_target, Y_target = batch

        Y_prediction_mean, Y_prediction_cov = self.forward(
            X_context, Y_context, X_target
        )

        log_ll = multivariate_log_likelihood(
            Y_prediction_mean, Y_prediction_cov, Y_target
        )

        return log_ll

    def training_step(self, batch, batch_idx):
        log_ll = self.compute_batch_log_loss(batch)

        self.log("train_loss", -log_ll.mean(), on_step=True, on_epoch=False)
        self.log("train_log_lik", log_ll.mean(), on_step=True, on_epoch=False)
        self.log(
            "kernel_length_scale",
            self.kernel.scalar_kernel.log_length_scale.detach().exp(),
            on_step=True,
        )

        return -log_ll.mean()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        log_ll = self.compute_batch_log_loss(batch)

        self.val_ll(log_ll)

    def validation_epoch_end(self, validation_step_outputs):
        val_ll = self.val_ll.compute()
        self.log("val_ll", val_ll)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        log_ll = self.compute_batch_log_loss(batch)

        self.test_ll(log_ll)

    def test_epoch_end(self, test_step_outputs):
        test_ll = self.test_ll.compute()
        self.log("test_ll", test_ll)

    def configure_optimizers(self):
        if self.optimise:
            return torch.optim.Adam(self.kernel.parameters(), self.lr)
        else:
            return None

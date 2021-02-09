import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Subset

from steer_cnp import CNP
from steer_cnp.utils import multivariate_log_likelihood
from steer_cnp.lightning import Mean

from steer_cnp.covariance_activations import (
    quadratic_covariance_activation,
    diagonal_quadratic_covariance_activation,
    diagonal_softplus_covariance_activation,
    diagonal_quadratic_softplus_covariance_activation,
)

covariance_activation_functions = {
    "quadratic": quadratic_covariance_activation,
    "diagonal_quadratic": diagonal_quadratic_covariance_activation,
    "diagonal_softplus": diagonal_softplus_covariance_activation,
    "diagonal_softplus_quadratic": diagonal_quadratic_softplus_covariance_activation,
}


class LightningCNP(pl.LightningModule):
    def __init__(
        self,
        x_dim=2,
        context_dim=1,
        embedding_dim=512,
        prediction_dim=1,
        encoder_hidden_dims=[512, 512],
        decoder_hidden_dim=[512, 512, 512],
        x_encoder_hidden_dims=None,
        covariance_activation_function="diagonal_softplus",
        min_cov=0.01,
        lr=1e-3,
        batch_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cnp = CNP(
            x_dim,
            context_dim,
            embedding_dim,
            prediction_dim,
            list(encoder_hidden_dims),
            list(x_encoder_hidden_dims) if x_encoder_hidden_dims is not None else None,
            list(decoder_hidden_dim),
            covariance_activation_functions[covariance_activation_function],
            min_cov,
            batch_norm,
        )
        self.lr = lr

        self.val_ll = Mean()
        self.test_ll = Mean()

        self.name = f"CNP"

    def forward(self, X_context, Y_context, X_target):
        return self.cnp(X_context, Y_context, X_target)

    def compute_batch_log_loss(self, batch):
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
        optimizer = torch.optim.Adam(self.cnp.parameters(), self.lr)
        return optimizer


class LightningImageCNP(LightningCNP):
    def __init__(
        self,
        *args,
        sigmoid_mean=True,
        **kwargs,
    ):
        super(LightningImageCNP, self).__init__(*args, **kwargs)

        self.sigmoid_mean = sigmoid_mean

    def forward(self, X_context, Y_context, X_target):
        Y_prediction_mean, Y_prediction_cov = self.cnp(X_context, Y_context, X_target)

        if self.sigmoid_mean:
            Y_prediction_mean = torch.sigmoid(Y_prediction_mean)

        return Y_prediction_mean, Y_prediction_cov
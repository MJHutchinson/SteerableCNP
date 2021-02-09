import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Subset

from steer_cnp.steer_cnp import SteerCNP
from steer_cnp.utils import multivariate_log_likelihood
from steer_cnp.lightning import Mean

from .utils import parse_cnn_covariance_activation, parse_kernel


class LightningSteerCNP(pl.LightningModule):
    def __init__(
        self,
        embedding_kernel_type="rbf",
        embedding_kernel_length_scale=1.0,
        embedding_kernel_sigma_var=1.0,
        embedding_kernel_learnable=True,
        normalise_embedding=True,
        cnn_decoder="C4-regular_big",
        covariance_activation="quadratic",
        covariance_activation_parameters={},
        padding_mode="zeros",
        output_kernel_type="rbf",
        output_kernel_length_scale=1.0,
        output_kernel_sigma_var=1.0,
        output_kernel_learnable=True,
        normalise_output=True,
        grid_ranges=[-10, 10],
        n_axes=30,
        x_dim=2,
        context_dim=2,
        prediction_dim=2,
        context_in_target=False,
        min_cov=0.0,
        lr=5e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # parse the embedding kernel to use
        embedding_kernel = parse_kernel(
            kernel_type=embedding_kernel_type,
            x_dim=x_dim,
            rkhs_dim=context_dim + 1,
            length_scale=embedding_kernel_length_scale,
            sigma_var=embedding_kernel_sigma_var,
            learnable_length_scale=embedding_kernel_learnable,
        )
        # parse the CNN decoder and the activation function to apply to the covariances
        cnn, covariance_activation = parse_cnn_covariance_activation(
            cnn_decoder,
            context_dim,
            prediction_dim,
            covariance_activation,
            covariance_activation_parameters,
            padding_mode,
        )
        # parse the output kernel
        output_kernel = parse_kernel(
            kernel_type=output_kernel_type,
            x_dim=x_dim,
            rkhs_dim=prediction_dim + prediction_dim ** 2,
            length_scale=output_kernel_length_scale,
            sigma_var=output_kernel_sigma_var,
            learnable_length_scale=output_kernel_learnable,
        )

        # create the SteerCNP
        self.steer_cnp = SteerCNP(
            prediction_dim=prediction_dim,
            covariance_activation_function=covariance_activation,
            grid_ranges=grid_ranges,
            n_axes=n_axes,
            embedding_kernel=embedding_kernel,
            normalise_embedding=normalise_embedding,
            normalise_output=normalise_output,
            cnn=cnn,
            output_kernel=output_kernel,
            dim=x_dim,
            min_cov=min_cov,
        )

        self.context_in_target = context_in_target

        self.lr = lr

        self.val_ll = Mean()
        self.test_ll = Mean()

        self.name = f"EquivCNP_{cnn_decoder}"

    def forward(self, X_context, Y_context, X_target):
        return self.steer_cnp(X_context, Y_context, X_target)

    def compute_batch_log_loss(self, batch, context_in_target=False):
        X_context, Y_context, X_target, Y_target = batch

        if context_in_target:
            X_target = torch.cat([X_context, X_target], dim=1)
            Y_target = torch.cat([Y_context, Y_target], dim=1)

        Y_prediction_mean, Y_prediction_cov = self.forward(
            X_context, Y_context, X_target
        )

        log_ll = multivariate_log_likelihood(
            Y_prediction_mean, Y_prediction_cov, Y_target
        )

        # self.log("mean_pred", Y_prediction_mean.mean(), on_step=True, on_epoch=False)
        # self.log("mean_cov", Y_prediction_cov.mean(), on_step=True, on_epoch=False)

        return log_ll

    def training_step(self, batch, batch_idx):
        log_ll = self.compute_batch_log_loss(
            batch, context_in_target=self.context_in_target
        )

        self.log("train_loss", -log_ll.mean(), on_step=True, on_epoch=False)
        self.log("train_log_lik", log_ll.mean(), on_step=True, on_epoch=False)
        self.log(
            "input_length_scale",
            self.steer_cnp.discrete_rkhs_embedder.kernel.scalar_kernel.log_length_scale.detach().exp(),
            on_step=True,
        )
        self.log(
            "output_length_scale",
            self.steer_cnp.output_kernel.scalar_kernel.log_length_scale.detach().exp(),
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
        optimizer = torch.optim.Adam(self.steer_cnp.parameters(), self.lr)
        return optimizer


class LightningImageSteerCNP(LightningSteerCNP):
    def __init__(
        self,
        *args,
        sigmoid_mean=True,
        **kwargs,
    ):
        super(LightningImageSteerCNP, self).__init__(
            *args, sigmoid_mean=sigmoid_mean, **kwargs
        )

        self.sigmoid_mean = sigmoid_mean

    def on_train_epoch_start(self, *args, **kwargs):
        """Sets the grid in the EquivCNP appropriately for the data about to be passed"""
        if isinstance(self.train_dataloader().dataset, Subset):
            img_size = self.train_dataloader().dataset.dataset.grid_size
        else:
            img_size = self.train_dataloader().dataset.grid_size

        self.steer_cnp.discrete_rkhs_embedder.set_grid([-3, img_size + 2], img_size + 6)

    def on_test_epoch_start(self, *args, **kwargs):
        """Sets the grid in the EquivCNP appropriately for the data about to be passed"""
        test_dataloader = self.test_dataloader()
        if isinstance(test_dataloader, list):
            # currently no way to figure out which
            test_dataloader = test_dataloader[0]

        if isinstance(test_dataloader.dataset, Subset):
            img_size = test_dataloader.dataset.dataset.grid_size
        else:
            img_size = test_dataloader.dataset.grid_size

        self.steer_cnp.discrete_rkhs_embedder.set_grid([-3, img_size + 2], img_size + 6)

    def on_validation_epoch_start(self, *args, **kwargs):
        """Sets the grid in the EquivCNP appropriately for the data about to be passed"""
        val_dataloader = self.val_dataloader()
        if isinstance(val_dataloader, list):
            # currently no way to figure out which
            val_dataloader = val_dataloader[0]

        if isinstance(val_dataloader.dataset, Subset):
            img_size = val_dataloader.dataset.dataset.grid_size
        else:
            img_size = val_dataloader.dataset.grid_size

        self.steer_cnp.discrete_rkhs_embedder.set_grid([-3, img_size + 2], img_size + 6)

    def forward(self, X_context, Y_context, X_target):
        Y_prediction_mean, Y_prediction_cov = self.steer_cnp(
            X_context, Y_context, X_target
        )

        if self.sigmoid_mean:
            Y_prediction_mean = torch.sigmoid(Y_prediction_mean)

        return Y_prediction_mean, Y_prediction_cov

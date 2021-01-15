import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import Subset

from equiv_cnp.equiv_cnp import EquivCNP
from equiv_cnp.utils import multivariate_log_likelihood, get_e2_decoder, get_cnn_decoder
from equiv_cnp.kernel import (
    SeparableKernel,
    RBFKernel,
    RBFKernelReparametrised,
    RBFDivergenceFreeKernelReparametrised,
    RBFCurlFreeKernelReparametrised,
)
from equiv_cnp.covariance_activations import (
    quadratic_covariance_activation,
    diagonal_quadratic_covariance_activation,
    diagonal_softplus_covariance_activation,
    diagonal_quadratic_softplus_covariance_activation,
)
from equiv_cnp.lightning import Mean

covariance_activation_functions = {
    "quadratic": quadratic_covariance_activation,
    "diagonal_quadratic": diagonal_quadratic_covariance_activation,
    "diagonal_softplus": diagonal_softplus_covariance_activation,
    "diagonal_softplus_quadratic": diagonal_quadratic_softplus_covariance_activation,
}


class LightningEquivCNP(pl.LightningModule):
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
        lr=5e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # parse the embedding kernel to use
        embedding_kernel = self.parse_kernel(
            kernel_type=embedding_kernel_type,
            x_dim=x_dim,
            rkhs_dim=context_dim + 1,
            length_scale=embedding_kernel_length_scale,
            sigma_var=embedding_kernel_sigma_var,
            learnable_length_scale=embedding_kernel_learnable,
        )
        # parse the CNN decoder and the activation function to apply to the covariances
        cnn, covariance_activation = self.parse_cnn_covariance_activation(
            cnn_decoder,
            context_dim,
            prediction_dim,
            covariance_activation,
            covariance_activation_parameters,
        )
        # parse the output kernel
        output_kernel = self.parse_kernel(
            kernel_type=output_kernel_type,
            x_dim=x_dim,
            rkhs_dim=prediction_dim + prediction_dim ** 2,
            length_scale=output_kernel_length_scale,
            sigma_var=output_kernel_sigma_var,
            learnable_length_scale=output_kernel_learnable,
        )

        # create the EquivCNP
        self.equiv_cnp = EquivCNP(
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
        )

        self.context_in_target = context_in_target

        self.lr = lr

        self.val_ll = Mean()
        self.test_ll = Mean()

        self.name = f"EquivCNP_{cnn_decoder}"

    def parse_kernel(
        self,
        kernel_type,
        x_dim,
        rkhs_dim,
        length_scale,
        sigma_var,
        learnable_length_scale=False,
    ):
        if kernel_type == "rbf":
            kernel = SeparableKernel(
                x_dim,
                rkhs_dim,
                RBFKernelReparametrised(
                    x_dim,
                    log_length_scale=nn.Parameter(torch.tensor(length_scale).log())
                    if learnable_length_scale
                    else torch.tensor(length_scale).log(),
                    sigma_var=sigma_var,
                ),
            )
        elif kernel_type == "divfree":
            if (rkhs_dim != x_dim) & rkhs_dim != 2:
                raise ValueError(
                    f"RKHS and X dim for {kernel_type} must be 2. Given {rkhs_dim=}, {x_dim=}."
                )
            kernel = RBFDivergenceFreeKernelReparametrised(
                x_dim,
                log_length_scale=nn.Parameter(torch.tensor(length_scale).log())
                if learnable_length_scale
                else torch.tensor(length_scale).log(),
                sigma_var=sigma_var,
            )
        elif kernel_type == "curlfree":
            if (rkhs_dim != x_dim) & rkhs_dim != 2:
                raise ValueError(
                    f"RKHS and X dim for {kernel_type} must be 2. Given {rkhs_dim=}, {x_dim=}."
                )
            kernel = RBFCurlFreeKernelReparametrised(
                x_dim,
                log_length_scale=nn.Parameter(torch.tensor(length_scale).log())
                if learnable_length_scale
                else torch.tensor(length_scale).log(),
                sigma_var=sigma_var,
            )
        else:
            raise ValueError(f"{kernel_type} is not a recognised kernel type to use.")

        return kernel

    def parse_cnn_covariance_activation(
        self,
        cnn_decoder_string,
        context_dim,
        prediction_dim,
        covariance_activation,
        covariance_activation_parameters,
    ):
        group, name = cnn_decoder_string.split("-")

        # get the right activation function and create a lambda with the specified arguments passed in
        covariance_activation_function = lambda X: covariance_activation_functions[
            covariance_activation
        ](X, **covariance_activation_parameters)

        if group == "T2":
            return (
                get_cnn_decoder(
                    name,
                    context_dim,
                    prediction_dim,
                    covariance_activation=covariance_activation,
                ),
                covariance_activation_function,
            )
        else:
            flip = group[0] == "D"
            N = int(group[1:])

            if context_dim == 1:
                rep_ids = [[0]] if not flip else [[0, 0]]
            elif context_dim == 2:
                rep_ids = [[1]] if not flip else [[1, 1]]

            return (
                get_e2_decoder(
                    N, flip, name, rep_ids, rep_ids, covariance_activation="quadratic"
                ),
                covariance_activation_function,
            )

    def forward(self, X_context, Y_context, X_target):
        return self.equiv_cnp(X_context, Y_context, X_target)

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
            self.equiv_cnp.discrete_rkhs_embedder.kernel.scalar_kernel.log_length_scale.detach().exp(),
            on_step=True,
        )
        self.log(
            "output_length_scale",
            self.equiv_cnp.output_kernel.scalar_kernel.log_length_scale.detach().exp(),
            on_step=True,
        )

        return -log_ll.mean()

    def validation_step(self, batch, batch_idx):
        log_ll = self.compute_batch_log_loss(batch)

        self.val_ll(log_ll)

    def validation_epoch_end(self, validation_step_outputs):
        val_ll = self.val_ll.compute()
        self.log("val_ll", val_ll)

    def test_step(self, batch, batch_idx):
        log_ll = self.compute_batch_log_loss(batch)

        self.test_ll(log_ll)

    def test_epoch_end(self, test_step_outputs):
        test_ll = self.test_ll.compute()

        self.log("test_ll", test_ll, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.equiv_cnp.parameters(), self.lr)
        return optimizer


class LightningImageEquivCNP(LightningEquivCNP):
    def __init__(
        self,
        *args,
        sigmoid_mean=True,
        **kwargs,
    ):
        super(LightningImageEquivCNP, self).__init__(
            *args, sigmoid_mean=sigmoid_mean, **kwargs
        )

        self.sigmoid_mean = sigmoid_mean

    def on_train_epoch_start(self, *args, **kwargs):
        """Sets the grid in the EquivCNP appropriately for the data about to be passed"""
        if isinstance(self.train_dataloader().dataset, Subset):
            img_size = self.train_dataloader().dataset.dataset.grid_size
        else:
            img_size = self.train_dataloader().dataset.grid_size

        self.equiv_cnp.discrete_rkhs_embedder.set_grid([-3, img_size + 2], img_size + 6)

    def on_test_epoch_start(self, *args, **kwargs):
        """Sets the grid in the EquivCNP appropriately for the data about to be passed"""
        if isinstance(self.test_dataloader().dataset, Subset):
            img_size = self.test_dataloader().dataset.dataset.grid_size
        else:
            img_size = self.test_dataloader().dataset.grid_size

        self.equiv_cnp.discrete_rkhs_embedder.set_grid([-3, img_size + 2], img_size + 6)

    def on_validation_epoch_start(self, *args, **kwargs):
        """Sets the grid in the EquivCNP appropriately for the data about to be passed"""
        if isinstance(self.val_dataloader().dataset, Subset):
            img_size = self.val_dataloader().dataset.dataset.grid_size
        else:
            img_size = self.val_dataloader().dataset.grid_size

        self.equiv_cnp.discrete_rkhs_embedder.set_grid([-3, img_size + 2], img_size + 6)

    def forward(self, X_context, Y_context, X_target):
        Y_prediction_mean, Y_prediction_cov = self.equiv_cnp(
            X_context, Y_context, X_target
        )

        if self.sigmoid_mean:
            Y_prediction_mean = torch.sigmoid(Y_prediction_mean)

        return Y_prediction_mean, Y_prediction_cov

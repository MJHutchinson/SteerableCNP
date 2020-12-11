import torch
import torch.nn as nn
import pytorch_lightning as pl

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
    diagonal_covariance_activation,
)
from equiv_cnp.lightning import Mean


class LightningEquivCNP(pl.LightningModule):
    def __init__(
        self,
        embedding_kernel_type="rbf",
        embedding_kernel_length_scale=1.0,
        embedding_kernel_sigma_var=1.0,
        embedding_kernel_learnable=True,
        normalise_embedding=True,
        cnn_decoder="C4-regular_big",
        output_kernel_type="rbf",
        output_kernel_length_scale=1.0,
        output_kernel_sigma_var=1.0,
        output_kernel_learnable=True,
        normalise_output=True,
        grid_ranges=[-10, 10],
        n_axes=30,
        dim=2,
    ):
        super().__init__()

        embedding_kernel = self.parse_kernel(
            embedding_kernel_type,
            3,
            embedding_kernel_length_scale,
            embedding_kernel_sigma_var,
            embedding_kernel_learnable,
        )
        grid_ranges = [-10, 10]
        cnn, covariance_activation = self.parse_cnn(cnn_decoder)
        output_kernel = self.parse_kernel(
            output_kernel_type,
            6,
            output_kernel_length_scale,
            output_kernel_sigma_var,
            output_kernel_learnable,
        )

        self.equiv_cnp = EquivCNP(
            prediction_dim=dim,
            covariance_activation_function=covariance_activation,
            grid_ranges=grid_ranges,
            n_axes=n_axes,
            embedding_kernel=embedding_kernel,
            normalise_embedding=normalise_embedding,
            normalise_output=normalise_output,
            cnn=cnn,
            output_kernel=output_kernel,
            dim=dim,
        )

        self.val_ll = Mean()
        self.test_ll = Mean()

        self.name = f"EquivCNP_{cnn_decoder}"

    def parse_kernel(
        self,
        kernel_type,
        rkhs_dim,
        length_scale,
        sigma_var,
        learnable_length_scale=False,
    ):
        if kernel_type == "rbf":
            kernel = SeparableKernel(
                2,
                rkhs_dim,
                RBFKernelReparametrised(
                    2,
                    log_length_scale=nn.Parameter(torch.tensor(length_scale).log())
                    if learnable_length_scale
                    else torch.tensor(length_scale).log(),
                    sigma_var=sigma_var,
                ),
            )
        elif kernel_type == "divfree":
            if rkhs_dim != 2:
                raise ValueError(
                    f"RKHS dim for {kernel_type} must be 2. Given {rkhs_dim}."
                )
            kernel = RBFDivergenceFreeKernelReparametrised(
                2,
                log_length_scale=nn.Parameter(torch.tensor(length_scale).log())
                if learnable_length_scale
                else torch.tensor(length_scale).log(),
                sigma_var=sigma_var,
            )
        elif kernel_type == "curlfree":
            if rkhs_dim != 2:
                raise ValueError(
                    f"RKHS dim for {kernel_type} must be 2. Given {rkhs_dim}."
                )
            kernel = RBFCurlFreeKernelReparametrised(
                2,
                log_length_scale=nn.Parameter(torch.tensor(length_scale).log())
                if learnable_length_scale
                else torch.tensor(length_scale).log(),
                sigma_var=sigma_var,
            )
        else:
            raise ValueError(f"{kernel_type} is not a recognised kernel type to use.")

        return kernel

    def parse_cnn(self, cnn_decoder_string):
        group, name = cnn_decoder_string.split("-")

        if group == "T2":
            return (
                get_cnn_decoder(name, 3, 2, covariance_activation="diagonal"),
                diagonal_covariance_activation,
            )
        else:
            flip = group[0] == "D"
            N = int(group[1])

            return (
                get_e2_decoder(
                    N, flip, name, [1], [1], covariance_activation="quadratic"
                ),
                quadratic_covariance_activation,
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

        return log_ll

    def training_step(self, batch, batch_idx):
        log_ll = self.compute_batch_log_loss(batch, context_in_target=False)

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
        optimizer = torch.optim.Adam(self.equiv_cnp.parameters(), 1e-4)
        return optimizer

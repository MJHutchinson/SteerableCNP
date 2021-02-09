import torch
import torch.nn as nn

from steer_cnp.utils import get_e2_decoder, get_cnn_decoder

from steer_cnp.kernel import (
    SeparableKernel,
    RBFKernel,
    RBFKernelReparametrised,
    RBFDivergenceFreeKernelReparametrised,
    RBFCurlFreeKernelReparametrised,
)
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


def parse_kernel(
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
    cnn_decoder_string,
    context_dim,
    prediction_dim,
    covariance_activation,
    covariance_activation_parameters,
    padding_mode,
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
                padding_mode=padding_mode,
            ),
            covariance_activation_function,
        )
    elif group == "SO2":
        if context_dim == 1:
            rep_ids = [[0]]
        elif context_dim == 2:
            rep_ids = [[1]]
        return (
            get_e2_decoder(
                -1,
                False,
                name,
                rep_ids,
                rep_ids,
                covariance_activation=covariance_activation,
                padding_mode=padding_mode,
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
                N,
                flip,
                name,
                rep_ids,
                rep_ids,
                covariance_activation=covariance_activation,
                padding_mode=padding_mode,
            ),
            covariance_activation_function,
        )

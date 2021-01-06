import torch
import torch.nn as nn

activations = {"relu": nn.ReLU}


def build_cnn_2d(in_dim, hidden_dims, kernel_sizes, out_dim, activation="relu"):
    """
    Input:
        in_dim - number of channels to the input data
        hidden_dims - list of hidden channel dimensions
        kernel_size - size of the kernels to use in each cnn layer
        out_dim - number of output channels
        non_linearity - string name of the activation to use between layers
    -->Creates a stack of CNN with the specified parameters
    """

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * (len(hidden_dims) + 1)

    layer_dims = [in_dim, *hidden_dims, out_dim]

    layers = []

    for i in range(len(layer_dims) - 1):
        layers.append(
            nn.Conv2d(
                layer_dims[i],
                layer_dims[i + 1],
                kernel_sizes[i],
                padding=int((kernel_sizes[i] - 1) / 2),
            )
        )
        layers.append(activations[activation]())

    # chop off last unused activation
    layers = layers[:-1]

    return nn.Sequential(*layers)


def build_cnn_decoder(
    context_dim,
    hidden_dims,
    kernel_sizes,
    mean_dim,
    cov_est_dim,
    activation="relu",
):
    return build_cnn_2d(
        context_dim + 1,
        hidden_dims,
        kernel_sizes,
        mean_dim + cov_est_dim,
        activation,
    )

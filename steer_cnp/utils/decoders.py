import torch
import torch.nn as nn

from e2cnn import gspaces
from e2cnn import nn as gnn
from e2cnn import group
import e2cnn


# from e3nn.networks import ImageGatedConvNetwork


from steer_cnp.cnns import build_steer_cnn_decoder, build_cnn_decoder
from steer_cnp.utils import get_pre_covariance_dim

"""
Number of parameters:
little - ca 1 000
small - ca 20 000
middle - ca 100 000
big - ca 500 000
huge - ca 2M
"""


LIST_NAMES = [
    "regular_little",
    "regular_small",
    "regular_middle",
    "regular_big",
    "regular_huge",
    "irrep_little",
    "irrep_small",
    "irrep_middle",
    "irrep_big",
    "irrep_huge",
]


def get_continuous_decoder_parameters(name, flip=False):
    if name == "irrep_little":
        if flip:
            hidden_reps_ids = 2 * [4 * [[1, 1]]]
        else:
            hidden_reps_ids = 2 * [8 * [1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["normrelu"]

    elif name == "irrep_small":
        if flip:
            hidden_reps_ids = 5 * [6 * [[1, 1]]]
        else:
            hidden_reps_ids = 5 * [12 * [1]]
        kernel_sizes = [j for j in range(3, 24, 4)]
        non_linearity = ["normrelu"]

    elif name == "irrep_middle":
        if flip:
            hidden_reps_ids = 7 * [18 * [[1, 1]]]
        else:
            hidden_reps_ids = 7 * [36 * [1]]
        kernel_sizes = [3, 3, 5, 5, 11, 11, 13, 13]
        non_linearity = ["normrelu"]

    elif name == "irrep_big":
        if flip:
            hidden_reps_ids = 8 * [32 * [[1, 1]]]
        else:
            hidden_reps_ids = 8 * [64 * [1]]
        kernel_sizes = [5, 5, 7, 7, 11, 13, 15, 17, 19]
        non_linearity = ["normrelu"]

    elif name == "irrep_huge":
        if flip:
            hidden_reps_ids = 10 * [40 * [[1, 1]]]
        else:
            hidden_reps_ids = 10 * [80 * [1]]
        kernel_sizes = [5, 5, 7, 7, 11, 11, 11, 13, 17, 19, 21]
        non_linearity = ["normrelu"]

    elif name == "irrep_big_gated":
        hidden_reps_ids = 10 * [32 * [0] + 32 * [1]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11, 13, 15, 17, 19]
        non_linearity = ["Gated"]

    else:
        raise ValueError(
            f"{name} is not a recognised architecture for continous decoders"
        )

    return hidden_reps_ids, kernel_sizes, non_linearity[0]


def get_C16_parameters(name):
    # Family of decoders using purely regular fiber representations:
    if name == "regular_little":
        hidden_reps_ids = 2 * [3 * [-1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["relu"]

    elif name == "regular_small":
        hidden_reps_ids = 4 * [4 * [-1]]
        kernel_sizes = [j for j in range(3, 20, 4)]
        non_linearity = ["relu"]

    elif name == "regular_middle":
        hidden_reps_ids = 6 * [12 * [-1]]
        kernel_sizes = [3, 3, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_big":
        hidden_reps_ids = 6 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_huge":
        hidden_reps_ids = 8 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 5, 7, 7, 9, 15, 21]
        non_linearity = ["relu"]

    # Family of decoders using irreps and regular representations:

    elif name == "irrep_little":
        hidden_reps_ids = 2 * [8 * [1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["normrelu"]

    elif name == "irrep_small":
        hidden_reps_ids = 5 * [12 * [1]]
        kernel_sizes = [j for j in range(3, 24, 4)]
        non_linearity = ["normrelu"]

    elif name == "irrep_middle":
        hidden_reps_ids = 7 * [36 * [1]]
        kernel_sizes = [3, 3, 5, 5, 11, 11, 13, 13]
        non_linearity = ["normrelu"]

    elif name == "irrep_big":
        hidden_reps_ids = 10 * [64 * [1]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11, 13, 15, 17, 19]
        non_linearity = ["normrelu"]

    elif name == "irrep_huge":
        hidden_reps_ids = 16 * [80 * [1]]
        kernel_sizes = [5, 5, 5, 5, 7, 7, 7, 7, 11, 11, 11, 11, 13, 15, 17, 19, 21]
        non_linearity = ["normrelu"]
    else:
        raise ValueError(f"{name} is not a recognised architecture for C16 decoders")

    return hidden_reps_ids, kernel_sizes, non_linearity[0]


def get_D8_parameters(name):
    # Family of decoders using purely regular fiber representations:
    if name == "regular_little":
        hidden_reps_ids = 2 * [3 * [-1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["relu"]

    elif name == "regular_small":
        hidden_reps_ids = 4 * [4 * [-1]]
        kernel_sizes = [j for j in range(3, 20, 4)]
        non_linearity = ["relu"]

    elif name == "regular_middle":
        hidden_reps_ids = 6 * [12 * [-1]]
        kernel_sizes = [3, 3, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_big":
        hidden_reps_ids = 6 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_huge":
        hidden_reps_ids = 8 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 5, 7, 7, 9, 15, 21]
        non_linearity = ["relu"]

    # Family of decoders using irreps and regular representations:

    elif name == "irrep_little":
        hidden_reps_ids = 2 * [8 * [[1, 1]]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["normrelu"]

    elif name == "irrep_small":
        hidden_reps_ids = 5 * [12 * [[1, 1]]]
        kernel_sizes = [j for j in range(3, 24, 4)]
        non_linearity = ["normrelu"]

    elif name == "irrep_middle":
        hidden_reps_ids = 7 * [36 * [[1, 1]]]
        kernel_sizes = [3, 3, 5, 5, 11, 11, 13, 13]
        non_linearity = ["normrelu"]

    elif name == "irrep_big":
        hidden_reps_ids = 10 * [64 * [[1, 1]]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11, 13, 15, 17, 19]
        non_linearity = ["normrelu"]

    elif name == "irrep_huge":
        hidden_reps_ids = 16 * [80 * [[1, 1]]]
        kernel_sizes = [5, 5, 5, 5, 7, 7, 7, 7, 11, 11, 11, 11, 13, 15, 17, 19, 21]
        non_linearity = ["normrelu"]
    else:
        raise ValueError(f"{name} is not a recognised architecture for C16 decoders")

    return hidden_reps_ids, kernel_sizes, non_linearity[0]


def get_D4_parameters(name):
    # Family of decoders using purely regular fiber representations:
    if name == "regular_little":
        hidden_reps_ids = 2 * [3 * [-1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["relu"]

    elif name == "regular_small":
        hidden_reps_ids = 4 * [4 * [-1]]
        kernel_sizes = [j for j in range(3, 20, 4)]
        non_linearity = ["relu"]

    elif name == "regular_middle":
        hidden_reps_ids = 6 * [12 * [-1]]
        kernel_sizes = [3, 3, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_big":
        hidden_reps_ids = 6 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_huge":
        hidden_reps_ids = 8 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 5, 7, 7, 9, 15, 21]
        non_linearity = ["relu"]

    # Family of decoders using irreps and regular representations:

    elif name == "irrep_little":
        hidden_reps_ids = 2 * [4 * [[1, 1]]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["normrelu"]

    elif name == "irrep_small":
        hidden_reps_ids = 5 * [6 * [[1, 1]]]
        kernel_sizes = [j for j in range(3, 24, 4)]
        non_linearity = ["normrelu"]

    elif name == "irrep_middle":
        hidden_reps_ids = 7 * [18 * [[1, 1]]]
        kernel_sizes = [3, 3, 5, 5, 11, 11, 13, 13]
        non_linearity = ["normrelu"]

    elif name == "irrep_big":
        hidden_reps_ids = 8 * [32 * [[1, 1]]]
        kernel_sizes = [5, 5, 7, 7, 11, 13, 15, 17, 19]
        non_linearity = ["normrelu"]

    elif name == "irrep_huge":
        hidden_reps_ids = 10 * [40 * [[1, 1]]]
        kernel_sizes = [5, 5, 7, 7, 11, 11, 11, 13, 17, 19, 21]
        non_linearity = ["normrelu"]
    else:
        raise ValueError(f"{name} is not a recognised architecture for C16 decoders")
    return hidden_reps_ids, kernel_sizes, non_linearity[0]


def get_C8_parameters(name):
    # Family of decoders using purely regular fiber representations:
    if name == "regular_little":
        hidden_reps_ids = 2 * [3 * [-1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["relu"]

    elif name == "regular_small":
        hidden_reps_ids = 4 * [4 * [-1]]
        kernel_sizes = [j for j in range(3, 20, 4)]
        non_linearity = ["relu"]

    elif name == "regular_middle":
        hidden_reps_ids = 6 * [12 * [-1]]
        kernel_sizes = [3, 3, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_big":
        hidden_reps_ids = 6 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_huge":
        hidden_reps_ids = 8 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 5, 7, 7, 9, 15, 21]
        non_linearity = ["relu"]
    else:
        raise ValueError(f"{name} is not a recognised architecture for C8 decoders")
    return hidden_reps_ids, kernel_sizes, non_linearity[0]


def get_C4_parameters(name):
    # Family of decoders using purely regular fiber representations:
    if name == "regular_little":
        hidden_reps_ids = 2 * [3 * [-1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["relu"]

    elif name == "regular_small":
        hidden_reps_ids = 4 * [4 * [-1]]
        kernel_sizes = [j for j in range(3, 20, 4)]
        non_linearity = ["relu"]

    elif name == "regular_middle":
        hidden_reps_ids = 6 * [12 * [-1]]
        kernel_sizes = [3, 3, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_big":
        hidden_reps_ids = 6 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 7, 7, 11, 11]
        non_linearity = ["relu"]

    elif name == "regular_huge":
        hidden_reps_ids = 8 * [24 * [-1]]
        kernel_sizes = [5, 5, 5, 5, 7, 7, 9, 15, 21]
        non_linearity = ["relu"]

    # Family of decoders using irreps and regular representations:

    elif name == "irrep_little":
        hidden_reps_ids = 2 * [4 * [1]]
        kernel_sizes = [j for j in range(7, 20, 6)]
        non_linearity = ["normrelu"]

    elif name == "irrep_small":
        hidden_reps_ids = 5 * [6 * [1]]
        kernel_sizes = [j for j in range(3, 24, 4)]
        non_linearity = ["normrelu"]

    elif name == "irrep_middle":
        hidden_reps_ids = 7 * [18 * [1]]
        kernel_sizes = [3, 3, 5, 5, 11, 11, 13, 13]
        non_linearity = ["normrelu"]

    elif name == "irrep_big":
        hidden_reps_ids = 8 * [32 * [1]]
        kernel_sizes = [5, 5, 7, 7, 11, 13, 15, 17, 19]
        non_linearity = ["normrelu"]

    elif name == "irrep_huge":
        hidden_reps_ids = 10 * [40 * [1]]
        kernel_sizes = [5, 5, 7, 7, 11, 11, 11, 13, 17, 19, 21]
        non_linearity = ["normrelu"]
    else:
        raise ValueError(f"{name} is not a recognised architecture for C16 decoders")
    return hidden_reps_ids, kernel_sizes, non_linearity[0]


# def get_E3_paramters(name):
#     if name == 'little':
#         hidden_reps_ids =
#     elif name == "small":
#     elif name == "middle":
#     elif name == "big":
#     elif name == "huge":
#     else:
#         raise ValueError(f"{name} is not a recognised architecture for E3 decoders")
#     return hidden_reps_ids, kernel_size, layers

# def get_e3_decoder(name, context_rep_ids, mean_rep_ids):


def get_e2_decoder(
    N,
    flip,
    name,
    context_rep_ids,
    mean_rep_ids,
    covariance_activation="quadratic",
    padding_mode="zeros",
    max_frequency=30,
):

    if N == -1:
        (
            hidden_reps_ids,
            kernel_sizes,
            non_linearity,
        ) = get_continuous_decoder_parameters(name, flip)
    elif N == 16 and flip == False:
        hidden_reps_ids, kernel_sizes, non_linearity = get_C16_parameters(name)
    elif N == 8 and flip == True:
        hidden_reps_ids, kernel_sizes, non_linearity = get_D8_parameters(name)
    elif N == 8 and flip == False:
        hidden_reps_ids, kernel_sizes, non_linearity = get_C8_parameters(name)
    elif N == 4 and flip == True:
        hidden_reps_ids, kernel_sizes, non_linearity = get_D4_parameters(name)
    elif N == 4 and flip == False:
        hidden_reps_ids, kernel_sizes, non_linearity = get_C4_parameters(name)
    else:
        raise ValueError(
            f"{N}, {flip}, {name} are not a valid combination to build an architecture"
        )

    return build_steer_cnn_decoder(
        context_rep_ids,
        hidden_reps_ids,
        kernel_sizes,
        mean_rep_ids,
        covariance_activation,
        N=N,
        flip=flip,
        activation=non_linearity,
        padding_mode=padding_mode,
        max_frequency=max_frequency,
    )


def get_cnn_decoder(
    name,
    context_dim,
    mean_dim,
    covariance_activation="diagonal",
    activation="relu",
    padding_mode="zeros",
):

    if name == "little":
        list_hid_channels = [4, 5]
        kernel_sizes = [5, 7, 5]
        non_linearity = ["relu"]

    elif name == "small":
        list_hid_channels = [8, 12, 12]
        kernel_sizes = [5, 9, 11, 5]
        non_linearity = "relu"

    elif name == "middle":
        list_hid_channels = 4 * [24]
        kernel_sizes = 5 * [7]
        non_linearity = "relu"

    elif name == "big":
        list_hid_channels = 6 * [40]
        kernel_sizes = 7 * [7]
        non_linearity = "relu"

    elif name == "huge":
        list_hid_channels = 8 * [52]
        kernel_sizes = 9 * [9]
        non_linearity = "relu"
    else:
        raise ValueError(f"{name} is not a recognised architecture for cnn decoders")

    cov_est_dim = get_pre_covariance_dim(mean_dim, covariance_activation)

    return build_cnn_decoder(
        context_dim,
        list_hid_channels,
        kernel_sizes,
        mean_dim,
        cov_est_dim,
        padding_mode=padding_mode,
        activation=non_linearity,
    )

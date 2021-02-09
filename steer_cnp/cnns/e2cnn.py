import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

import e2cnn
from e2cnn import gspaces, group
from e2cnn import nn as gnn

from steer_cnp.utils import Expression, get_pre_covariance_field_type, reps_from_ids

activations = {"relu": gnn.ReLU, "normrelu": gnn.NormNonLinearity}


def build_steer_cnn_2d(
    in_field_type,
    hidden_field_types,
    kernel_sizes,
    out_field_type,
    gspace,
    activation="relu",
    padding_mode="zeros",
    modify_init=1.0,
):
    """
    Input:
        in_rep - rep of representation of the input data
        hidden_reps - the reps to use in the hidden layers
        kernel sizes - the size of the kernel used in each layer
        out_rep - the rep to use in the ouput layer
        activation - the activation to use between layers
        gspace - the gsapce that data lives in
    """

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * (len(hidden_reps) + 1)

    layer_field_types = [in_field_type, *hidden_field_types, out_field_type]

    layers = []

    for i in range(len(layer_field_types) - 1):
        layers.append(
            gnn.R2Conv(
                layer_field_types[i],
                layer_field_types[i + 1],
                kernel_sizes[i],
                padding=int((kernel_sizes[i] - 1) / 2),
                padding_mode=padding_mode,
                initialize=True,
            )
        )
        if i != len(layer_field_types) - 2:
            layers.append(activations[activation](layer_field_types[i + 1]))

    cnn = gnn.SequentialModule(*layers)

    # TODO: dirty fix to alleviate weird initialisations
    for p in cnn.parameters():
        if p.dim() == 0:
            p.data = p.data * modify_init
        else:
            p.data[:] = p.data * modify_init

    return nn.Sequential(
        Expression(lambda X: gnn.GeometricTensor(X, in_field_type)),
        cnn,
        Expression(lambda X: X.tensor),
    )


def build_steer_cnn_decoder(
    context_rep_ids,
    hidden_reps_ids,
    kernel_sizes,
    mean_rep_ids,
    covariance_activation="quadratic",
    N=4,
    flip=True,
    max_frequency=30,
    activation="relu",
    padding_mode="zeros",
):
    if flip:
        gspace = (
            gspaces.FlipRot2dOnR2(N=N)
            if N != -1
            else gspaces.FlipRot2dOnR2(N=N, maximum_frequency=max_frequency)
        )
    else:
        gspace = (
            gspaces.Rot2dOnR2(N=N)
            if N != -1
            else gspaces.Rot2dOnR2(N=N, maximum_frequency=max_frequency)
        )

    in_field_type = gnn.FieldType(
        gspace, [gspace.trivial_repr, *reps_from_ids(gspace, context_rep_ids)]
    )

    hidden_field_types = [
        gnn.FieldType(gspace, reps_from_ids(gspace, ids)) for ids in hidden_reps_ids
    ]

    mean_field_type = gnn.FieldType(gspace, reps_from_ids(gspace, mean_rep_ids))

    pre_covariance_field_type = get_pre_covariance_field_type(
        gspace, mean_field_type, covariance_activation
    )

    out_field_type = mean_field_type + pre_covariance_field_type

    init_modify = (
        1.0 if not (mean_rep_ids == [[0]] and context_rep_ids == [[0]]) else 0.833,
    )
    if N == -1:
        init_modify = 1.0

    return build_steer_cnn_2d(
        in_field_type,
        hidden_field_types,
        kernel_sizes,
        out_field_type,
        gspace,
        activation,
        padding_mode,
    )

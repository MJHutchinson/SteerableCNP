import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

import e2cnn
from e2cnn import gspaces, group
from e2cnn import nn as gnn


def eigenvalue_covariance_estimate_rep(gspace):
    """
    Input:
        gspace - instance of e2cnn.gspaces.r2.rot2d_on_r2.Rot2dOnR2 - underlying group

    Output:
        psd_rep - instance of e2cnn.group.Representation - group representation of the group representation before the covariance
    """
    # Change of basis matrix:
    change_of_basis = np.array([[1, 1.0, 0.0], [0.0, 0.0, 1.0], [1, -1.0, 0.0]])

    # Get group order and control:
    N = gspace.fibergroup.order()
    if N <= 3 and N != -1:
        sys.exit("Group order is not valid.")

    if isinstance(gspace, gspaces.FlipRot2dOnR2):
        irreps = (
            ["irrep_0,0", "irrep_1,2"]
            if N > 4
            else ["irrep_0,0", "irrep_1,2", "irrep_1,2"]
        )
    elif isinstance(gspace, gspaces.Rot2dOnR2):
        irreps = ["irrep_0", "irrep_2"] if N > 4 else ["irrep_0", "irrep_2", "irrep_2"]
    else:
        sys.exit("Error: Unknown group.")

    psd_rep = e2cnn.group.Representation(
        group=gspace.fibergroup,
        name="eig_val_rep",
        irreps=irreps,
        change_of_basis=change_of_basis,
        supported_nonlinearities=["n_relu"],
    )

    return psd_rep


def get_pre_covariance_field_type(
    gspace, mean_field_type, covariance_activation="quadratic"
):
    if covariance_activation == "quadratic":
        # stack dimension copies of the mean prediction field type
        return sum(
            (mean_field_type.size - 1) * [mean_field_type], start=mean_field_type
        )
    elif covariance_activation in [
        "diagonal_softplus_quadratic",
        "diagonal_softplus",
        "diagonal_quadratic",
    ]:
        # Should ONLY be used with scale fields
        return mean_field_type
    elif covariance_activation == "eigenvalue":
        return gnn.FieldType(gspace, eigenvalue_covariance_estimate_rep(gspace))
    else:
        raise ValueError(
            f"{covariance_activation} is not a recognised covariance activation type"
        )


def get_pre_covariance_dim(mean_dim, covariance_activation="quadratic"):
    if covariance_activation == "quadratic":
        return mean_dim ** 2
    elif covariance_activation == "eigenvalue":
        return 6
    elif covariance_activation in [
        "diagonal_softplus_quadratic",
        "diagonal_softplus",
        "diagonal_quadratic",
    ]:
        return mean_dim
    else:
        raise ValueError(
            f"{covariance_activation} is not a recognised covariance activation type"
        )

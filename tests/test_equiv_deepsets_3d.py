# %%
import torch

from equiv_cnp.kernel import (
    RBFKernel,
    SeparableKernel,
    DotProductKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
    kernel_smooth,
)

from equiv_cnp.equiv_deepsets import EquivDeepSet
from equiv_cnp.utils import (
    get_e2_decoder,
    grid_2d,
    plot_vector_field,
    plot_inference,
    plot_embedding,
)

import e2cnn.nn as gnn

import matplotlib.pyplot as plt

# %%

embedding_kernel = SeparableKernel(2, 3, RBFKernel(2, 1.0))
grid_ranges = [-4.0, 4.0]
n_axes = 20
normalise = True
cnn = get_e2_decoder(4, False, "regular_small", [1], [1], activation="normrelu")
output_kernel = SeparableKernel(2, 6, RBFKernel(2, 1.0))
dim = 2

deepset = EquivDeepSet(
    grid_ranges=grid_ranges,
    n_axes=n_axes,
    embedding_kernel=embedding_kernel,
    normalise_embedding=normalise,
    normalise_output=normalise,
    cnn=cnn,
    output_kernel=output_kernel,
    dim=dim,
)
# %%

X_target = grid_2d(*grid_ranges, 20).unsqueeze(0)

X_context = torch.Tensor([[1, 2], [2, 1], [-1, -1]]).unsqueeze(0)
Y_context = torch.Tensor([[1, 1], [1, -2], [-2, 3]]).unsqueeze(0)

plot_vector_field(X_context.squeeze(), Y_context.squeeze())
plt.xlim([-4, 4])
plt.ylim([-4, 4])

# %%

X_enc, Y_enc = deepset.encoder((X_context, Y_context))

plot_embedding(
    X_context.squeeze(), Y_context.squeeze(), X_enc.squeeze(), Y_enc.squeeze()
)

# %%

Y_target = deepset(X_context, Y_context, X_target).detach()

plot_inference(
    X_context.squeeze(),
    Y_context.squeeze(),
    X_target.squeeze(),
    mean_prediction=Y_target[:, :, :2].squeeze(),
)
plt.xlim([-4, 4])
plt.ylim([-4, 4])


# %%

gspace = cnn[1].in_type.gspace
input_type = gnn.FieldType(gspace, [gspace.irrep(1)])

# %%

for g in input_type.testing_elements:
    R = torch.tensor(input_type.representation(g), dtype=X_context.dtype)
    X_context_t = X_context @ R.t()
    Y_context_t = Y_context @ R.t()
    X_target_t = X_target @ R.t()

    X_enc_t, Y_enc_t = deepset.encoder((X_context_t, Y_context_t))

    plot_embedding(
        X_context_t.squeeze(),
        Y_context_t.squeeze(),
        X_enc_t.squeeze(),
        Y_enc_t.squeeze(),
    )
    plt.title(f"{g}")

# %%

Y_mean = deepset(X_context, Y_context, X_target)
Y_mean = Y_mean[:, :, :2].detach().squeeze()
# Y_cov = Y_cov.detach().squeeze()

for g in input_type.testing_elements:
    R = torch.tensor(input_type.representation(g), dtype=X_context.dtype)
    X_context_t = X_context @ R.t()
    Y_context_t = Y_context @ R.t()
    X_target_t = X_target @ R.t()

    Y_mean_t = deepset(X_context_t, Y_context_t, X_target_t)
    Y_mean_t = Y_mean_t[:, :, :2].detach().squeeze()
    # Y_cov_t = Y_cov_t.detach().squeeze()

    plot_inference(
        X_context_t.squeeze() @ R,
        Y_context_t.squeeze() @ R,
        X_target_t.squeeze() @ R,
        mean_prediction=Y_mean_t.squeeze() @ R,
        covariance_prediction=None,
        ellipse_scale=0.05,
    )
    # plot_inference(
    #     X_context_t.squeeze(),
    #     Y_context_t.squeeze(),
    #     X_target_t.squeeze(),
    #     mean_prediction=Y_mean_t.squeeze(),
    #     covariance_prediction=None,
    #     ellipse_scale=0.05,
    # )
    plt.title(f"{g}")
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    ax = plt.gca()
    ax.set_aspect("equal")

    # print((Y_mean @ R.t() - Y_mean_t).sum())
    # print(((R @ Y_cov @ R.t()) - Y_cov_t).sum())
# %%

Y_mean = cnp(X_context, Y_context, X_target)
Y_mean = Y_mean.detach().squeeze()
# Y_cov = Y_cov.detach().squeeze()

for g in input_type.testing_elements:
    R = torch.tensor(input_type.representation(g), dtype=X_context.dtype)
    X_context_t = X_context @ R.t()
    Y_context_t = Y_context @ R.t()
    X_target_t = X_target @ R.t()
    Y_mean_t = Y_mean @ R.t()
    Y_cov_t = R @ Y_cov @ R.t()

    # Y_mean_t, Y_cov_t = cnp(X_context_t, Y_context_t, X_target_t)
    # Y_mean_t = Y_mean_t.detach().squeeze()
    # Y_cov_t = Y_cov_t.detach().squeeze()

    plot_inference(
        X_context_t.squeeze(),
        Y_context_t.squeeze(),
        X_target_t.squeeze(),
        mean_prediction=Y_mean_t.squeeze() @ R,
        covariance_prediction=R.t() @ Y_cov_t.squeeze() @ R,
        ellipse_scale=0.05,
    )
    plt.title(f"{g}")
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    ax = plt.gca()
    ax.set_aspect("equal")

# %%

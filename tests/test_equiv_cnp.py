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

from equiv_cnp.equiv_cnp import EquivCNP
from equiv_cnp.covariance_activations import quadratic_covariance_activation
from equiv_cnp.utils import (
    get_e2_decoder,
    grid_2d,
    plot_vector_field,
    plot_inference,
    plot_embedding,
    plot_mean_cov,
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

# %%

cnp = EquivCNP(
    prediction_dim=2,
    covariance_activation_function=quadratic_covariance_activation,
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

X_enc, Y_enc = cnp.encoder((X_context, Y_context))

plot_embedding(
    X_context.squeeze(), Y_context.squeeze(), X_enc.squeeze(), Y_enc.squeeze()
)

# %%

Y_mean, Y_cov = cnp(X_context, Y_context, X_target)
Y_mean = Y_mean.detach()
Y_cov = Y_cov.detach()

plot_inference(
    X_context.squeeze(),
    Y_context.squeeze(),
    X_target.squeeze(),
    mean_prediction=Y_mean.squeeze(),
    covariance_prediction=Y_cov.squeeze(),
)
plt.xlim([-4, 4])
plt.ylim([-4, 4])

# %%

gspace = cnn[1].in_type.gspace
input_type = gnn.FieldType(gspace, [gspace.irrep(1)])

# %%
def test_equivaraince_example(model, X_context, Y_context, X_target, in_field_type):
    n_test = len(list(in_field_type.testing_elements))
    dtype = X_context.dtype
    device = X_context.device

    # mean_diffs = torch.zeros(n_test, device=device)
    # cov_diffs = torch.zeros(n_test, device=device)

    mean_diffs_normed = torch.zeros(n_test, device=device)
    cov_diffs_normed = torch.zeros(n_test, device=device)

    Y_mean, Y_cov = model(
        X_context.unsqueeze(0), Y_context.unsqueeze(0), X_target.unsqueeze(0)
    )
    Y_mean = Y_mean.squeeze(0)
    Y_cov = Y_cov.squeeze(0)

    Y_mean_norm = Y_mean.abs().sum()
    Y_cov_norm = Y_cov.abs().sum()

    for i, g in enumerate(in_field_type.testing_elements):
        R = torch.Tensor(in_field_type.representation(g), device=device)

        X_context_t = X_context @ R.t()
        Y_context_t = Y_context @ R.t()
        X_target_t = X_target @ R.t()

        Y_mean_t, Y_cov_t = model(
            X_context_t.unsqueeze(0), Y_context_t.unsqueeze(0), X_target_t.unsqueeze(0)
        )
        Y_mean_t = Y_mean_t.squeeze(0)
        Y_cov_t = Y_cov_t.squeeze(0)

        mean_diff = Y_mean_t - (Y_mean @ R.t())
        cov_diff = Y_cov_t - (R @ Y_cov @ R.t())

        # mean_diffs[i] = mean_diff.sum(-1).mean()
        # cov_diffs[i] = cov_diff.sum(-1).mean()

        mean_diffs_normed[i] = mean_diff.abs().sum() / Y_mean_norm
        cov_diffs_normed[i] = cov_diff.abs().sum() / Y_cov_norm

    return mean_diffs_normed.detach(), cov_diffs_normed.detach()


# %%
mean_error, cov_error = test_equivaraince_example(
    cnp, X_context.squeeze(), Y_context.squeeze(), X_target.squeeze(), input_type
)

# %%
plt.plot(mean_error, label="mean error")
plt.plot(cov_error, label="cov error")
plt.legend()

# %%

for g in input_type.testing_elements:
    R = torch.tensor(input_type.representation(g), dtype=X_context.dtype)
    X_context_t = X_context @ R.t()
    Y_context_t = Y_context @ R.t()
    X_target_t = X_target @ R.t()

    X_enc_t, Y_enc_t = cnp.encoder((X_context_t, Y_context_t))

    plot_embedding(
        X_context_t.squeeze(),
        Y_context_t.squeeze(),
        X_enc_t.squeeze(),
        Y_enc_t.squeeze(),
    )
    plt.title(f"{g}")

# %%

Y_mean, Y_cov = cnp(X_context, Y_context, X_target)
Y_mean = Y_mean.detach().squeeze()
Y_cov = Y_cov.detach().squeeze()

for g in input_type.testing_elements:
    R = torch.tensor(input_type.representation(g), dtype=X_context.dtype)
    X_context_t = X_context @ R.t()
    Y_context_t = Y_context @ R.t()
    X_target_t = X_target @ R.t()

    Y_mean_t, Y_cov_t = cnp(X_context_t, Y_context_t, X_target_t)
    Y_mean_t = Y_mean_t.detach().squeeze()
    Y_cov_t = Y_cov_t.detach().squeeze()

    print((Y_cov - R.t() @ Y_cov_t.squeeze() @ R).sum())

    plot_inference(
        X_context_t.squeeze() @ R,
        Y_context_t.squeeze() @ R,
        X_target_t.squeeze() @ R,
        mean_prediction=Y_mean_t.squeeze() @ R,
        covariance_prediction=R.t() @ Y_cov_t.squeeze() @ R,
        ellipse_scale=0.05,
    )
    plt.title(f"{g}")
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    ax = plt.gca()
    ax.set_aspect("equal")

    # print((Y_mean @ R.t() - Y_mean_t).sum())
    # print(((R @ Y_cov @ R.t()) - Y_cov_t).sum())
# %%

Y_mean, Y_cov = cnp(X_context, Y_context, X_target)
Y_mean = Y_mean.detach().squeeze()
Y_cov = Y_cov.detach().squeeze()

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
        X_context_t.squeeze() @ R,
        Y_context_t.squeeze() @ R,
        X_target_t.squeeze() @ R,
        mean_prediction=Y_mean_t.squeeze() @ R,
        covariance_prediction=R.t() @ Y_cov_t.squeeze() @ R,
        ellipse_scale=0.1,
    )
    plt.title(f"{g}")
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    ax = plt.gca()
    ax.set_aspect("equal")

# %%

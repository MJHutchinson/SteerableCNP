import math

import torch

import numpy as np

from einops import rearrange

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.tri as tri
import matplotlib

matplotlib.rcParams["text.usetex"] = True


def plot_scalar_field(
    X,
    Y,
    ax=None,
    colormap="viridis",
    zorder=1,
    n_axis=50,
    levels=8,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    X_1, X_2 = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), num=n_axis),
        np.linspace(X[:, 1].min(), X[:, 1].max(), num=n_axis),
    )

    triang = tri.Triangulation(X[:, 0], X[:, 1])
    interpolator = tri.LinearTriInterpolator(triang, Y)
    Z = interpolator(X_1, X_2)

    ax.contourf(X_1, X_2, Z, cmap=colormap, zorder=zorder, levels=levels)

    return ax


def plot_vector_field(
    X, Y, ax=None, color=None, scale=15, width=None, label=None, zorder=1
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    if color is None:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            np.hypot(Y[:, 0], Y[:, 1]),
            scale=scale,
            width=width,
            label=label,
            zorder=zorder,
            pivot="mid",
        )
    else:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            color=color,
            scale=scale,
            width=width,
            label=label,
            zorder=zorder,
            pivot="mid",
        )

    return ax


def plot_covariances(
    X,
    covariances,
    ax=None,
    alpha=0.5,
    color="cyan",
    edgecolor="k",
    scale=0.8,
    label=None,
    zorder=0,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
        x_lim = None
        y_lim = None
    else:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

    for i in range(X.size(0)):
        A = covariances[i]
        if len(A.size()) == 1:
            A = torch.diag(A)

        eigen_decomp = torch.eig(A, eigenvectors=True)
        u = eigen_decomp[1][:, 0]

        angle = 360 * torch.atan(u[1] / u[0]) / (2 * math.pi)

        if (eigen_decomp[0][:, 0] < 0).sum() > 0:
            print("Error: Ill conditioned covariance in plot. Skipping")
            continue

        # Get the width and height of the ellipses (eigenvalues of A):
        D = torch.sqrt(eigen_decomp[0][:, 0])

        # Plot the Ellipse:
        E = Ellipse(
            xy=X[
                i,
            ].numpy(),
            width=scale * D[0],
            height=scale * D[1],
            angle=angle,
            color=color,
            linewidth=1,
            alpha=alpha,
            # edgecolor=edgecolor,
            # facecolor="none",
            zorder=zorder,
        )
        ax.add_patch(E)

    if label is not None:
        label_ellipse = Ellipse(
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            label=label,
            xy=0,
            width=1,
            height=1,
        )


def plot_inference(
    X_context,
    Y_context,
    X_prediction,
    mean_prediction=None,
    covariance_prediction=None,
    title="",
    size_scale=2,
    ellipse_scale=0.8,
    quiver_scale=60,
    ax=None,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    plot_vector_field(
        X_context, Y_context, ax=ax, color="red", scale=quiver_scale, zorder=2
    )
    if mean_prediction is not None:
        plot_vector_field(
            X_prediction, mean_prediction, ax=ax, scale=quiver_scale, zorder=1
        )
    if covariance_prediction is not None:
        plot_covariances(
            X_prediction, covariance_prediction, ax, scale=ellipse_scale, zorder=0
        )


def plot_mean_cov(
    X,
    Y_mean,
    Y_cov,
    title="",
    size_scale=2,
    ellipse_scale=0.8,
    quiver_scale=60,
    ax=None,
):

    if ax == None:
        fig, ax = plt.subplots(1, 1)
    plot_vector_field(X, Y_mean, ax=ax, scale=quiver_scale, zorder=1)
    plot_covariances(X, Y_cov, ax=ax, scale=ellipse_scale, zorder=0)


def plot_embedding(
    X_context,
    Y_context,
    X_embed,
    Y_embed,
    ax=None,
    quiver_scale=60,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    plot_vector_field(
        X_context, Y_context, ax=ax, color="red", scale=quiver_scale, zorder=2
    )
    plot_vector_field(X_embed, Y_embed[:, 1:], ax=ax, scale=quiver_scale, zorder=1)

    n, _ = X_embed.shape

    X_embed = rearrange(
        X_embed, "(m1 m2) d -> m1 m2 d", m1=int(math.sqrt(n)), m2=int(math.sqrt(n))
    )
    Y_embed = rearrange(
        Y_embed, "(m1 m2) d -> m1 m2 d", m1=int(math.sqrt(n)), m2=int(math.sqrt(n))
    )

    ax.contourf(
        X_embed[:, :, 0],
        X_embed[:, :, 1],
        Y_embed[:, :, 0],
        levels=14,
        linewidths=0.5,
        cmap=cm.get_cmap("viridis"),
        zorder=0,
    )

    ax.set_aspect("equal")


def points_to_partial_img(img_size, x_points, y_points, fill_color=[0.0, 0.0, 1.0]):
    img = np.zeros([img_size, img_size, 3])
    x_points = x_points.astype(int)

    if len(y_points.shape) == 1:
        y_points = np.repeat(y_points[:, np.newaxis], 3, axis=1)

    img[:, :, 0] = fill_color[0]
    img[:, :, 1] = fill_color[1]
    img[:, :, 2] = fill_color[2]

    for point, color in zip(x_points, y_points):
        img[point[1], point[0]] = color

    return img


def points_to_img(img_size, x_points, y_points):
    img = np.zeros([img_size, img_size])
    x_points = x_points.astype(int)

    for point, val in zip(x_points, y_points):
        img[point[1], point[0]] = val

    return img


def plot_image_compleation(
    X_context,
    Y_context,
    X_target,
    Y_target,
    Y_pred_mean,
    Y_pred_cov,
    img_size,
    fill_color=[0.0, 0.0, 1.0],
):
    fig, axs = plt.subplots(2, 3)

    context_img = points_to_partial_img(
        img_size,
        X_context,
        Y_context,
        fill_color,
    )
    target_img = points_to_img(img_size, X_target, Y_target)
    mean_img = points_to_img(
        img_size,
        X_target,
        Y_pred_mean,
    )
    var_img = points_to_img(
        img_size,
        X_target,
        Y_pred_cov,
    )
    mean_diff_img = np.abs(mean_img - target_img)

    axs[0][0].imshow(context_img)
    axs[0][0].set_title("Context")

    axs[0][1].imshow(mean_img, cmap="gray", vmin=0, vmax=1)
    axs[0][1].set_title("Mean")

    im = axs[0][2].imshow(var_img, cmap="viridis")
    axs[0][2].set_title("Var")
    fig.colorbar(im, ax=axs[0][2])

    axs[1][1].imshow(target_img, cmap="gray", vmin=0, vmax=1)
    axs[1][1].set_title("Target")

    im = axs[1][2].imshow(mean_diff_img, cmap="viridis")
    axs[1][2].set_title("Mean - Target")
    fig.colorbar(im, ax=axs[1][2])

    axs[1][0].axis("off")

    plt.tight_layout()

    return fig, axs

import math

import torch

from einops import rearrange

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm


def plot_vector_field(X, Y, ax=None, color="black", scale=15, label=None, zorder=1):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    if label is None:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            color=color,
            scale=scale,
            zorder=zorder,
        )
    else:
        ax.quiver(
            X[:, 0],
            X[:, 1],
            Y[:, 0],
            Y[:, 1],
            color=color,
            scale=scale,
            label=label,
            zorder=zorder,
        )


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

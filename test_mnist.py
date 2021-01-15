# %%
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

torch.autograd.set_detect_anomaly(True)

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from equiv_cnp.lightning import (
    LightningImageEquivCNP,
    LightningMNISTDataModule,
    ImageCompleationPlotCallback,
    interpolate_filename,
)
from equiv_cnp.datasets import MNISTDataset, MultiMNIST
from equiv_cnp.utils import plot_image_compleation, points_to_img, points_to_partial_img
import matplotlib.pyplot as plt

import PIL
from PIL import Image

# %%
train_dataset = MultiMNIST(
    root=os.path.join("/data/ziz/not-backed-up/mhutchin/EquivCNP/data/", "mnist"),
    min_context_fraction=0.01,
    max_context_fraction=0.5,
    n_points_fraction=1.0,
    train=True,
    translate=False,
    rotate=False,
    n_digits=1,
    canvas_multiplier=1,
)

test_dataset = MultiMNIST(
    root=os.path.join("/data/ziz/not-backed-up/mhutchin/EquivCNP/data/", "mnist"),
    min_context_fraction=0.01,
    max_context_fraction=0.5,
    n_points_fraction=1.0,
    train=False,
    translate=True,
    rotate=True,
    n_digits=2,
    canvas_multiplier=3,
)

Image.fromarray(
    (test_dataset.data[np.random.randint(len(test_dataset)), 0] * 255)
    .numpy()
    .astype("uint8")
)
# %%
checkpoint = "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/T2-huge/multimnist_sweep_symmetry_/0/checkpoints/last.ckpt"
# checkpoint = "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MultiMNIST/C4-regular_huge/multimnist_sweep_symmetry_/0/checkpoints/last.ckpt"
# checkpoint = (
#     "logs/image/MNIST/C4-regular_huge/mnist_sweep_symmetry_/0/checkpoints/last.ckpt"
# )

model = LightningImageEquivCNP.load_from_checkpoint(
    checkpoint, map_location=torch.device("cpu"), strict=False
)
# %%
model = model.eval()
# %%


def test_contexts(model, X_contexts, Y_contexts):
    means = []
    covs = []
    n = X.shape[0]

    for i in range(len(X_contexts)):
        X_context = X_contexts[i]
        Y_context = Y_contexts[i]

        m, c = model(X_context.unsqueeze(0), Y_context.unsqueeze(0), X.unsqueeze(0))
        means.append(m.detach().cpu().squeeze(0).numpy())
        covs.append(c.detach().cpu().squeeze(0).numpy())

    return (
        means,
        covs,
    )


def plot_means(means, xs, titles=None, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, len(means), figsize=(len(means) * 1, 1))

    for i in range(len(means)):
        axs[i].imshow(
            points_to_img(int(math.sqrt(means[i].shape[0])), xs, means[i]),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])

        if titles is not None:
            axs[i].set_title(titles[i])

    if titles is not None:
        plt.tight_layout()


def plot_contexts(context_xs, context_ys, titles=None, img_size=28, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, len(context_xs), figsize=(len(context_xs) * 1, 1))

    for i in range(len(context_xs)):
        # axs[i].imshow(
        #     points_to_partial_img(img_size, context_xs[i], context_ys[i]),
        #     cmap="gray",
        #     vmin=0,
        #     vmax=1,
        # )
        # print((28.0 / float(img_size)) ** 2)
        axs[i].scatter(
            context_xs[i][:, 0],
            img_size - context_xs[i][:, 1],
            s=1,
            c=context_ys[i],
            marker="s",
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axs[i].set_facecolor([0, 0, 1])
        axs[i].set_aspect("equal")
        # axs[i].invert_yaxis()
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])
        axs[i].set_ylim([0, img_size - 1])
        axs[i].set_xlim([0, img_size - 1])

        if titles is not None:
            axs[i].set_title(titles[i])

    if titles is not None:
        plt.tight_layout()


def plot_compare_means(
    means,
    mean_xs,
    context_xs,
    context_ys,
    mean_titles=None,
    context_titles=None,
    img_size=28,
):
    rows = len(means[0])
    cols = 1 + len(means)

    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * img_size / 28, rows * img_size / 28)
    )

    plot_contexts(
        context_xs,
        context_ys,
        titles=None,
        axs=[axs[j][0] for j in range(rows)],
        img_size=img_size,
    )
    if context_titles is not None:
        for ax, t in zip([axs[j][0] for j in range(rows)], context_titles):
            ax.set_ylabel(t)

    for i in range(len(means)):
        plot_means(means[i], mean_xs, axs=[axs[j][i + 1] for j in range(rows)])

    plt.tight_layout()

    return fig, axs


# %%
dataset = test_dataset

model.equiv_cnp.discrete_rkhs_embedder.set_grid(
    grid_ranges=[
        -3,
        dataset.grid_size + 2,
    ],  # pad the grid with 3 extra points on each edge
    n_axes=dataset.grid_size + 6,
)

X, Y, _ = dataset[3]

n = int(0.3 * X.shape[0])

X_context = X[:n, :].unsqueeze(0)
Y_context = Y[:n, :].unsqueeze(0)
X_target = X.unsqueeze(0)
Y_target = Y.unsqueeze(0)

mean, cov = model(X_context, Y_context, X_target)

plot_image_compleation(
    X_context[0].detach().numpy(),
    Y_context[0].detach().numpy(),
    X_target[0].detach().numpy(),
    Y_target[0].detach().numpy(),
    mean[0].detach().numpy(),
    cov[0].detach().numpy(),
    dataset.grid_size,
)
plt.savefig("test.png", dpi=300)

# plt.savefig("tmp.png")

# %%

percentages = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
Xs = [X[: int(p * X.shape[0])] for p in percentages]
Ys = [Y[: int(p * Y.shape[0])] for p in percentages]
xs = [x.numpy() for x in Xs]
ys = [y.numpy() for y in Ys]

m, c = test_contexts(model, Xs, Ys)
plot_means(m, X.numpy())
plt.savefig("test.png", facecolor="w", dpi=300)
# %%

plot_compare_means([m], X.numpy(), xs, ys, ["test"], percentages, dataset.grid_size)
plt.savefig("percentage.png", facecolor="w", dpi=300)

# %%
def rot_mat(theta):
    theta = torch.tensor(theta, dtype=torch.float32)
    return torch.tensor(
        [[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]]
    )


def rot_points(points, theta, img_size):
    return (
        (points.to(torch.float32) - img_size / 2.0) @ rot_mat(theta).T
    ) + img_size / 2.0


# %%

X_context = X[:n]
Y_context = Y[:n]
angles = np.linspace(0, 2 * np.pi, 16 + 1)[:-1]
Xs = [rot_points(X_context, t, dataset.grid_size) for t in angles]
Ys = len(angles) * [Y_context]
xs = [x.numpy().astype(int) for x in Xs]
ys = [y.numpy() for y in Ys]

m, c = test_contexts(model, Xs, Ys)
plot_means(m, X.numpy())
plt.savefig("test.png")
# %%
plot_compare_means(
    [m],
    X.numpy(),
    xs,
    ys,
    context_titles=[f"{a:0.2f}" for a in angles],
    img_size=dataset.grid_size,
)
plt.savefig("rotate.png", facecolor="w", dpi=300)
# %%

# %%

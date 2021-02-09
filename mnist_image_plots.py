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

from steer_cnp.lightning import (
    LightningImageSteerCNP,
    LightningGP,
    LightningImageCNP,
    LightningMNISTDataModule,
    ImageCompleationPlotCallback,
    interpolate_filename,
)
from steer_cnp.datasets import MNISTDataset, MultiMNIST
from steer_cnp.utils import plot_image_compleation, points_to_img, points_to_partial_img
import matplotlib.pyplot as plt
import matplotlib

from e2cnn import nn as gnn

matplotlib.rcParams["text.usetex"] = True

import PIL
from PIL import Image

# %%
noaug_train_dataset = MultiMNIST(
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

aug_train_dataset = MultiMNIST(
    root=os.path.join("/data/ziz/not-backed-up/mhutchin/EquivCNP/data/", "mnist"),
    min_context_fraction=0.01,
    max_context_fraction=0.5,
    n_points_fraction=1.0,
    train=True,
    translate=False,
    rotate=True,
    n_digits=1,
    canvas_multiplier=1,
)

train_oneshot_dataset = MultiMNIST(
    root=os.path.join("/data/ziz/not-backed-up/mhutchin/EquivCNP/data/", "mnist"),
    min_context_fraction=0.01,
    max_context_fraction=0.5,
    n_points_fraction=1.0,
    train=False,
    translate=False,
    rotate=False,
    n_digits=1,
    canvas_multiplier=2,
)

translate_oneshot_dataset = MultiMNIST(
    root=os.path.join("/data/ziz/not-backed-up/mhutchin/EquivCNP/data/", "mnist"),
    min_context_fraction=0.01,
    max_context_fraction=0.5,
    n_points_fraction=1.0,
    train=False,
    translate=True,
    rotate=False,
    n_digits=2,
    canvas_multiplier=2,
)

single_oneshot_dataset = MultiMNIST(
    root=os.path.join("/data/ziz/not-backed-up/mhutchin/EquivCNP/data/", "mnist"),
    min_context_fraction=0.01,
    max_context_fraction=0.5,
    n_points_fraction=1.0,
    train=False,
    translate=True,
    rotate=False,
    n_digits=2,
    canvas_multiplier=2,
)

full_oneshot_dataset = MultiMNIST(
    root=os.path.join("/data/ziz/not-backed-up/mhutchin/EquivCNP/data/", "mnist"),
    min_context_fraction=0.01,
    max_context_fraction=0.5,
    n_points_fraction=1.0,
    train=False,
    translate=True,
    rotate=True,
    n_digits=2,
    canvas_multiplier=2,
)
# %%
i = np.random.randint(len(aug_train_dataset))
print(i)
Image.fromarray((aug_train_dataset.data[i, 0] * 255).numpy().astype("uint8"))
# %%

fig, axs = plt.subplots(1, 3, figsize=(6, 2))
axs[0].imshow((noaug_train_dataset.data[6, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
axs[1].imshow((aug_train_dataset.data[6, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
axs[2].imshow((full_oneshot_dataset.data[8, 0]).numpy(), cmap="gray", vmin=0, vmax=1)

axs[0].set_title("MNIST")
axs[1].set_title("rotMNIST")
axs[2].set_title("extMNIST")

for ax in axs:
    ax.axis("off")

plt.tight_layout()
plt.savefig("plots/mnist_examples.pdf")
# %%
run = "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/T2-huge/mnist_experiments/dataset.test_args.rotate=False,dataset.train_args.rotate=False,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1"
checkpoint = os.path.join(run, "checkpoints", "last.ckpt")
model = LightningImageSteerCNP.load_from_checkpoint(
    checkpoint, map_location=torch.device("cpu"), strict=False
)
# %%
# runs = {
#     "ConvCNP\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/T2-huge/mnist_experiments/dataset.test_args.rotate=False,dataset.train_args.rotate=False,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
#     "ConvCNP\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/T2-huge/mnist_experiments/dataset.test_args.rotate=True,dataset.train_args.rotate=True,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
#     "SteerCNP($C_4$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C4-regular_huge/mnist_experiments/dataset.test_args.rotate=False,dataset.train_args.rotate=False,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
#     "SteerCNP($C_4$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C4-regular_huge/mnist_experiments/dataset.test_args.rotate=True,dataset.train_args.rotate=True,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
#     "SteerCNP($D_4$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D4-regular_huge/mnist_experiments/dataset.test_args.rotate=False,dataset.train_args.rotate=False,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
#     "SteerCNP($D_4$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D4-regular_huge/mnist_experiments/dataset.test_args.rotate=True,dataset.train_args.rotate=True,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
#     "SteerCNP($C_{16}$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C16-regular_huge/mnist_experiments/dataset.test_args.rotate=False,dataset.train_args.rotate=False,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
#     "SteerCNP($C_{16}$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C16-regular_huge/mnist_experiments/dataset.test_args.rotate=True,dataset.train_args.rotate=True,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True/1",
# }
runs = {
    "ConvCNP\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/T2-huge/mnist_experiments_blanks/dataset.test_args.rotate=False,dataset.train_args.include_blanks=True,dataset.train_args.rotate=False,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($C_4$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C4-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=False,dataset.train_args.include_blanks=True,dataset.train_args.rotate=False,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($C_8$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C8-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=False,dataset.train_args.include_blanks=True,dataset.train_args.rotate=False,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($C_{16}$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C16-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=False,dataset.train_args.include_blanks=True,dataset.train_args.rotate=False,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($D_4$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D4-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=False,dataset.train_args.include_blanks=True,dataset.train_args.rotate=False,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($D_8$)\nMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D8-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=False,dataset.train_args.include_blanks=True,dataset.train_args.rotate=False,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "ConvCNP\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/T2-huge/mnist_experiments_blanks/dataset.test_args.rotate=True,dataset.train_args.include_blanks=True,dataset.train_args.rotate=True,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($C_4$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C4-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=True,dataset.train_args.include_blanks=True,dataset.train_args.rotate=True,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($C_8$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C8-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=True,dataset.train_args.include_blanks=True,dataset.train_args.rotate=True,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($C_{16}$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/C16-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=True,dataset.train_args.include_blanks=True,dataset.train_args.rotate=True,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($D_4$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D4-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=True,dataset.train_args.include_blanks=True,dataset.train_args.rotate=True,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
    "SteerCNP($D_8$)\nrotMNIST": "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/D8-regular_huge/mnist_experiments_blanks/dataset.test_args.rotate=True,dataset.train_args.include_blanks=True,dataset.train_args.rotate=True,min_context_fraction=0.01,model.embedding_kernel_learnable=True,model.output_kernel_learnable=True,model.padding_mode=circular/1",
}

models = {}

models["GP"] = LightningGP.load_from_checkpoint(
    os.path.join(
        "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/GP-RBF/mnist_experiments_more_chol/batch_size=300,model.chol_noise=1e-06,model.kernel_length_scale=1.0,model.kernel_sigma_var=0.05/1",
        "checkpoints",
        "last.ckpt",
    ),
    map_location=torch.device("cpu"),
    strict=False,
)
models["CNP\nMNIST"] = LightningImageCNP.load_from_checkpoint(
    os.path.join(
        "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/CNP/mnist_experiments_blanks/dataset.test_args.rotate=False,dataset.train_args.include_blanks=True,dataset.train_args.rotate=False,min_context_fraction=0.01/1",
        "checkpoints",
        "last.ckpt",
    ),
    map_location=torch.device("cpu"),
    strict=False,
)
models["CNP\nrotMNIST"] = LightningImageCNP.load_from_checkpoint(
    os.path.join(
        "/data/ziz/not-backed-up/mhutchin/EquivCNP/logs/image/MNIST/CNP/mnist_experiments_blanks/dataset.test_args.rotate=True,dataset.train_args.include_blanks=True,dataset.train_args.rotate=True,min_context_fraction=0.01/1",
        "checkpoints",
        "last.ckpt",
    ),
    map_location=torch.device("cpu"),
    strict=False,
)

for k in runs:
    models[k] = LightningImageSteerCNP.load_from_checkpoint(
        os.path.join(runs[k], "checkpoints", "last.ckpt"),
        map_location=torch.device("cpu"),
        strict=False,
    )

for model in models.values():
    model.eval()

for model in models.values():
    if isinstance(model, LightningImageSteerCNP):
        model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 + 2], 28 + 6)

extrapolate_models = models

for model in extrapolate_models.values():
    if isinstance(model, LightningImageSteerCNP):
        model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 * 2 + 2], 28 * 2 + 6)


# %%
def test_contexts(model, X_contexts, Y_contexts, X_target):
    means = []
    covs = []
    n = X_target.shape[0]

    for i in range(len(X_contexts)):
        X_context = X_contexts[i]
        Y_context = Y_contexts[i]

        m, c = model(
            X_context.unsqueeze(0), Y_context.unsqueeze(0), X_target.unsqueeze(0)
        )
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
        rows, cols, figsize=(cols * img_size / 28, rows * img_size / 28), squeeze=False
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

    if mean_titles is not None:
        axs[0][0].set_title("Context")
        for ax, t in zip(axs[0][1:], mean_titles):
            ax.set_title(t, fontsize=8)

    for i in range(len(means)):
        plot_means(means[i], mean_xs, axs=[axs[j][i + 1] for j in range(rows)])

    plt.tight_layout()

    return fig, axs


def rot_mat(theta):
    theta = torch.tensor(theta, dtype=torch.float32)
    return torch.tensor(
        [[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]]
    )


def rot_points(points, theta, img_size):
    return (
        (points.to(torch.float32) - img_size / 2.0) @ rot_mat(theta).T
    ) + img_size / 2.0


def rotation_plot(dataset, idx, models, n_rot, frac=0.3):
    X, Y, _ = dataset[idx]

    # X_context = X[:n]
    # Y_context = Y[:n]
    X_target = X
    Y_target = Y
    angles = np.linspace(0, 2 * np.pi, n_rot + 1)[:-1]

    x_min = int(X.min())
    x_max = int(X.max())
    r = (x_max - x_min) * 1.41421 / 2
    c = (x_max - x_min) / 2

    Xs = []
    Ys = []
    for i, j in itertools.product(
        np.arange(-int(r) - 1, int(r) + 2, 1), np.arange(-int(r) - 1, int(r) + 2, 1)
    ):
        x = int(i + c)
        y = int(j + c)
        if (x > x_max) or (x < x_min) or (y > x_max) or (y < x_min):
            Xs.append(x)
            Ys.append(y)

    Y = torch.Tensor(np.concatenate([Y, np.zeros_like(Xs)[:, np.newaxis]]))
    X = torch.Tensor(np.concatenate([X, np.stack([Xs, Ys], axis=1)]))

    inds = torch.randperm(X.shape[0])
    X = X[inds]
    Y = Y[inds]

    n = int(frac * X.shape[0])

    X_context = X[:n]
    Y_context = Y[:n]

    Xs = []
    Ys = []
    for t in angles:
        rot_x = rot_points(X_context, t, dataset.grid_size)
        inds = (rot_x <= x_max).all(axis=1) & (rot_x >= x_min).all(axis=1)
        Xs.append(rot_x[inds])
        Ys.append(Y_context[inds])

    xs = [x.numpy().astype(int) for x in Xs]
    ys = [y.numpy() for y in Ys]

    means = {
        k: test_contexts(model, Xs, Ys, X_target)[0] for (k, model) in models.items()
    }

    plot_compare_means(
        list(means.values()),
        X_target.numpy(),
        Xs,
        Ys,
        list(means.keys()),
        [f"{i}/{int(len(angles)/2)} $\pi$" for i in range(len(angles))],
        int(math.sqrt(X_target.shape[0])),
    )


def percentage_plot(dataset, idx, models, percentages=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9]):
    X, Y, _ = dataset[idx]
    n = int(X.shape[0])

    X_target = X
    Y_target = Y

    Xs = [X[: int(p * n)] for p in percentages]
    Ys = [Y[: int(p * n)] for p in percentages]
    xs = [x.numpy().astype(int) for x in Xs]
    ys = [y.numpy() for y in Ys]

    means = {
        k: test_contexts(model, Xs, Ys, X_target)[0] for (k, model) in models.items()
    }

    plot_compare_means(
        list(means.values()),
        X.numpy(),
        Xs,
        Ys,
        list(means.keys()),
        [f"{int(p*100)}\%" for p in percentages],
        int(math.sqrt(X.shape[0])),
    )


# %%
pl.seed_everything(3)
percentage_plot(
    noaug_train_dataset,
    25544,
    models,
    percentages=[0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 0.8],
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("plots/train_percentage.pdf")
# %%
pl.seed_everything(3)
percentage_plot(
    aug_train_dataset,
    25544,
    models,
    percentages=[0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 0.8],
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("plots/train_rot_percentage.pdf")
# %%
# 8 39168
pl.seed_everything(3)
rotation_plot(noaug_train_dataset, 25544, models, 8, frac=0.1)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("plots/train_rotation.pdf")
# %%
# sep rot 4 8 5994
# close 5 9 1705
# v sep 8060
pl.seed_everything(3)
percentage_plot(
    full_oneshot_dataset,
    8060,
    extrapolate_models,
    percentages=[0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8],
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("plots/extrapolation_percantage.pdf")
# %%
# sep rot 4 8 5994
# close 5 9 1705
# v sep 8060
pl.seed_everything(3)
rotation_plot(full_oneshot_dataset, 8060, extrapolate_models, 8, frac=0.15)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("plots/extrapolation_rotation.pdf")
# %%
i = np.random.randint(len(full_oneshot_dataset))
print(i)
Image.fromarray((full_oneshot_dataset.data[i, 0] * 255).numpy().astype("uint8"))

# %%
for model in extrapolate_models.values():
    for m in model.modules():
        if isinstance(m, gnn.R2Conv):
            print(m.padding_mode)  # = "circular"

# %%
for model in extrapolate_models.values():
    if isinstance(model, LightningImageSteerCNP):
        model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 * 2 + 2], 28 * 2 + 6)
# %%
for model in extrapolate_models.values():
    model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 + 2], 28 + 6)

# %%
for model in models.values():
    model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 * 2 + 2], 28 * 2 + 6)
# %%
for model in models.values():
    if isinstance(model, LightningImageSteerCNP):
        model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 + 2], 28 + 6)

# %%
for model in blanks_models.values():
    model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 * 2 + 2], 28 * 2 + 6)
# %%
for model in blanks_models.values():
    model.steer_cnp.discrete_rkhs_embedder.set_grid([-3, 28 + 2], 28 + 6)

# %%
# sep rot 4 8 5994
# close 5 9 1705
# v sep 8060
pl.seed_everything(3)
percentage_plot(
    full_oneshot_dataset,
    8060,
    blanks_models,
    percentages=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8],
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.savefig("plots/extrapolation_percantage.pdf")
# %%
id = np.random.randint(len(noaug_train_dataset))
print(id)
percentage_plot(
    noaug_train_dataset,
    id,
    {"cnp": models["CNP\nMNIST"]},
    percentages=[0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 0.8],
)

# 25544
# 25365
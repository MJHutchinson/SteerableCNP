import os
import torch
import torch.utils.data as data

from einops import rearrange

import numpy as np

from steer_cnp.kernel import (
    RBFKernel,
    SeparableKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
)
from steer_cnp.utils import sample_gp_radial_grid_2d

from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from PIL import Image

import pytorch_lightning as pl


class MNISTDataset(MNIST):

    n_classes = 0
    grid_size = 28
    name = "MNIST"

    def __init__(
        self,
        root,
        min_context,
        max_context,
        n_points=None,
        train=True,
        augment=False,
        download=False,
    ):

        super(MNISTDataset, self).__init__(
            root,
            train=train,
            transform=transforms.Compose(
                [
                    transforms.RandomRotation(180),
                    transforms.ToTensor(),
                ]
            )
            if augment
            else transforms.ToTensor(),
            target_transform=None,
            download=download,
        )

        assert n_points <= 28 * 28, f"{n_points=} must be <= {28*28=}"

        xx = torch.arange(0, 28)
        yy = torch.arange(0, 28)

        X, Y = torch.meshgrid(xx, yy)

        self.grid = rearrange(torch.stack([X, Y], dim=-1), "w h d -> (w h) d")
        self.n = 28 * 28

        self.min_context = min_context
        self.max_context = max_context
        self.n_points = n_points if n_points is not None else self.n

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, context_points) where target is index of the target class.
        """
        img = self.data[index]

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        img = rearrange(img, "d h w -> (w h) d")

        if self.min_context == self.max_context:
            n_context = self.max_context
        else:
            n_context = torch.randint(self.min_context, self.max_context, [1]).item()

        shuffle = torch.randperm(self.n)[: self.n_points]
        # shuffle = torch.arange(self.n)[: self.n_points]

        X = self.grid[shuffle]
        Y = img[shuffle]

        return X, Y, n_context

    @staticmethod
    def _collate_fn(batch):
        n_context = batch[0][2]
        X = torch.stack([item[0] for item in batch])
        Y = torch.stack([item[1] for item in batch])

        return (
            X[:, :n_context, :],
            Y[:, :n_context, :],
            X[:, n_context:, :],
            Y[:, n_context:, :],
        )


class MultiMNIST(data.Dataset):
    def __init__(
        self,
        root,
        min_context_fraction,
        max_context_fraction,
        n_points_fraction=None,
        train=True,
        translate=True,
        rotate=True,
        n_digits=2,
        canvas_multiplier=3,
        include_blanks=False,
        seed=0,
    ):
        self.dir = os.path.join(root, self.__class__.__name__)
        self.translate = translate
        self.rotate = rotate
        self.n_digits = n_digits
        self.seed = seed
        self.base_size = 28
        self.grid_size = self.base_size * canvas_multiplier
        self.include_blanks = include_blanks

        data_path = os.path.join(
            self.dir,
            f"{'train' if train else 'test'}_seed_{self.seed}_digits_{self.n_digits}_rotate_{self.rotate}_translate_{self.translate}_size_{canvas_multiplier}_blanks_{self.include_blanks}.pt",
        )

        try:
            self.data = torch.load(data_path)
        except FileNotFoundError:
            if not os.path.exists(self.dir):
                os.mkdir(self.dir)
            mnist = datasets.MNIST(root=root, train=train, download=True)
            self.data = self.make_multi_mnist(
                mnist.data.numpy().astype("uint8"),
                self.seed,
                self.rotate,
                self.translate,
                self.n_digits,
                self.base_size,
                self.grid_size,
                self.include_blanks,
            )
            torch.save(self.data, data_path)

        self.data = self.data.astype(np.float32) / 255.0
        self.data = torch.tensor(self.data).unsqueeze(1).float()

        xx = torch.arange(0, self.grid_size)
        yy = torch.arange(0, self.grid_size)

        X, Y = torch.meshgrid(xx, yy)

        self.grid = rearrange(torch.stack([X, Y], dim=-1), "w h d -> (w h) d")
        self.n = self.grid_size ** 2
        self.min_context = int(self.n * min_context_fraction)
        self.max_context = int(self.n * max_context_fraction)
        self.n_points = (
            int(self.n * n_points_fraction) if n_points_fraction is not None else self.n
        )

    def __len__(self):
        return self.data.size(0)

    def make_multi_mnist(
        self,
        dataset,
        seed,
        rotate,
        translate,
        n_digits,
        base_size,
        canvas_size,
        include_blanks,
    ):
        pl.seed_everything(seed)

        n = dataset.data.shape[0]

        images = np.zeros((n, canvas_size, canvas_size)).astype("uint16")

        max_shift = canvas_size - base_size
        max_rot = 360

        indices = np.stack([np.random.permutation(n) for i in range(n_digits)], axis=1)
        # indices = np.stack([np.arange(n) for i in range(n_digits)], axis=1)
        shifts = (
            np.random.randint(0, max_shift, size=(n, n_digits, 2))
            if translate
            else int((canvas_size - base_size) / 2)
            * np.ones((n, n_digits, 2), dtype=int)
        )
        rots = np.random.uniform(size=(n, n_digits)) * max_rot

        for i in range(n):
            for j in range(n_digits):
                img = Image.fromarray(dataset[indices[i, j]])
                if rotate:
                    img = img.rotate(rots[i, j], resample=Image.BICUBIC)

                images[
                    i,
                    shifts[i, j, 0] : shifts[i, j, 0] + base_size,
                    shifts[i, j, 1] : shifts[i, j, 1] + base_size,
                ] = (
                    np.asarray(img)
                    + images[
                        i,
                        shifts[i, j, 0] : shifts[i, j, 0] + base_size,
                        shifts[i, j, 1] : shifts[i, j, 1] + base_size,
                    ]
                )

        images[images >= 255] = 255

        if include_blanks:
            images = np.concatenate(
                [images, np.zeros((int(0.1 * n), canvas_size, canvas_size))]
            )

        return images.astype("uint8")

    def __getitem__(self, index):
        img = self.data[index]
        img = rearrange(img, "d h w -> (w h) d")

        if self.min_context == self.max_context:
            n_context = self.max_context
        else:
            n_context = torch.randint(self.min_context, self.max_context, [1]).item()

        shuffle = torch.randperm(self.n)[: self.n_points]
        # shuffle = torch.arange(self.n)[: self.n_points]

        X = self.grid[shuffle]
        Y = img[shuffle]

        return X, Y, n_context

    @staticmethod
    def _collate_fn(batch):
        n_context = batch[0][2]
        X = torch.stack([item[0] for item in batch])
        Y = torch.stack([item[1] for item in batch])

        return (
            X[:, :n_context, :],
            Y[:, :n_context, :],
            X[:, n_context:, :],
            Y[:, n_context:, :],
        )
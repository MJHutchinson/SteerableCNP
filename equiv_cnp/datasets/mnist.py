import os
import torch
import torch.utils.data as data

from einops import rearrange

import numpy as np

from equiv_cnp.kernel import (
    RBFKernel,
    SeparableKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
)
from equiv_cnp.utils import sample_gp_radial_grid_2d

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from PIL import Image


class MNISTDataset(MNIST):
    def __init__(
        self,
        root,
        min_context,
        max_context,
        n_points,
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

        self.min_context = min_context
        self.max_context = max_context
        self.n_points = n_points

        assert n_points <= 28 * 28, f"{n_points=} must be <= {28*28=}"

        xx = torch.arange(0, 28)
        yy = torch.arange(0, 28)

        X, Y = torch.meshgrid(xx, yy)

        self.grid = rearrange(torch.stack([X, Y], dim=-1), "w h d -> (w h) d")
        self.n = 28 * 28

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

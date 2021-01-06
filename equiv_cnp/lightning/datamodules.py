import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

from equiv_cnp.datasets import MNISTDataset


class LightningGPDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, splits, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.splits = splits
        self.kwargs = {**{"num_workers": 4, "pin_memory": True}, **kwargs}

    def setup(self, stage=None):
        n = len(self.dataset)
        self.trainset, self.validset, self.testset = random_split(
            self.dataset, [int(i * n) for i in self.splits]
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset._collate_fn,
            **self.kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset._collate_fn,
            **self.kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset._collate_fn,
            **self.kwargs
        )


class LightningMNISTDataModule(pl.LightningDataModule):
    def __init__(self, trainset, testset, batch_size, test_valid_splits, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.trainset = trainset
        self.testset = testset
        self.test_valid_splits = test_valid_splits
        self.kwargs = {**{"num_workers": 4, "pin_memory": True}, **kwargs}

    def setup(self, stage=None):
        n = len(self.testset)
        self.testset, self.validset = random_split(
            self.testset, [int(i * n) for i in self.test_valid_splits]
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.trainset._collate_fn,
            **self.kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.trainset._collate_fn,
            **self.kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.trainset._collate_fn,
            **self.kwargs
        )
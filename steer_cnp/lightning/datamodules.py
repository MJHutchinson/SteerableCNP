import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

from steer_cnp.datasets import MNISTDataset


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
    def __init__(
        self,
        trainset,
        testset,
        batch_size,
        test_valid_splits,
        test_batch_size=None,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = (
            test_batch_size if test_batch_size is not None else batch_size
        )
        self.trainset = trainset
        self.testset = testset
        self.test_valid_splits = test_valid_splits
        self.kwargs = {**{"num_workers": 4, "pin_memory": True}, **kwargs}

    def setup(self, stage=None):
        if isinstance(self.testset, list):
            testset = []
            validset = []
            for dataset in self.testset:
                n = len(dataset)
                ts, vs = random_split(
                    dataset,
                    [
                        int(self.test_valid_splits[0] * n),
                        n - int(self.test_valid_splits[0] * n),
                    ],
                )
                testset.append(ts)
                validset.append(vs)
            self.testset = testset
            self.validset = validset
        else:
            n = len(self.testset)
            self.testset, self.validset = random_split(
                self.testset,
                [
                    int(self.test_valid_splits[0] * n),
                    n - int(self.test_valid_splits[0] * n),
                ],
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
        if isinstance(self.validset, list):
            return [
                DataLoader(
                    vs,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    collate_fn=self.trainset._collate_fn,
                    **self.kwargs
                )
                for vs in self.validset
            ]
        else:
            return DataLoader(
                self.validset,
                batch_size=self.test_batch_size,
                shuffle=False,
                collate_fn=self.trainset._collate_fn,
                **self.kwargs
            )

    def test_dataloader(self):
        if isinstance(self.testset, list):
            return [
                DataLoader(
                    ts,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    collate_fn=self.trainset._collate_fn,
                    **self.kwargs
                )
                for ts in self.testset
            ]
        else:
            return DataLoader(
                self.testset,
                batch_size=self.test_batch_size,
                shuffle=False,
                collate_fn=self.trainset._collate_fn,
                **self.kwargs
            )
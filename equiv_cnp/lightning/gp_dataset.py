import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl


class LightningGPDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, splits, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.splits = splits
        self.kwargs = {**{"num_workers": 4, "pin_memory": True}, **kwargs}

    def setup(self, stage=None):
        n = len(self.dataset)
        self.train_set, self.valid_set, self.test_set = random_split(
            self.dataset, [int(i * n) for i in self.splits]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset._collate_fn,
            **self.kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset._collate_fn,
            **self.kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset._collate_fn,
            **self.kwargs
        )
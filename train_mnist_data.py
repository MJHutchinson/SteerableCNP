import os

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
from equiv_cnp.datasets import MNISTDataset

import hydra


@hydra.main(config_path="config/image", config_name="config")
def main(args):

    # get current dir as set by hydra
    run_path = os.getcwd()
    print(run_path)

    # Build dataset
    if args.dataset.name == "MNIST":
        img_size = 28
        trainset = MNISTDataset(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context=args.min_context,
            max_context=args.max_context,
            n_points=args.total_points,
            train=True,
            augment=args.dataset.augment_train,
            download=True,
            # **args.dataset.dataset_args,
        )
        testset = MNISTDataset(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context=args.min_context,
            max_context=args.max_context,
            n_points=args.total_points,
            train=False,
            augment=args.dataset.augment_test,
            download=True,
            # **args.dataset.dataset_args,
        )
        datamodule = LightningMNISTDataModule(
            trainset=trainset,
            testset=testset,
            batch_size=args.batch_size,
            test_valid_splits=args.test_valid_splits,
        )

    pl.seed_everything(args.seed)

    equiv_cnp = LightningImageEquivCNP(**args.model)

    log_dir = os.path.join(args.paths.logdir, args.dataset.name, equiv_cnp.name)
    run_name = args.experiment_name
    run_dir = os.path.join(log_dir, run_name)
    run_dir = interpolate_filename(run_dir)

    loggers = [
        # Log results to a csv file
        CSVLogger(save_dir="", name="logs", version=""),
        # Log data to tensorboard
        TensorBoardLogger(save_dir="", name="logs", version=""),
    ]

    callbacks = [
        # Callback to save recent + best validation checkpoint
        ModelCheckpoint(
            dirpath="checkpoints",
            monitor="val_ll",
            mode="max",
            save_last=True,
        ),
        # Callback to plot image compleations
        ImageCompleationPlotCallback(img_size=28, dirpath="plots", n_plots=3),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        # default_root_dir=args.paths.logdir,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        flush_logs_every_n_steps=args.flush_logs_every_n_steps,
        val_check_interval=args.val_check_interval,
        gpus=int(torch.cuda.is_available()),
        log_gpu_memory=args.log_gpu_memory,
    )

    trainer.fit(equiv_cnp, datamodule)
    trainer.test(equiv_cnp, datamodule=datamodule)


if __name__ == "__main__":
    main()
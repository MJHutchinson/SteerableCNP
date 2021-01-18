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
from equiv_cnp.datasets import MNISTDataset, MultiMNIST

import hydra

import torch.utils.data


@hydra.main(config_path="config/image", config_name="config")
def main(args):

    # get current dir as set by hydra
    run_path = os.getcwd()
    print(run_path)

    # Build dataset
    if args.dataset.name == "MNIST":
        train_img_size = 28
        test_img_size = 28
        trainset = MNISTDataset(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context=int(args.min_context_fraction * train_img_size ** 2),
            max_context=int(args.max_context_fraction * train_img_size ** 2),
            n_points=int(args.n_points_fraction * train_img_size ** 2),
            train=True,
            augment=args.dataset.augment_train,
            download=True,
            # **args.dataset.dataset_args,
        )
        testset = MNISTDataset(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context=int(args.min_context_fraction * test_img_size ** 2),
            max_context=int(args.max_context_fraction * test_img_size ** 2),
            n_points=int(args.n_points_fraction * test_img_size ** 2),
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
        datamodule.setup()
        args.finetune_epochs = 0
    elif args.dataset.name == "MultiMNIST":
        trainset = MultiMNIST(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context_fraction=args.min_context_fraction,
            max_context_fraction=args.max_context_fraction,
            n_points_fraction=args.n_points_fraction,
            **args.dataset.train_args,
            seed=args.dataset.seed,
        )
        finetuneset = MultiMNIST(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context_fraction=args.min_context_fraction,
            max_context_fraction=args.max_context_fraction,
            n_points_fraction=args.n_points_fraction,
            **args.dataset.finetune_args,
            seed=args.dataset.seed,
        )
        testset = MultiMNIST(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context_fraction=args.min_context_fraction,
            max_context_fraction=args.max_context_fraction,
            n_points_fraction=args.n_points_fraction,
            **args.dataset.test_args,
            seed=args.dataset.seed,
        )
        train_img_size = trainset.grid_size
        finetune_img_size = finetuneset.grid_size
        test_img_size = testset.grid_size
        datamodule = LightningMNISTDataModule(
            trainset=trainset,
            testset=testset,
            batch_size=args.batch_size,
            test_batch_size=int(
                args.batch_size * (train_img_size / test_img_size) ** 2
            ),
            test_valid_splits=args.test_valid_splits,
        )
        datamodule.setup()
    else:
        raise ValueError(f"{args.dataset.name} is not a recognised dataset name")

    pl.seed_everything(args.seed)

    equiv_cnp = LightningImageEquivCNP(
        **args.model,
        grid_ranges=[
            -3,
            testset.grid_size + 2,
        ],  # pad the grid with 3 extra points on each edge
        n_axes=testset.grid_size + 6,
    )

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
        ImageCompleationPlotCallback(dirpath="plots", n_plots=3, train=True),
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
        limit_val_batches=(test_img_size / train_img_size) ** 2
        if not args.debug
        else 3,
        limit_test_batches=1.0 if not args.debug else 3,
        limit_train_batches=1.0 if not args.debug else 3,
    )

    trainer.fit(
        equiv_cnp,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    trainer.test(equiv_cnp, test_dataloaders=datamodule.test_dataloader())
    if args.finetune_epochs > 0:
        trainer.current_epoch = args.epochs
        trainer.max_epochs = args.epochs + args.finetune_epochs
        trainer.fit(
            equiv_cnp,
            train_dataloader=DataLoader(
                finetuneset,
                batch_size=args.finetune_batch_size,
                shuffle=True,
                collate_fn=finetuneset._collate_fn,
            ),
            val_dataloaders=datamodule.val_dataloader(),
        )
        trainer.test(equiv_cnp, test_dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    main()
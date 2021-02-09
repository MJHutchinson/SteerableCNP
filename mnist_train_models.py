import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

torch.autograd.set_detect_anomaly(True)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from steer_cnp.lightning import (
    LightningImageSteerCNP,
    LightningImageCNP,
    LightningGP,
    LightningMNISTDataModule,
    ImageCompleationPlotCallback,
    PercentageCompleationPlotCallback,
    CSVLogger,
    interpolate_filename,
)
from steer_cnp.datasets import MNISTDataset, MultiMNIST

import hydra

import torch.utils.data


@hydra.main(config_path="config/image", config_name="config")
def main(args):

    # get current dir as set by hydra
    run_path = os.getcwd()

    print(args)

    # Check if config run and succeeded
    if os.path.exists(os.path.join(run_path, "success")):
        print("Experiment previously successful.")
        sys.exit(0)

    # Build dataset
    if args.dataset.name == "MNIST":
        train_img_size = 28
        test_img_size = 28
        trainset = MultiMNIST(
            root=os.path.join(args.paths.datadir, "mnist"),
            min_context_fraction=args.min_context_fraction,
            max_context_fraction=args.max_context_fraction,
            n_points_fraction=args.n_points_fraction,
            **args.dataset.train_args,
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

    if "GP" in args.model.name:
        model_class = LightningGP
    elif args.model.name == "CNP":
        model_class = LightningImageCNP
    else:
        model_class = LightningImageSteerCNP

    print(args.pretrained_checkpoint)
    if args.pretrained_checkpoint is not None:
        print(f"Loading pretrained model from {args.pretrained_checkpoint}")
        model = model_class.load_from_checkpoint(
            args.pretrained_checkpoint,
            strict=False,
        )
    else:
        model = model_class(
            **args.model,
        )

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
        PercentageCompleationPlotCallback(dirpath="plots", n_plots=3, train=True),
    ]

    if os.path.exists(os.path.join(run_path, "checkpoints", "last.ckpt")):
        resume_dir = os.path.join(run_path, "checkpoints", "last.ckpt")
        print(f"Resuming from checkpoint {resume_dir}")
    elif args.pretrained_checkpoint is not None:
        resume_dir = args.pretrained_checkpoint
    else:
        resume_dir = None

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
        limit_train_batches=args.limit_train_batches if not args.debug else 3,
        resume_from_checkpoint=resume_dir,
    )
    trainer.fit(
        model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    if args.finetune_epochs > 0:
        trainer.current_epoch = args.epochs
        trainer.max_epochs = args.epochs + args.finetune_epochs
        trainer.fit(
            model,
            train_dataloader=DataLoader(
                finetuneset,
                batch_size=args.finetune_batch_size,
                shuffle=True,
                collate_fn=finetuneset._collate_fn,
            ),
            val_dataloaders=datamodule.val_dataloader(),
        )
        trainer.test(model, test_dataloaders=datamodule.test_dataloader())
    else:
        trainer.test(model, test_dataloaders=datamodule.test_dataloader())

    # Final log flush to enure everything is logged...
    trainer.logger.save()
    # Touch a file to mark run as compleate
    Path(os.path.join(run_path, "success")).touch()


if __name__ == "__main__":
    main()
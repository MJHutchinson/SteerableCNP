import os

import torch
from torch.utils.data import DataLoader, random_split

torch.autograd.set_detect_anomaly(True)

import pytorch_lightning as pl
from pytorch_lightning.loggers import (
    CSVLogger,
    TensorBoardLogger,
)
from pytorch_lightning.callbacks import ModelCheckpoint

from steer_cnp.lightning import (
    LightningSteerCNP,
    LightningGPDataModule,
    InferencePlotCallback,
    GPComparePlotCallback,
    interpolate_filename,
)
from steer_cnp.datasets import GPDataset

import hydra


@hydra.main(config_path="config/gp", config_name="config")
def main(args):

    # get current dir as set by hydra
    run_path = os.getcwd()
    print(run_path)

    # Build dataset
    dataset = GPDataset(
        root=args.paths.datadir,
        min_context=args.min_context,
        max_context=args.max_context,
        n_points=args.total_points,
        **args.dataset,
    )

    pl.seed_everything(args.seed)

    steer_cnp = LightningSteerCNP(**args.model)
    datamodule = LightningGPDataModule(
        dataset, batch_size=args.batch_size, splits=args.splits
    )

    log_dir = os.path.join(args.paths.logdir, dataset.name, steer_cnp.name)
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
        # Callback to plot inferences on validation
        InferencePlotCallback(n_plots=3, dirpath="plots"),
        # Callback to plot comparison of the inference to the GP drawn from
        GPComparePlotCallback(
            n_plots=3,
            dirpath="plots",
            kernel=dataset.get_kernel(),
            obs_noise=dataset.obs_noise,
        ),
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

    trainer.fit(steer_cnp, datamodule)
    trainer.test(steer_cnp, datamodule=datamodule)


if __name__ == "__main__":
    main()
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
    ExperimentWriter,
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

    if not os.path.exists(os.path.join(run_path, "success")) and not (
        "GP" in args.model.name
    ):
        print("Experiment not fully run")
        sys.exit(0)

    if os.path.exists(os.path.join(run_path, "test_success")):
        print("Tests previously successful.")
        sys.exit(0)

    if "GP" in args.model.name:
        resume_dir = None
    elif os.path.exists(os.path.join(run_path, "checkpoints", "last.ckpt")):
        resume_dir = os.path.join(run_path, "checkpoints", "last.ckpt")
        print(f"Resuming from checkpoint {resume_dir}")
    else:
        raise ValueError("Checkpoint file missing.")

    pl.seed_everything(args.seed)

    no_aug_base_testset = MultiMNIST(
        root=os.path.join(args.paths.datadir, "mnist"),
        min_context_fraction=args.min_context_fraction,
        max_context_fraction=args.max_context_fraction,
        n_points_fraction=args.n_points_fraction,
        train=False,
        rotate=False,
        translate=False,
        n_digits=1,
        canvas_multiplier=1,
        seed=args.dataset.seed,
    )
    n = len(no_aug_base_testset)
    no_aug_base_testset = random_split(
        no_aug_base_testset,
        [
            int(args.test_valid_splits[0] * n),
            n - int(args.test_valid_splits[0] * n),
        ],
    )[0]

    aug_base_testset = MultiMNIST(
        root=os.path.join(args.paths.datadir, "mnist"),
        min_context_fraction=args.min_context_fraction,
        max_context_fraction=args.max_context_fraction,
        n_points_fraction=args.n_points_fraction,
        train=False,
        rotate=True,
        translate=False,
        n_digits=1,
        canvas_multiplier=1,
        seed=args.dataset.seed,
    )
    n = len(aug_base_testset)
    aug_base_testset = random_split(
        aug_base_testset,
        [
            int(args.test_valid_splits[0] * n),
            n - int(args.test_valid_splits[0] * n),
        ],
    )[0]

    no_rotate_extrapolate_testset = MultiMNIST(
        root=os.path.join(args.paths.datadir, "mnist"),
        min_context_fraction=args.min_context_fraction,
        max_context_fraction=args.max_context_fraction,
        n_points_fraction=args.n_points_fraction,
        train=False,
        rotate=False,
        translate=True,
        n_digits=2,
        canvas_multiplier=2,
        seed=args.dataset.seed,
    )
    n = len(no_rotate_extrapolate_testset)
    no_rotate_extrapolate_testset = random_split(
        no_rotate_extrapolate_testset,
        [
            int(args.test_valid_splits[0] * n),
            n - int(args.test_valid_splits[0] * n),
        ],
    )[0]

    rotate_extrapolate_testset = MultiMNIST(
        root=os.path.join(args.paths.datadir, "mnist"),
        min_context_fraction=args.min_context_fraction,
        max_context_fraction=args.max_context_fraction,
        n_points_fraction=args.n_points_fraction,
        train=False,
        rotate=True,
        translate=True,
        n_digits=2,
        canvas_multiplier=2,
        seed=args.dataset.seed,
    )
    n = len(rotate_extrapolate_testset)
    rotate_extrapolate_testset = random_split(
        rotate_extrapolate_testset,
        [
            int(args.test_valid_splits[0] * n),
            n - int(args.test_valid_splits[0] * n),
        ],
    )[0]

    pl.seed_everything(args.seed)

    if "GP" in args.model.name:
        model_class = LightningGP
    elif args.model.name == "CNP":
        model_class = LightningImageCNP
    else:
        model_class = LightningImageSteerCNP

    if resume_dir is None:
        model = model_class(**args.model)
        callabacks = [
            ModelCheckpoint(
                dirpath="checkpoints",
                save_last=True,
            ),
            ImageCompleationPlotCallback(dirpath="plots", n_plots=3, train=True),
            PercentageCompleationPlotCallback(dirpath="plots", n_plots=3, train=True),
        ]
        loggers = [
            # Log results to a csv file
            CSVLogger(save_dir="", name="logs", version=""),
        ]
    else:
        model = model_class.load_from_checkpoint(
            resume_dir,
            strict=False,
        )
        callabacks = []
        loggers = []

    # grid = model.steer_cnp.discrete_rkhs_embedder.grid_ranges
    # n_axes = steer_cnp.steer_cnp.discrete_rkhs_embedder.n_axes

    # Load up the trainer per the checkpoint, minus the loggers and callbacks
    trainer = pl.Trainer(
        # max_epochs=args.epochs,
        logger=loggers,
        callbacks=callabacks,
        log_every_n_steps=args.log_every_n_steps,
        flush_logs_every_n_steps=args.flush_logs_every_n_steps,
        val_check_interval=args.val_check_interval,
        gpus=int(torch.cuda.is_available()),
        log_gpu_memory=args.log_gpu_memory,
        resume_from_checkpoint=resume_dir,
    )

    if os.path.exists(os.path.join(run_path, "logs", "test_metrics.csv")):
        os.remove(os.path.join(run_path, "logs", "test_metrics.csv"))

    results_writer = ExperimentWriter(
        os.path.join(run_path, "logs"),
        metrics_file="test_metrics.csv",
        hparams_file="test_hparams.yaml",
    )

    result = trainer.test(
        model,
        test_dataloaders=[
            DataLoader(
                no_aug_base_testset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=no_aug_base_testset.dataset._collate_fn,
                **{"num_workers": 4, "pin_memory": True},
            ),
        ],
    )

    results_writer.log_metrics({"no_aug_base": result[0]["test_ll"]})

    result = trainer.test(
        model,
        test_dataloaders=[
            DataLoader(
                aug_base_testset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=aug_base_testset.dataset._collate_fn,
                **{"num_workers": 4, "pin_memory": True},
            ),
        ],
    )

    results_writer.log_metrics({"aug_base": result[0]["test_ll"]})

    result = trainer.test(
        model,
        test_dataloaders=[
            DataLoader(
                no_rotate_extrapolate_testset,
                batch_size=int(
                    args.batch_size / (16 if "GP" in args.model.name else 4)
                ),
                shuffle=False,
                collate_fn=no_rotate_extrapolate_testset.dataset._collate_fn,
                **{"num_workers": 4, "pin_memory": True},
            ),
        ],
    )

    results_writer.log_metrics({"no_rotate_extrapolate": result[0]["test_ll"]})

    # steer_cnp.steer_cnp.discrete_rkhs_embedder.set_grid(grid, n_axes)
    result = trainer.test(
        model,
        test_dataloaders=[
            DataLoader(
                rotate_extrapolate_testset,
                batch_size=int(
                    args.batch_size / (16 if "GP" in args.model.name else 4)
                ),
                shuffle=False,
                collate_fn=rotate_extrapolate_testset.dataset._collate_fn,
                **{"num_workers": 4, "pin_memory": True},
            ),
        ],
    )

    results_writer.log_metrics({"rotate_extrapolate_testset": result[0]["test_ll"]})

    results_writer.save()

    # Save a GP model for reloading later
    if len(callabacks) > 0:
        callabacks[0].save_checkpoint(trainer, model)
        loggers[0].save()

    Path(os.path.join(run_path, "test_success")).touch()


if __name__ == "__main__":
    main()
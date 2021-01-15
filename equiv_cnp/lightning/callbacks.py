import os
import math

import torch

from pytorch_lightning.callbacks import ModelCheckpoint as BaseModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import CSVLogger as BaseCSVLogger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.cloud_io import get_filesystem

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equiv_cnp.gp import conditional_gp_posterior
from equiv_cnp.utils import (
    plot_inference,
    plot_vector_field,
    plot_covariances,
    plot_image_compleation,
)


import os


def interpolate_filename(filepath):
    version_cnt = 0

    fs = get_filesystem(filepath)

    version = 0
    interpolate_filepath = os.path.join(filepath, str(version))

    while fs.exists(interpolate_filepath):
        version += 1
        interpolate_filepath = os.path.join(filepath, str(version))
        # this epoch called before
    return interpolate_filepath


class ModelCheckpoint(BaseModelCheckpoint):
    """ Force checkpoint to override .ckpt file if existing """

    def _get_metric_interpolated_filepath_name(self, epoch, ckpt_name_metrics):
        return self.format_checkpoint_name(epoch, ckpt_name_metrics)


class CSVLogger(BaseCSVLogger):
    """ """

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
        params = self._convert_params(params)
        self.experiment.log_hparams(params)

    @property
    def log_dir(self):
        return self.root_dir


class PlotCallback(Callback):
    def __init__(self, n_plots, dirpath, train=False, valid=True, test=True):
        super().__init__()
        self.n_plots = n_plots

        self.train = train
        self.test = test
        self.valid = valid

        self.__init_plots_dir(dirpath)

    def make_plots(self, pl_module, batch, trainer, dataset):
        pass

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx == 0) and self.train:
            self.make_plots(pl_module, batch, trainer, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx == 0) and self.valid:
            self.make_plots(pl_module, batch, trainer, "valid")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx == 0) and self.test:
            self.make_plots(pl_module, batch, trainer, "test")

    def _save_plot(self, filepath, trainer):
        if trainer.is_global_zero:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath)

    def __init_plots_dir(self, dirpath):
        self._fs = get_filesystem(str(dirpath))

        if dirpath and self._fs.protocol == "file":
            dirpath = os.path.realpath(dirpath)

        self.dirpath = dirpath or None

    def __resolve_ckpt_dir(self, trainer, pl_module):
        """
        Determines model checkpoint save directory at runtime. References attributes from the
        trainer's logger to determine where to save checkpoints.
        The base path for saving weights is set in this priority:

        1.  Checkpoint callback's path (if passed in)
        2.  The default_root_dir from trainer if trainer has no logger
        3.  The weights_save_path from trainer, if user provides it
        4.  User provided weights_saved_path

        The base path gets extended with logger name and version (if these are available)
        and subfolder "checkpoints".
        """
        if self.dirpath is not None:
            return  # short circuit

        if trainer.logger is not None:
            save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str)
                else f"version_{trainer.logger.version}"
            )

            version, name = trainer.accelerator_backend.broadcast(
                (version, trainer.logger.name)
            )

            ckpt_path = os.path.join(save_dir, name, version, "checkpoints")
        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        self.dirpath = ckpt_path

        if trainer.is_global_zero:
            self._fs.makedirs(self.dirpath, exist_ok=True)


class InferencePlotCallback(PlotCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_plots(self, pl_module, batch, trainer, dataset):
        X_context, Y_context, X_target, Y_target = batch

        if trainer.on_gpu:
            X_context = X_context.to(trainer.root_gpu)
            Y_context = Y_context.to(trainer.root_gpu)
            X_target = X_target.to(trainer.root_gpu)

        # print("INFERENCE PLOT: ", trainer.root_device, trainer.root_gpu, trainer.on_gpu)
        # print("INFERENCE PLOT: ", X_context.device, Y_context.device, X_target.device)

        Y_prediction_mean, Y_prediction_cov = pl_module.forward(
            X_context, Y_context, X_target
        )

        X_context = X_context.cpu().detach()
        Y_context = Y_context.cpu().detach()
        X_target = X_target.cpu().detach()
        Y_target = Y_target.cpu().detach()
        Y_prediction_mean = Y_prediction_mean.cpu().detach()
        Y_prediction_cov = Y_prediction_cov.cpu().detach()

        for i in range(self.n_plots):
            fig, ax = plt.subplots(1, 1)
            plot_inference(
                X_context[i],
                Y_context[i],
                X_target[i],
                Y_prediction_mean[i],
                Y_prediction_cov[i],
                ax=ax,
            )
            plot_vector_field(X_target[i], Y_target[i], ax=ax)

            self._save_plot(
                os.path.join(
                    self.dirpath,
                    f"inference_epoch_{trainer.current_epoch}_{trainer.global_step}_{dataset}_{i}.png",
                ),
                trainer,
            )

            plt.close()


class GPComparePlotCallback(PlotCallback):
    def __init__(self, kernel, obs_noise, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel = kernel
        self.obs_noise = obs_noise

    def make_plots(self, pl_module, batch, trainer, dataset):
        X_context, Y_context, X_target, Y_target = batch

        if trainer.on_gpu:
            X_context = X_context.to(trainer.root_gpu)
            Y_context = Y_context.to(trainer.root_gpu)
            X_target = X_target.to(trainer.root_gpu)

        Y_prediction_mean, Y_prediction_cov = pl_module.forward(
            X_context, Y_context, X_target
        )

        Y_gp_mean, _, Y_gp_cov = conditional_gp_posterior(
            X_context, Y_context, X_target, self.kernel, obs_noise=self.obs_noise
        )

        X_context = X_context.cpu().detach()
        Y_context = Y_context.cpu().detach()
        X_target = X_target.cpu().detach()
        Y_target = Y_target.cpu().detach()
        Y_prediction_mean = Y_prediction_mean.cpu().detach()
        Y_prediction_cov = Y_prediction_cov.cpu().detach()
        Y_gp_mean = Y_gp_mean.cpu().detach()
        Y_gp_cov = Y_gp_cov.cpu().detach()

        for i in range(self.n_plots):
            fig, ax = plt.subplots(1, 1)
            plot_vector_field(
                X_context[i], Y_context[i], scale=50, ax=ax, color="black"
            )

            plot_vector_field(
                X_target[i],
                Y_prediction_mean[i],
                scale=50,
                ax=ax,
                color="blue",
                label="cnp",
            )
            plot_vector_field(
                X_target[i], Y_gp_mean[i], scale=50, ax=ax, color="red", label="gp"
            )

            plot_covariances(
                X_target[i],
                Y_prediction_cov[i],
                ax=ax,
                scale=1.5,
                color="blue",
                alpha=0.5,
            )
            plot_covariances(
                X_target[i], Y_gp_cov[i], ax=ax, scale=1.5, color="red", alpha=0.5
            )

            plt.legend()

            self._save_plot(
                os.path.join(
                    self.dirpath,
                    f"epoch_{trainer.current_epoch}_{trainer.global_step}_{dataset}_{i}.png",
                ),
                trainer,
            )

            plt.close()


class ImageCompleationPlotCallback(PlotCallback):
    def __init__(self, *args, fill_color=[0.0, 0.0, 1.0], **kwargs):
        super().__init__(*args, **kwargs)

        self.fill_color = fill_color

    def make_plots(self, pl_module, batch, trainer, dataset):
        X_context, Y_context, X_target, Y_target = batch

        if trainer.on_gpu:
            X_context = X_context.to(trainer.root_gpu)
            Y_context = Y_context.to(trainer.root_gpu)
            X_target = X_target.to(trainer.root_gpu)
            Y_target = Y_target.to(trainer.root_gpu)

        X_target = torch.cat([X_context, X_target], dim=1)
        Y_target = torch.cat([Y_context, Y_target], dim=1)

        Y_prediction_mean, Y_prediction_cov = pl_module.forward(
            X_context, Y_context, X_target
        )

        X_context = X_context.cpu().numpy()
        Y_context = Y_context.cpu().numpy()
        X_target = X_target.cpu().numpy()
        Y_target = Y_target.cpu().numpy()
        Y_prediction_mean = Y_prediction_mean.detach().cpu().numpy()
        Y_prediction_cov = Y_prediction_cov.detach().cpu().numpy()

        img_size = int(math.sqrt(X_target.shape[1]))

        for i in range(self.n_plots):

            fig, axs = plot_image_compleation(
                X_context[i],
                Y_context[i],
                X_target[i],
                Y_target[i],
                Y_prediction_mean[i],
                Y_prediction_cov[i],
                img_size,
                self.fill_color,
            )

            self._save_plot(
                os.path.join(
                    self.dirpath,
                    f"epoch_{trainer.current_epoch}_{trainer.global_step}_{dataset}_{i}.png",
                ),
                trainer,
            )

            plt.close()

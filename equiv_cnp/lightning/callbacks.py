import os

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
from equiv_cnp.utils import plot_inference, plot_vector_field, plot_covariances


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
    def __init__(self, n_plots, dirpath):
        super().__init__()
        self.n_plots = n_plots

        self.__init_plots_dir(dirpath)

    def make_plots(self, pl_module, batch, trainer):
        pass

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 0:
            self.make_plots(pl_module, batch, trainer)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 0:
            self.make_plots(pl_module, batch, trainer)

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
    def make_plots(self, pl_module, batch, trainer):
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

        X_context = X_context.cpu()
        Y_context = Y_context.cpu()
        X_target = X_target.cpu()
        Y_target = Y_target.cpu()
        Y_prediction_mean = Y_prediction_mean.cpu()
        Y_prediction_cov = Y_prediction_cov.cpu()

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
                    f"inference_epoch_{trainer.current_epoch}_{trainer.global_step}_plot_{i}.png",
                ),
                trainer,
            )

            plt.close()


class GPComparePlotCallback(PlotCallback):
    def __init__(self, kernel, obs_noise, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel = kernel
        self.obs_noise = obs_noise

    def make_plots(self, pl_module, batch, trainer):
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

        X_context = X_context.cpu()
        Y_context = Y_context.cpu()
        X_target = X_target.cpu()
        Y_target = Y_target.cpu()
        Y_prediction_mean = Y_prediction_mean.cpu()
        Y_prediction_cov = Y_prediction_cov.cpu()
        Y_gp_mean = Y_gp_mean.cpu()
        Y_gp_cov = Y_gp_cov.cpu()

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
                    f"epoch_{trainer.current_epoch}_{trainer.global_step}_plot_{i}.png",
                ),
                trainer,
            )

            plt.close()


def points_to_partial_img(img_size, x_points, y_points, fill_color=[0.0, 0.0, 1.0]):
    img = np.zeros([img_size, img_size, 3])
    x_points = x_points.astype(int)

    if len(y_points.shape) == 1:
        y_points = np.repeat(y_points[:, np.newaxis], 3, axis=1)

    img[:, :, 0] = fill_color[0]
    img[:, :, 1] = fill_color[1]
    img[:, :, 2] = fill_color[2]

    for point, color in zip(x_points, y_points):
        img[point[1], point[0]] = color

    return img


def points_to_img(img_size, x_points, y_points):
    img = np.zeros([img_size, img_size])
    x_points = x_points.astype(int)

    for point, val in zip(x_points, y_points):
        img[point[1], point[0]] = val

    return img


class ImageCompleationPlotCallback(PlotCallback):
    def __init__(self, img_size, *args, fill_color=[0.0, 0.0, 1.0], **kwargs):
        super().__init__(*args, **kwargs)

        self.img_size = img_size
        self.fill_color = fill_color

    def make_plots(self, pl_module, batch, trainer):
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
        Y_prediction_mean = Y_prediction_mean.cpu().numpy()
        Y_prediction_cov = Y_prediction_cov.cpu().numpy()

        for i in range(self.n_plots):
            fig, axs = plt.subplots(1, 3)

            context_img = points_to_partial_img(
                self.img_size,
                X_context[i],
                Y_context[i],
                self.fill_color,
            )
            mean_img = points_to_img(
                self.img_size,
                X_target[i],
                Y_prediction_mean[i],
            )
            var_img = points_to_img(
                self.img_size,
                X_target[i],
                Y_prediction_cov[i],
            )

            im = axs[0].imshow(context_img)
            axs[0].set_title("Context")
            # fig.colorbar(im, ax=axs[0])

            im = axs[1].imshow(mean_img, cmap="gray", vimn=0, vmax=1)
            axs[1].set_title("Mean")
            # fig.colorbar(im, ax=axs[1])

            im = axs[2].imshow(var_img, cmap="viridis")
            axs[2].set_title("Var")
            fig.colorbar(im, ax=axs[2])

            plt.tight_layout()

            self._save_plot(
                os.path.join(
                    self.dirpath,
                    f"epoch_{trainer.current_epoch}_{trainer.global_step}_plot_{i}.png",
                ),
                trainer,
            )

            plt.close()

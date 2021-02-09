import torch

from pytorch_lightning.metrics.metric import Metric


def _mean_update(values):
    return values.sum(), values.numel()


def _mean_compute(sum_values, n_values):
    return sum_values / n_values


class Mean(Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values):
        sum_values, n_obs = _mean_update(values)

        self.sum_values += sum_values
        self.total += n_obs

    def compute(self):
        return _mean_compute(self.sum_values, self.total)
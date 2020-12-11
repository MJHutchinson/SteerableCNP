import os
import torch
import torch.utils.data as data

import numpy as np

from equiv_cnp.kernel import (
    RBFKernel,
    SeparableKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
)
from equiv_cnp.utils import sample_gp_radial_grid_2d


class GPDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
        self,
        root,
        min_context,
        max_context,
        n_points,
        kernel_type="rbf",
        length_scale=5.0,
        sigma_var=10.0,
        obs_noise=0.02,
        max_r=20,
        n_grid=30,
        dataset_size=100_000,
        dataset_seed=0,
        create=True,
    ):
        """Load a GP dataset, generating if need be

        Parameters
        ----------
        root : The root data directory to store files in
        min_context : int
            Minimum number of context points per item
        max_context : int
            Maximum number of context points per item
        n_points : int
            Total number of context + target points in a sample
        kernel_type : string
            The type of kernel to use. [rbf/divfree/curlfree]
        length_scale : float
            The length scale of the kernel to use
        sigma_var : float
            The variance of the kernel
        max_r : float
            The radius of the grid to use to sample to GP on
        n_grid : int
            The number of grid points on the diameter
        dataset_size : int
            The number of total datapoints in the dataset
        dataset_seed : int
            The seed to use to generate the dataset
        create : bool
            Create the dataset if it doesn't exists
        """
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = os.path.join(root, "gp", kernel_type)

        self.min_context = min_context
        self.max_context = max_context
        self.n_points = n_points
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.sigma_var = sigma_var
        self.obs_noise = obs_noise
        self.max_r = max_r
        self.n_grid = n_grid
        self.dataset_size = dataset_size
        self.dataset_seed = dataset_seed

        self.file_name = (
            f"{length_scale}_{dataset_size}_{dataset_seed}_{max_r}_{n_grid}"
        )

        if not self.check_exists(self.file_name):
            if create:
                self.create()
            else:
                raise RuntimeError(
                    "Dataset not created. Run with create = True to generate"
                )

        self.X = np.load(os.path.join(self.root, self.file_name + "_X.npy"))
        self.Y = np.load(os.path.join(self.root, self.file_name + "_Y.npy"))

        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

        self.n = self.X.shape[1]

    def __getitem__(self, index):
        X, Y = self.X[index], self.Y[index]

        if self.min_context == self.max_context:
            n_context = self.max_context
        else:
            n_context = torch.randint(self.min_context, self.max_context, [1]).item()

        shuffle = torch.randperm(self.n)[: self.n_points]

        X = X[shuffle]
        Y = Y[shuffle]

        return X, Y, n_context

    @staticmethod
    def _collate_fn(batch):
        n_context = batch[0][2]
        X = torch.stack([item[0] for item in batch])
        Y = torch.stack([item[1] for item in batch])

        return (
            X[:, :n_context, :],
            Y[:, :n_context, :],
            X[:, n_context:, :],
            Y[:, n_context:, :],
        )

    def create(self):
        os.makedirs(os.path.join(self.root), exist_ok=True)

        kernel = self.get_kernel()

        np.random.seed(self.dataset_seed)
        torch.manual_seed(self.dataset_seed)

        X, Y = sample_gp_radial_grid_2d(
            kernel,
            samples=self.dataset_size,
            max_r=self.max_r,
            n_axis=self.n_grid,
            obs_noise=self.obs_noise,
        )

        np.save(os.path.join(self.root, self.file_name + "_X.npy"), X.numpy())
        np.save(os.path.join(self.root, self.file_name + "_Y.npy"), Y.numpy())

    def get_kernel(self):
        if self.kernel_type == "rbf":
            kernel = SeparableKernel(
                2,
                2,
                RBFKernel(2, length_scale=self.length_scale, sigma_var=self.sigma_var),
            )
        elif self.kernel_type == "divfree":
            kernel = RBFDivergenceFreeKernel(
                2, length_scale=self.length_scale, sigma_var=self.sigma_var
            )
        elif self.kernel_type == "curlfree":
            kernel = RBFCurlFreeKernel(
                2, length_scale=self.length_scale, sigma_var=self.sigma_var
            )
        else:
            raise ValueError(
                f"{self.kernel_type} is not a recognised kernel type to use."
            )

        return kernel

    def check_exists(self, file_name):
        return os.path.exists(
            os.path.join(self.root, file_name + "_X.npy")
        ) and os.path.exists(os.path.join(self.root, file_name + "_Y.npy"))

    @property
    def name(self):
        # return f"GP_{self.kernel_type}_{self.length_scale}_{self.sigma_var}_{self.obs_noise}_{self.dataset_size}"
        return f"GP_{self.kernel_type}"

    def __len__(self):
        return self.X.shape[0]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return (
            f"kernel_type: {self.kernel_type}\n"
            f"length_scale: {self.length_scale}\n"
            f"sigma_var: {self.sigma_var}\n"
            f"n_grid: {self.n_grid}\n"
            f"max_r: {self.max_r}\n"
            f"min_context: {self.min_context}\n"
            f"max_context: {self.max_context}\n"
            f"n_points: {self.n_points}\n"
            f"dataset_seed: {self.dataset_seed}\n"
        )
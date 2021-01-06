# %%
from equiv_cnp.datasets import GPDataset
from equiv_cnp.utils import plot_mean_cov, plot_vector_field

from torch.utils.data import DataLoader

# %%
dataset = GPDataset(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/data",
    5,
    50,
    50,
    kernel_type="curlfree",
    length_scale=5.0,
    sigma_var=10.0,
    obs_noise=0.02,
    max_r=10,
    n_grid=30,
    dataset_size=100_000,
    dataset_seed=0,
)
# %%
dataset
# %%

dataloader = DataLoader(dataset, 10, shuffle=True, collate_fn=dataset._collate_fn)

# %%

for data in dataloader:
    print(data[0].shape)
# %%
id = 2

X = dataset.X[id]
Y = dataset.Y[id]

plot_vector_field(X, Y, scale=150)

# %%
id = 5

X, Y, _ = dataset[0]

plot_vector_field(X, Y, scale=150)
# %%

from equiv_cnp.utils import sample_gp_radial_grid_2d, sample_gp_grid_2d

X, Y = sample_gp_radial_grid_2d(
    dataset.get_kernel(), samples=10, max_r=dataset.max_r, n_axis=dataset.n_grid
)

plot_vector_field(X[0], Y[0], scale=150)

# %%

X, Y = sample_gp_grid_2d(
    dataset.get_kernel(),
    samples=100,
    min_x=-dataset.max_r,
    max_x=dataset.max_r,
    n_axis=dataset.n_grid,
)

plot_vector_field(X[1], Y[1], scale=150)

# %%

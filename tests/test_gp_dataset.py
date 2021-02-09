# %%
import numpy as np
import matplotlib.pyplot as plt
from steer_cnp.datasets import GPDataset
from steer_cnp.utils import plot_mean_cov, plot_vector_field

from torch.utils.data import DataLoader

# %%
r = 15
n = 30
curl = GPDataset(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/data",
    5,
    50,
    50,
    kernel_type="curlfree",
    length_scale=5.0,
    sigma_var=10.0,
    obs_noise=0.02,
    max_r=r,
    n_grid=n,
    dataset_size=100_000,
    dataset_seed=0,
)
div = GPDataset(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/data",
    5,
    50,
    50,
    kernel_type="divfree",
    length_scale=5.0,
    sigma_var=10.0,
    obs_noise=0.02,
    max_r=r,
    n_grid=n,
    dataset_size=100_000,
    dataset_seed=0,
)
rbf = GPDataset(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/data",
    5,
    50,
    50,
    kernel_type="rbf",
    length_scale=5.0,
    sigma_var=10.0,
    obs_noise=0.02,
    max_r=r,
    n_grid=n,
    dataset_size=100_000,
    dataset_seed=0,
)
# %%
dataset = curl
id = 43890  # np.random.randint(len(dataset))  # 43567  #68755
print(id)

X = dataset.X[id]
Y = dataset.Y[id]

ax = plot_vector_field(X, Y, scale=100, width=0.006)
ax.set_aspect("equal")
ax.axis("off")
plt.savefig(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/plots/curlfree.pdf", bbox_inches="tight"
)
# %%
dataset = div
id = 95939  # np.random.randint(len(dataset))  # 87387  #
print(id)

X = dataset.X[id]
Y = dataset.Y[id]

ax = plot_vector_field(X, Y, scale=100, width=0.006)
ax.set_aspect("equal")
ax.axis("off")
plt.savefig(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/plots/divfree.pdf", bbox_inches="tight"
)

# %%
dataset = rbf
id = 6921  # np.random.randint(len(dataset))  # 351  #95939  #
print(id)

X = dataset.X[id]
Y = dataset.Y[id]

ax = plot_vector_field(X, Y, scale=100, width=0.006)
ax.set_aspect("equal")
ax.axis("off")
plt.savefig(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/plots/rbf.pdf", bbox_inches="tight"
)

# %%
id = 5

X, Y, _ = dataset[0]

plot_vector_field(X, Y, scale=150)
# %%

from equiv_cnp.utils import sample_gp_radial_grid_2d, sample_gp_grid_2d

X, Y = sample_gp_grid_2d(
    dataset.get_kernel(), samples=10, min_x=-10, max_x=10, n_axis=dataset.n_grid
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
fig, axs = plt.subplots(1, 3)

scale = 100
width = 0.006

dataset = curl
id = 43890  # np.random.randint(len(dataset)) #
ax = axs[2]

X_curl = dataset.X[id]
Y_curl = dataset.Y[id]

plot_vector_field(X_curl, Y_curl, scale=scale, width=width, ax=ax)
ax.set_aspect("equal")
ax.axis("off")
circle = plt.Circle((0, 0), 15.2, color="darkgray", zorder=-2)
circle2 = plt.Circle((0, 0), 15.2, edgecolor="black", facecolor=(0, 0, 0, 0), zorder=3)
ax.add_patch(circle)
ax.add_patch(circle2)
# ax.set_facecolor('lightgray')
# ax.set_title("Curl-free")

dataset = div
id = 95939  # np.random.randint(len(dataset))
ax = axs[1]

X_div = dataset.X[id]
Y_div = dataset.Y[id]

ax = plot_vector_field(X_div, Y_div, scale=scale, width=width, ax=ax)
ax.set_aspect("equal")
ax.axis("off")
circle = plt.Circle((0, 0), 15.2, color="grey", zorder=-2)
circle2 = plt.Circle((0, 0), 15.2, edgecolor="black", facecolor=(0, 0, 0, 0), zorder=3)
ax.add_patch(circle)
ax.add_patch(circle2)

# ax.set_title("Divergence-free")


dataset = rbf
id = 6921  # np.random.randint(len(dataset))
print(id)
ax = axs[0]

X = dataset.X[id]
Y = dataset.Y[id]

# X = (X_div + X_curl) / 2
# Y = (Y_div + Y_curl) / 2

ax = plot_vector_field(X, Y, scale=scale, width=width, ax=ax)
ax.set_aspect("equal")
ax.axis("off")
circle = plt.Circle((0, 0), 15.2, color="lightgray", zorder=-2)
circle2 = plt.Circle((0, 0), 15.2, edgecolor="black", facecolor=(0, 0, 0, 0), zorder=3)
ax.add_patch(circle)
ax.add_patch(circle2)

# ax.set_title("RBF")


plt.tight_layout()

plt.savefig(
    "/data/ziz/not-backed-up/mhutchin/EquivCNP/plots/combined_gp_datasets.pdf",
    bbox_inches="tight",
)

# %%

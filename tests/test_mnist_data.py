# %%
import torch
import numpy as np
from steer_cnp.datasets import MNISTDataset
import matplotlib.pyplot as plt

from steer_cnp.kernel import (
    RBFKernel,
    RBFKernelReparametrised,
    SeparableKernel,
    DotProductKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
    kernel_smooth,
)
from steer_cnp.rkhs_embedding import DiscretisedRKHSEmbedding

# %%

m = MNISTDataset("data/mnist", 7, 390, 784, augment=True, download=True)
# %%

m0 = m[0]

plt.scatter(m0[0][:, 0], m0[0][:, 1], c=m0[1])
# %%
def points_to_partial(img_size, x_points, y_points, fill_color=[0.0, 0.0, 1.0]):
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


# %%
x, y, n_context = m[np.random.randint(1000)]

# %%

# plt.scatter(m0[0][:, 0], m0[0][:, 1], c=m0[1])

img = points_to_partial(28, x[:n_context].numpy(), y[:n_context].numpy())

plt.imshow(img)
# %%

kernel = SeparableKernel(
    2,
    2,
    RBFKernelReparametrised(
        2,
        log_length_scale=torch.tensor(3.0).log(),
        sigma_var=1.0,
    ),
)
embedder = DiscretisedRKHSEmbedding([0, 27], 28, dim=2, kernel=kernel, normalise=True)

# %%

grid, Y_target = embedder(x[:n_context].unsqueeze(0), y[:n_context].unsqueeze(0))
grid = grid.squeeze(0).numpy().astype(int)
Y_target = Y_target.squeeze(0).numpy()
img = points_to_img(28, grid, Y_target[:, 0])
plt.imshow(img)
plt.colorbar()
# %%
img = points_to_img(28, grid, Y_target[:, 1])
plt.imshow(img)
plt.colorbar()
# %%

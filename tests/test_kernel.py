# %%
import torch

from steer_cnp.kernel import (
    RBFKernel,
    SeparableKernel,
    DotProductKernel,
    RBFDivergenceFreeKernel,
    RBFCurlFreeKernel,
    kernel_smooth,
)

# %%

rbf = RBFKernel(3, 1.0)

# %%
X = torch.randn((10, 5, 3))
Y = torch.randn((10, 4, 3))

rbf(X, Y).shape
# %%

sep = SeparableKernel(3, 2, rbf)

# %%

sep(X, Y, flatten=False).shape

# %%

dp = DotProductKernel(3)

# %%

dp(X, Y).shape
# %%

# %%

X = torch.randn((10, 5, 3))
Y = torch.randn((10, 4, 3))

X = X.unsqueeze(-3)
Y = Y.unsqueeze(-2)

ls = 1.0

diff = X - Y

dists = (diff ** 2).sum(dim=-1)

K = torch.exp(-0.5 * dists / ls).unsqueeze(-1).unsqueeze(-1)

outer_product = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
I = torch.eye(3).to(X.device)

A = (outer_product / ls) + (3 - 1 - dists / ls).unsqueeze(-1).unsqueeze(-1) * I

# %%

A = I - (outer_product / ls)

# %%

X_context = torch.Tensor([[1, 2], [2, 1], [-1, -1]])
Y_context = torch.Tensor([1, 3, -1.0]).unsqueeze(-1)

x = torch.arange(-4, 4, step=0.1)
x1, x2 = torch.meshgrid(x, x)
x1 = x1.flatten()
x2 = x2.flatten()

X_target = torch.stack([x1, x2], dim=-1)

Y_target = kernel_smooth(X_context, Y_context, X_target, rbf)

import matplotlib.pyplot as plt

plt.contourf(
    X_target[:, 0].view(80, 80), X_target[:, 1].view(80, 80), Y_target.view(80, 80)
)

plt.scatter(X_context[:, 0], X_context[:, 1], c=Y_context.squeeze())
# %%

kernel = SeparableKernel(2, 2, RBFKernel(2, 1.0))

X_context = torch.Tensor([[1, 2], [2, 1], [-1, -1]])
Y_context = torch.Tensor([[1, 1], [1, -2], [-4, 3]])

x = torch.arange(-4, 4, step=0.5)
x1, x2 = torch.meshgrid(x, x)
x1 = x1.flatten()
x2 = x2.flatten()

X_target = torch.stack([x1, x2], dim=-1)

Y_target = kernel_smooth(X_context, Y_context, X_target, kernel)

import matplotlib.pyplot as plt

plt.contourf(
    X_target[:, 0].view(16, 16),
    X_target[:, 1].view(16, 16),
    Y_target[:, 1].view(16, 16),
)

plt.scatter(X_context[:, 0], X_context[:, 1], c=Y_context[:, 1].squeeze())


# %%

plt.quiver(
    X_target[:, 0], X_target[:, 1], Y_target[:, 0], Y_target[:, 1], color="b", scale=100
)
plt.quiver(
    X_context[:, 0],
    X_context[:, 1],
    Y_context[:, 0],
    Y_context[:, 1],
    color="r",
    scale=100,
)
# %%
kernel = RBFDivergenceFreeKernel(2, 1.0)

X_context = torch.Tensor([[1, 2], [2, 1], [-1, -1]])
Y_context = torch.Tensor([[1, 1], [1, -2], [-4, 3]])

x = torch.arange(-4, 4, step=0.5)
x1, x2 = torch.meshgrid(x, x)
x1 = x1.flatten()
x2 = x2.flatten()

X_target = torch.stack([x1, x2], dim=-1)

Y_target = kernel_smooth(X_context, Y_context, X_target, kernel)

import matplotlib.pyplot as plt

plt.contourf(
    X_target[:, 0].view(16, 16),
    X_target[:, 1].view(16, 16),
    Y_target[:, 1].view(16, 16),
)

plt.scatter(X_context[:, 0], X_context[:, 1], c=Y_context[:, 1].squeeze())


# %%

plt.quiver(
    X_target[:, 0], X_target[:, 1], Y_target[:, 0], Y_target[:, 1], color="b", scale=100
)
plt.quiver(
    X_context[:, 0],
    X_context[:, 1],
    Y_context[:, 0],
    Y_context[:, 1],
    color="r",
    scale=100,
)

# %%
kernel = RBFCurlFreeKernel(2, 1.0)

X_context = torch.Tensor([[1, 2], [2, 1], [-1, -1]])
Y_context = torch.Tensor([[1, 1], [1, -2], [-4, 3]])

x = torch.arange(-4, 4, step=0.5)
x1, x2 = torch.meshgrid(x, x)
x1 = x1.flatten()
x2 = x2.flatten()

X_target = torch.stack([x1, x2], dim=-1)

Y_target = kernel_smooth(X_context, Y_context, X_target, kernel)

import matplotlib.pyplot as plt

plt.contourf(
    X_target[:, 0].view(16, 16),
    X_target[:, 1].view(16, 16),
    Y_target[:, 1].view(16, 16),
)

plt.scatter(X_context[:, 0], X_context[:, 1], c=Y_context[:, 1].squeeze())


# %%

plt.quiver(
    X_target[:, 0], X_target[:, 1], Y_target[:, 0], Y_target[:, 1], color="b", scale=100
)
plt.quiver(
    X_context[:, 0],
    X_context[:, 1],
    Y_context[:, 0],
    Y_context[:, 1],
    color="r",
    scale=100,
)

# %%

X_context = torch.tensor([1, 2, 3, 4.0]).unsqueeze(-1)
Y_context = torch.tensor([1, 2, 3, 4.0]).unsqueeze(-1)

X_target = torch.arange(0, 5, 0.1).unsqueeze(-1)

rbf = RBFKernel(1, 0.50)

Y_target = kernel_smooth(X_context, Y_context, X_target, rbf, normalise=True)

plt.scatter(X_context, Y_context)
plt.plot(X_target, Y_target)


# %%

import torch.nn as nn


def MLP(in_dim, hidden_dim, out_dim, batch_norm=False):
    if not isinstance(hidden_dim, list):
        hidden_dim = [hidden_dim]

    dims = [in_dim] + hidden_dim + [out_dim]
    layers = []

    for i in range(len(dims[:-1])):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU())

    if not batch_norm:
        layers = layers[:-1]
    else:
        layers = layers[:-2]

    return nn.Sequential(*layers)


class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Pass(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x):
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return xs
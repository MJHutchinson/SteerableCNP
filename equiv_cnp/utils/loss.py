import math

import torch


def multivariate_log_likelihood(mean, cov, prediction):
    _, _, d = mean.shape
    diff = mean - prediction
    log_normaliser = -0.5 * torch.log(((2 * math.pi) ** d) * cov.det())
    quadratic_term = (diff.unsqueeze(-2) @ cov.inverse() @ diff.unsqueeze(-1)).squeeze()
    return log_normaliser - 0.5 * quadratic_term

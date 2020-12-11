import torch
import torch.nn as nn

from einops import rearrange


def kernel_smooth(X_context, Y_context, X_target, kernel, normalise=False):
    """
    Inputs: X_Context - torch.tensor -shape (batch_size,n_context_points,D)
            Y_Context - torch.tensor - shape (batch_size,n_context_points,d)
            X_Target - torch.tensor - shape (batch_size,n_target_points,D)
            kernel - equiv_cnp.kernel.kernels.Kernel
    Output:
            Kernel smooth estimates at X_Target
            torch.tensor - shape - (batch_size,n_target_points,d)
    TODO: add normalise option from Peters code
    """
    upranked = False
    if len(X_context.shape) == 2:
        X_context = X_context.unsqueeze(0)
        upranked = True
    if len(Y_context.shape) == 2:
        Y_context = Y_context.unsqueeze(0)
        upranked = True
    if len(X_target.shape) == 2:
        X_target = X_target.unsqueeze(0)
        upranked = True

    _, _, d = Y_context.shape

    K = kernel(X_target, X_context, flatten=True)
    Y_context = rearrange(Y_context, "b n d -> b (n d)")
    Y_target = (K @ Y_context.unsqueeze(-1)).squeeze(-1)
    Y_target = rearrange(Y_target, "b (n d) -> b n d", d=d)

    if normalise:
        K = rearrange(K, "b (n1 d1) (n2 d2) -> b n1 n2 d1 d2", d1=d, d2=d)
        Y_target = (K.sum(dim=2).inverse() @ Y_target.unsqueeze(-1)).squeeze(-1)

    if upranked:
        Y_target = Y_target.squeeze(0)

    return Y_target

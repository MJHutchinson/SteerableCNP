import torch

from einops import rearrange


def sample_gp_prior(X, kernel, samples=1, obs_noise=1e-4, chol_noise=1e-8):
    n, _ = X.shape

    # Compute kernel and add cholesky numerical stability factor
    K = kernel(X, X) + torch.eye(n * kernel.rkhs_dim) * (chol_noise + obs_noise)

    L = K.cholesky()

    z = torch.randn((n * kernel.rkhs_dim, samples))

    Y = (L @ z).squeeze(0)

    Y = rearrange(Y, "(n d) s -> s n d", d=kernel.rkhs_dim)

    if samples == 1:
        Y = Y.squeeze(0)

    return Y


def conditional_gp_posterior(
    X_context, Y_context, X_target, kernel, obs_noise=1e-4, chol_noise=1e-8
):
    """
    Input:
        X_Context - torch.tensor - Shape (n, n_context_points,d)
        Y_Context - torch.tensor- Shape (n, n_context_points,D)
        X_Target - torch.tensor - Shape (n, n_target_points,d)
    Output:
        Means - torch.tensor - Shape (n, n_target_points, D) - Means of conditional dist.
        Cov_Mat- torch.tensor - Shape (n, n_target_points*D,n_target_points*D) - Covariance Matrix of conditional dist.
        Vars - torch.tensor - Shape (n, n_target_points,D) - Variance of individual components
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

    n, c, D = Y_context.shape
    _, t, d = X_target.shape

    # print("GP: ", X_context.device, X_target.device, Y_context.device)

    Kcc_inv = (
        kernel(X_context, X_context)
        + torch.eye(c * kernel.rkhs_dim, device=X_context.device)
        * (chol_noise + obs_noise)
    ).inverse()
    Ktt = kernel(X_target, X_target) + torch.eye(
        t * kernel.rkhs_dim, device=X_context.device
    ) * (chol_noise + obs_noise)
    Ktc = kernel(X_target, X_context)
    Kct = Ktc.transpose(-1, -2)

    mean = Ktc @ Kcc_inv @ rearrange(Y_context, "n c d -> n (c d)").unsqueeze(-1)
    mean = rearrange(mean.squeeze(-1), "n (t d) -> n t d", d=D)
    covariance = Ktt - Ktc @ Kcc_inv @ Kct
    variances = rearrange(covariance, "n (t1 d1) (t2 d2) -> n t1 t2 d1 d2", d1=D, d2=D)
    variances = variances[:, torch.arange(t), torch.arange(t), :, :]

    if upranked:
        mean = mean.squeeze(0)
        covariance = covariance.squeeze(0)
        variances = variances.squeeze(0)

    return mean, covariance, variances

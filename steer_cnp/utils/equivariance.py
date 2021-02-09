import torch


def test_equivaraince_example(model, X_context, Y_context, X_target, in_field_type):
    n_test = len(in_field_type.testing_elements)
    dtype = X_context.dtype
    device = X_context.device

    # mean_diffs = torch.zeros(n_test, device=device)
    # cov_diffs = torch.zeros(n_test, device=device)

    mean_diffs_normed = torch.zeros(n_test, device=device)
    cov_diffs_normed = torch.zeros(n_test, device=device)

    Y_mean, Y_cov = model(
        X_context.unsqueeze(0), Y_context.unsqueeze(0), X_target.unsqueeze(0)
    )
    Y_mean = Y_mean.squeeze(0)
    Y_cov = Y_cov.squeeze(0)

    Y_mean_norm = Y_mean.abs().sum()
    Y_cov_norm = Y_cov.abs().sum()

    for i, g in enumerate(in_field_type.testing_elements):
        R = torch.Tensor(in_field_type.representation(g), device=device)

        X_context_t = X_context @ M.t()
        Y_context_t = Y_context @ M.t()
        X_target_t = X_target @ M.t()

        Y_mean_t, Y_cov_t = model(
            X_context_t.unsqueeze(0), Y_context_t.unsqueeze(0), X_target_t.unsqueeze(0)
        )
        Y_mean_t = Y_mean_t.squeeze(0)
        Y_cov_t = Y_cov_t.squeeze(0)

        mean_diff = Y_mean_t - (Y_mean @ M.t())
        cov_diff = Y_cov_t - (Y_cov @ M.t())

        # mean_diffs[i] = mean_diff.sum(-1).mean()
        # cov_diffs[i] = cov_diff.sum(-1).mean()

        mean_diffs_normed[i] = mean_diff.abs().sum() / Y_mean_norm
        cov_diffs_normed[i] = cov_diff.abs().sum() / Y_cov_norm

    return mean_diffs_normed, cov_diffs_normed

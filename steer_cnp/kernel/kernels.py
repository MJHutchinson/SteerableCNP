import torch
import torch.nn as nn

from einops import rearrange


class Kernel(nn.Module):
    def __init__(self, input_dim, rkhs_dim):
        super().__init__()

        self.input_dim = input_dim
        self.rkhs_dim = rkhs_dim

    def flatten_gram_matrix(self, K):
        """
        Input:
            K - torch.tensor - shape (batch_size,n,m,D_1,D_2)
        Output:
            torch-tensor - shape (batch_size,n*D_1,m*D_2) - block (i,j) of size D_1*D_2 is matrix X[i,j] for i=1,...,n,j=1,...,m
        """

        return rearrange(K, "b n m d1 d2 -> b (n d1) (m d2)")

    def uprank_inputs(self, X, Y):
        """Upranks inputs from non batched to batched.
        Fills in the second argument with the first if missing.
        Unsqueezes the correct dimension for kernel operations.
        """
        upranked = False
        if Y == None:
            Y = X
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
            upranked = True
        if len(Y.shape) == 2:
            upranked = True
            Y = Y.unsqueeze(0)

        return X, Y, upranked

    def forward(self, X, Y=None, flatten=True):
        """
        Input:
        X: torch.tensor
            Shape: (...,n,d)...n number of obs, d...dimension of state space
        Y: torch.tensor or None
            Shape: (...,m,d)...m number of obs, d...dimension of state space

        Output:
        Gram_matrix: torch.tensor
                    Shape (...,n,m,rkhs_dim,rkhs_dim) (if Y is not given (...,n,n,rkhs_dim,rkhs_dim))
                    Block i,j of size DxD gives Kernel value of i-th X-data point and
                    j-th Y data point
        """
        raise NotImplementedError()


class ScalarKernel(Kernel):
    def __init__(self, input_dim):
        super().__init__(input_dim, 1)


class RBFKernel(ScalarKernel):
    def __init__(self, input_dim, length_scale, sigma_var=1):
        super().__init__(input_dim)

        self.length_scale = length_scale
        self.sigma_var = sigma_var

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        X = X.unsqueeze(-2)
        Y = Y.unsqueeze(-3)

        dists = ((X - Y) ** 2).sum(dim=-1)

        K = self.sigma_var * torch.exp(-0.5 * dists / self.length_scale).unsqueeze(
            -1
        ).unsqueeze(-1)

        if flatten:
            K = self.flatten_gram_matrix(K)
        if upranked:
            K = K.squeeze(0)
        return K.squeeze(-1).squeeze(-1)


class RBFKernelReparametrised(ScalarKernel):
    def __init__(self, input_dim, log_length_scale, sigma_var=1):
        super().__init__(input_dim)

        self.log_length_scale = log_length_scale
        self.sigma_var = sigma_var

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        X = X.unsqueeze(-2)
        Y = Y.unsqueeze(-3)

        length_scale = torch.exp(self.log_length_scale)

        dists = ((X - Y) ** 2).sum(dim=-1)

        # print("KERNL: ", dists.device, length_scale.device)

        K = self.sigma_var * torch.exp(-0.5 * dists / length_scale).unsqueeze(
            -1
        ).unsqueeze(-1)

        if flatten:
            K = self.flatten_gram_matrix(K)
        if upranked:
            K = K.squeeze(0)
        return K.squeeze(-1).squeeze(-1)


class DotProductKernel(ScalarKernel):
    def __init__(self, input_dim):
        super().__init__(input_dim)

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        K = X @ Y.transpose(-1, -2)

        if flatten:
            K = (
                self.flatten_gram_matrix(K.unsqueeze(-1).unsqueeze(-1))
                .squeeze(-1)
                .squeeze(-1)
            )
        if upranked:
            K = K.squeeze(0)
        return K


class SeparableKernel(Kernel):
    def __init__(self, input_dim, rkhs_dim, scalar_kernel, B=None):
        super().__init__(input_dim, rkhs_dim)

        if not isinstance(scalar_kernel, ScalarKernel):
            raise ValueError(f"{scalar_kernel} is not a scalar kernel")

        self.scalar_kernel = scalar_kernel

        if B is None:
            B = torch.eye(rkhs_dim)

        self.B = B

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        K = self.scalar_kernel(X, Y, flatten=False)

        K = K.unsqueeze(-1).unsqueeze(-1)

        K = K * self.B.to(K.device)

        if flatten:
            K = self.flatten_gram_matrix(K)
        if upranked:
            K.squeeze(0)

        return K


class RBFDivergenceFreeKernel(Kernel):
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    def __init__(self, dim, length_scale, sigma_var=1):
        super().__init__(dim, dim)

        self.length_scale = length_scale
        self.sigma_var = sigma_var

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        X = X.unsqueeze(-2)
        Y = Y.unsqueeze(-3)

        diff = X - Y

        dists = (diff ** 2).sum(dim=-1)

        K = self.sigma_var * torch.exp(-0.5 * dists / self.length_scale).unsqueeze(
            -1
        ).unsqueeze(-1)

        outer_product = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
        I = torch.eye(self.rkhs_dim).to(X.device)

        A = (outer_product / self.length_scale) + (
            self.rkhs_dim - 1 - dists / self.length_scale
        ).unsqueeze(-1).unsqueeze(-1) * I

        K = A * K

        if flatten:
            K = self.flatten_gram_matrix(K)
        if upranked:
            K = K.squeeze(0)

        return K


class RBFDivergenceFreeKernelReparametrised(Kernel):
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    def __init__(self, dim, log_length_scale, sigma_var=1):
        super().__init__(dim, dim)

        self.log_length_scale = log_length_scale
        self.sigma_var = sigma_var

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        X = X.unsqueeze(-2)
        Y = Y.unsqueeze(-3)

        length_scale = torch.exp(self.log_length_scale)

        diff = X - Y

        dists = (diff ** 2).sum(dim=-1)

        K = self.sigma_var * torch.exp(-0.5 * dists / length_scale).unsqueeze(
            -1
        ).unsqueeze(-1)

        outer_product = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
        I = torch.eye(self.rkhs_dim).to(X.device)

        A = (outer_product / length_scale) + (
            self.rkhs_dim - 1 - dists / length_scale
        ).unsqueeze(-1).unsqueeze(-1) * I

        K = A * K

        if flatten:
            K = self.flatten_gram_matrix(K)
        if upranked:
            K = K.squeeze(0)

        return K


class RBFCurlFreeKernel(Kernel):
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    def __init__(self, dim, length_scale, sigma_var=1):
        super().__init__(dim, dim)

        self.length_scale = length_scale
        self.sigma_var = sigma_var

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        X = X.unsqueeze(-2)
        Y = Y.unsqueeze(-3)

        diff = X - Y

        dists = (diff ** 2).sum(dim=-1)

        K = self.sigma_var * torch.exp(-0.5 * dists / self.length_scale).unsqueeze(
            -1
        ).unsqueeze(-1)

        outer_product = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
        I = torch.eye(self.rkhs_dim).to(X.device)

        A = I - (outer_product / self.length_scale)

        K = A * K

        if flatten:
            K = self.flatten_gram_matrix(K)
        if upranked:
            K = K.squeeze(0)

        return K


class RBFCurlFreeKernelReparametrised(Kernel):
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    def __init__(self, dim, log_length_scale, sigma_var=1):
        super().__init__(dim, dim)

        self.log_length_scale = log_length_scale
        self.sigma_var = sigma_var

    def forward(self, X, Y=None, flatten=True):
        X, Y, upranked = self.uprank_inputs(X, Y)

        X = X.unsqueeze(-2)
        Y = Y.unsqueeze(-3)

        length_scale = torch.exp(self.log_length_scale)

        diff = X - Y

        dists = (diff ** 2).sum(dim=-1)

        K = self.sigma_var * torch.exp(-0.5 * dists / length_scale).unsqueeze(
            -1
        ).unsqueeze(-1)

        outer_product = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
        I = torch.eye(self.rkhs_dim).to(X.device)

        A = I - (outer_product / length_scale)

        K = A * K

        if flatten:
            K = self.flatten_gram_matrix(K)
        if upranked:
            K = K.squeeze(0)

        return K
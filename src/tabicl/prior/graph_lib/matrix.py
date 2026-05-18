import torch

from tabiclv2.prior.graph_lib.base import PriorComponent
from tabiclv2.prior.graph_lib.activation import RandomActivationPlain
from tabiclv2.prior.graph_lib.weights import RandomWeights
from tabiclv2.prior.graph_lib.points import RandomGaussianMixturePoints


def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / (1e-6 + matrix.norm(dim=-1, keepdim=True))


class RandomMatrix(PriorComponent):
    """
    Base class for generating random matrices.
    """
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        if self.config.random_matrix_types == 'default':
            matrix_types = [
                RandomGaussianMatrix,
                RandomWeightsMatrix,
                RandomSingularValuesMatrix,
                RandomKernelMatrix,
                RandomActivationMatrix,
            ]
        elif self.config.random_matrix_types == 'gaussian':
            matrix_types = [RandomGaussianMatrix]
        else:
            raise ValueError(f'Unknown {self.config.random_matrix_types=}')
        matrix_type = self.sampler.choice("random_matrix_type", matrix_types)
        mat = matrix_type(self.context).sample(n_batch, n, m)
        mat += 1e-6 * RandomGaussianMatrix(self.context).sample(n_batch, n, m)
        return row_normalize(mat)


class SimpleRandomMatrix(RandomMatrix):
    """
    Random matrix without having RandomActivationMatrix as one of the options,
    to avoid infinite recursion when this is used inside RandomActivationMatrix.
    """
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        matrix_types = [
            RandomGaussianMatrix,
            RandomWeightsMatrix,
            RandomSingularValuesMatrix,
            RandomKernelMatrix,
        ]
        matrix_type = self.sampler.choice("simple_random_matrix_type", matrix_types)
        mat = matrix_type(self.context).sample(n_batch, n, m)
        mat += 1e-6 * RandomGaussianMatrix(self.context).sample(n_batch, n, m)
        return row_normalize(mat)


class RandomGaussianMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        return torch.randn(n_batch, n, m, device=self.device)


class RandomWeightsMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        matrix = RandomWeights(self.context).sample(n_batch * n, m).reshape(n_batch, n, m)
        # multiply with a Gaussian matrix so we don't always have nonnegative weights that sum to one
        matrix *= torch.randn_like(matrix)
        return row_normalize(matrix)


class RandomSingularValuesMatrix(RandomMatrix):
    # uses random Gaussian instead of random orthogonal matrices
    # still, the resulting distribution of matrices will be invariant to composition with rotations
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        k = min(n, m)
        U = RandomGaussianMatrix(self.context).sample(n_batch, n, k)
        diag = RandomWeights(self.context).sample(n_batch, k)
        V = RandomGaussianMatrix(self.context).sample(n_batch, k, m)

        return (U * diag[:, None, :]) @ V


class RandomKernelMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        d = 3  # arbitrary guess
        points = RandomGaussianMixturePoints(self.context).sample(n_batch * (n + m), d).reshape(n_batch, n + m, d)
        # use Laplace kernel for now
        dists = torch.cdist(points[:, :n], points[:, n:])
        dists *= self.sampler.numerical("random_kernel_matrix_dist_factor", 0.1, 10.0, use_log=True)
        return torch.exp(-dists) * torch.sign(RandomGaussianMatrix(self.context).sample(n_batch, n, m))


class RandomActivationMatrix(RandomMatrix):
    def sample(self, n_batch: int, n: int, m: int) -> torch.Tensor:
        # can't use RandomActivation with batch size 1 due to the standardize operation, so use without standardize
        # use SimpleRandomMatrix since it can't cause infinite recursion
        matrix = RandomActivationPlain(self.context)(
            SimpleRandomMatrix(self.context).sample(n_batch, n, m).reshape(n_batch, n * m)
        ).reshape(n_batch, n, m)
        matrix += 1e-3 * torch.randn_like(matrix)
        return matrix

import torch

from tabicl.prior.graph_lib._base import PriorComponent
from tabicl.prior.graph_lib._weights import RandomWeights


class RandomPoints(PriorComponent):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        # local import to avoid circular imports
        from tabicl.prior.graph_lib._function import RandomFunction

        base_points = self.sampler.choice(
            "random_base_points",
            [RandomUniformPoints, RandomCirclePoints, RandomGaussianPoints, RandomCovariancePoints],
        )
        return RandomFunction(self.context, n, n)(base_points(self.context).sample(n_batch, n))


class RandomCovariancePoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        matrix = torch.randn(n, n) * RandomWeights(self.context).sample(1, n)
        base_points = self.sampler.choice("random_covariance_base_points", [RandomUniformPoints, RandomGaussianPoints])
        return base_points(self.context).sample(n_batch, n) @ matrix.t()


class RandomGaussianPoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        return torch.randn(n_batch, n, device=self.device)


class RandomUniformPoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        return -1 + 2 * torch.rand(n_batch, n, device=self.device)


class RandomCirclePoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        # get angular part from Gaussian points
        # radial density is proportional to r^{n-1}, so radial CDF is F(r) = r^n, inverse CDF is r = u^{1/n}
        x = torch.randn(n_batch, n)
        x /= torch.linalg.vector_norm(x, ord=2, keepdim=True, dim=-1)  # normalize
        u = torch.rand(n_batch, 1)
        r = u ** (1 / n)
        return r * x


class RandomGaussianMixturePoints(RandomPoints):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        n_centers = self.sampler.randint("random_gmp_n_centers", 1, 16, use_log=True)
        weights = RandomWeights(self.context).sample(1, n_centers).squeeze(0)
        # Ensure weights are on the correct device
        weights = weights.to(self.device)

        # Use multinomial with tensor on the correct device
        center_idxs = torch.multinomial(weights, num_samples=n_batch, replacement=True)
        centers = torch.randn(n_centers, n, device=self.device)
        matrices = (
            torch.randn(n_centers, n, n, device=self.device)
            * RandomWeights(self.context).sample(n_centers, n)[:, None, :]
        )
        # multiply by random factors, which themselves have random mean and random std
        matrices = matrices * torch.exp(
            torch.randn(1, device=self.device)
            + torch.randn(1, device=self.device) * torch.randn(n_centers, 1, 1, device=self.device)
        )
        # todo: this can use a large amount of RAM
        points = centers[center_idxs] + (
            matrices[center_idxs] @ torch.randn(n_batch, n, 1, device=self.device)
        ).squeeze(-1)
        return points

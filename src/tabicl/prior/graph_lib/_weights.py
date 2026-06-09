import torch
import numpy as np

from tabicl.prior.graph_lib._base import PriorComponent


class RandomWeights(PriorComponent):
    """
    Base class for sampling random nonnegative "weight" vectors (can be normalized to get probability distributions).
    """
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        if self.config.random_weights_types == 'simple':
            return SimpleRandomWeights(self.context).sample(n_batch, n)
        elif self.config.random_weights_types == 'uniform':
            return UniformRandomWeights(self.context).sample(n_batch, n)
        else:
            raise ValueError(f'Unknown option {self.config.random_weights_types=}')


class SimpleRandomWeights(RandomWeights):
    """
    Use random polynomial decay + gaussian noise on log-values
    """
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        # first model the weights in log-space
        decay_rate = torch.as_tensor(
            self.sampler.numerical(
                "simple_random_weights_decay_rate",
                0.1 / np.log(1 + n),  # n^{-log n} = 1/e
                6,  # up to n^{-6}
                use_log=True,
                boundary_mass=True,
                size=n_batch,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        base_weights = torch.linspace(1.0, n, n, device=self.device)
        log_weights = -decay_rate[:, None] * torch.log(base_weights)
        std_scale = torch.as_tensor(
            self.sampler.numerical("simple_random_weights_std_scale", 0.0001, 10, use_log=True, size=n_batch),
            dtype=torch.float32,
            device=self.device,
        )
        logits = log_weights + std_scale[:, None] * torch.randn(n_batch, n, device=self.device)
        # unfortunately there is no batch randperm (I think), so we have to do the loop
        # for a batched version one can argsort random values instead
        logits = torch.stack([logits[i, torch.randperm(n)] for i in range(n_batch)], dim=0)
        weights = torch.as_tensor(torch.softmax(logits, dim=-1), dtype=torch.float32)
        weights = np.sqrt(n) * (weights / weights.norm(dim=-1, keepdim=True))
        return weights


class UniformRandomWeights(RandomWeights):
    def sample(self, n_batch: int, n: int) -> torch.Tensor:
        return torch.rand(n_batch, n)

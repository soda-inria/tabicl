"""
This module provides functionality for sampling hyperparameters using various probability distributions
and meta-distributions. It's designed to support flexible hyperparameter optimization and configuration
sampling for machine learning models.

Key Components:
1. Basic Distribution Samplers - Functions that create sampling closures for basic distributions
2. HpSampler - A modular class that handles both basic and meta-distribution sampling
3. HpSamplerList - A container class that manages multiple HpSamplers for batch sampling

Meta-distributions are distributions over distribution parameters, allowing for hierarchical sampling
where the parameters of a distribution are themselves sampled from another distribution.
"""

from __future__ import annotations

import math
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn


def trunc_norm_sampler(mu, sigma):
    """Creates a sampler for truncated normal distribution with given mean and std."""
    return lambda: stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]


def beta_sampler(a, b):
    """Creates a sampler for beta distribution with shape parameters a and b."""
    return lambda: np.random.beta(a, b)


def gamma_sampler(a, b):
    """Creates a sampler for gamma distribution with shape parameter a and scale parameter b."""
    return lambda: np.random.gamma(a, b)


def uniform_sampler(a, b):
    """Creates a sampler for uniform distribution between a and b."""
    return lambda: np.random.uniform(a, b)


def uniform_int_sampler(a, b):
    """Creates a sampler for uniform integer distribution between a and b."""
    return lambda: round(np.random.uniform(a, b))


class HpSampler(nn.Module):
    """
    A modular hyperparameter sampler that supports both basic and meta-distributions.

    Meta-distributions include:
    - meta_beta: Beta distribution with sampled parameters
    - meta_gamma: Gamma distribution with sampled parameters
    - meta_trunc_norm: Truncated normal with sampled parameters
    - meta_trunc_norm_log_scaled: Log-scaled truncated normal
    - meta_choice: Categorical distribution with sampled probabilities
    - meta_choice_mixed: Mixed categorical with sampled probabilities

    Parameters:
        distribution (str): Name of the distribution to use
        device (str): Device to use for tensor operations
        **kwargs: Distribution-specific parameters such as:
            - min, max: bounds for uniform distributions
            - scale: scaling factor for beta distribution
            - lower_bound: minimum value for truncated distributions
            - choice_values: possible values for categorical distributions
    """

    def __init__(self, distribution, device, **kwargs):
        super().__init__()
        self.distribution = distribution
        self.device = device
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.initialize_distribution()

    def initialize_distribution(self):
        if self.distribution.startswith("meta"):
            self.initialize_meta_distribution()
        elif self.distribution == "uniform":
            self.sampler = uniform_sampler(self.min, self.max)
        elif self.distribution == "beta":
            self.sampler = beta_sampler(self.a, self.b)
        elif self.distribution == "uniform_int":
            self.sampler = uniform_int_sampler(self.min, self.max)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def initialize_meta_distribution(self):
        if self.distribution == "meta_beta":
            self.sampler = self.setup_meta_beta_sampler()
        elif self.distribution == "meta_gamma":
            self.sampler = self.setup_meta_gamma_sampler()
        elif self.distribution == "meta_trunc_norm":
            self.sampler = self.setup_meta_trunc_norm_sampler()
        elif self.distribution == "meta_trunc_norm_log_scaled":
            self.sampler = self.setup_meta_trunc_norm_log_scaled_sampler()
        elif self.distribution == "meta_choice":
            self.sampler = self.setup_meta_choice_sampler()
        elif self.distribution == "meta_choice_mixed":
            self.sampler = self.setup_meta_choice_mixed_sampler()
        else:
            raise ValueError(f"Unsupported meta distribution: {self.distribution}")

    def ensure_hyperparameter(self, attr_name, distribution, min, max):
        if not hasattr(self, attr_name):
            setattr(
                self,
                attr_name,
                HpSampler(distribution=distribution, device=self.device, min=min, max=max),
            )

    def setup_meta_beta_sampler(self):
        """Sets up a meta-beta distribution sampler.
        Returns a closure that samples beta distribution parameters and then samples from that beta."""
        # Dynamically define b and k if not explicitly provided
        self.ensure_hyperparameter("b", "uniform", self.min, self.max)
        self.ensure_hyperparameter("k", "uniform", self.min, self.max)

        def sampler():
            b = self.b() if callable(self.b) else self.b
            k = self.k() if callable(self.k) else self.k
            return lambda: self.scale * beta_sampler(b, k)()

        return sampler

    def setup_meta_gamma_sampler(self):
        """Sets up a meta-gamma distribution sampler.
        Returns a closure that samples gamma distribution parameters and then samples from that gamma."""
        # Dynamically define alpha and scale if not explicitly provided
        self.ensure_hyperparameter("alpha", "uniform", 0.0, math.log(self.max_alpha))
        self.ensure_hyperparameter("scale", "uniform", 0.0, self.max_scale)

        def sampler():
            alpha = self.alpha() if callable(self.alpha) else self.alpha
            scale = self.scale() if callable(self.scale) else self.scale

            def sub_sampler():
                sample = gamma_sampler(math.exp(alpha), scale / math.exp(alpha))()
                return self.lower_bound + round(sample) if self.round else self.lower_bound + sample

            return sub_sampler

        return sampler

    def setup_meta_trunc_norm_sampler(self):
        """Sets up a meta truncated normal distribution sampler.
        Returns a closure that samples normal distribution parameters and then samples from that normal."""
        # Dynamically define mean and std if not explicitly provided
        self.min_std = self.min_std if hasattr(self, "min_std") else 0.01
        self.max_std = self.max_std if hasattr(self, "max_std") else 1.0
        self.ensure_hyperparameter("mean", "uniform", self.min_mean, self.max_mean)
        self.ensure_hyperparameter("std", "uniform", self.min_std, self.max_std)

        def sampler():
            mean = self.mean() if callable(self.mean) else self.mean
            std = self.std() if callable(self.std) else self.std

            def sub_sampler():
                sample = trunc_norm_sampler(mean, std)()
                return self.lower_bound + round(sample) if self.round else self.lower_bound + sample

            return sub_sampler

        return sampler

    def setup_meta_trunc_norm_log_scaled_sampler(self):
        """Sets up a log-scaled meta truncated normal distribution sampler.
        Useful for parameters that vary on logarithmic scales."""
        # Dynamically define log_mean and log_std if not explicitly provided
        self.min_std = self.min_std if hasattr(self, "min_std") else 0.01
        self.max_std = self.max_std if hasattr(self, "max_std") else 1.0
        self.ensure_hyperparameter("log_mean", "uniform", math.log(self.min_mean), math.log(self.max_mean))
        self.ensure_hyperparameter("log_std", "uniform", math.log(self.min_std), math.log(self.max_std))

        def sampler():
            log_mean = self.log_mean() if callable(self.log_mean) else self.log_mean
            log_std = self.log_std() if callable(self.log_std) else self.log_std
            mu = math.exp(log_mean)
            sigma = mu * math.exp(log_std)

            def sub_sampler():
                sample = trunc_norm_sampler(mu, sigma)()
                return self.lower_bound + round(sample) if self.round else self.lower_bound + sample

            return sub_sampler

        return sampler

    def setup_meta_choice_sampler(self):
        """Sets up a meta-categorical distribution sampler.
        Returns a closure that samples probabilities and then samples categorical values."""
        # Ensure that choice weights are defined or dynamically created
        for i in range(1, len(self.choice_values)):
            self.ensure_hyperparameter(f"choice_{i}_weight", distribution="uniform", min=-3.0, max=5.0)

        def sampler():
            weights = [1.0]
            for i in range(1, len(self.choice_values)):
                attr = getattr(self, f"choice_{i}_weight")
                weights.append(attr() if callable(attr) else attr)
            weights = torch.softmax(torch.tensor(weights, dtype=torch.float), 0)
            choice_idx = torch.multinomial(weights, 1).item()
            return self.choice_values[choice_idx]

        return sampler

    def setup_meta_choice_mixed_sampler(self):
        """Sets up a mixed meta-categorical distribution sampler.
        Similar to meta_choice but with different probability scaling."""
        # Similar to meta_choice but may include different logic for mixed scenarios
        for i in range(1, len(self.choice_values)):
            self.ensure_hyperparameter(f"choice_{i}_weight", distribution="uniform", min=-5.0, max=6.0)

        def sampler():
            weights = [1.0]
            for i in range(1, len(self.choice_values)):
                attr = getattr(self, f"choice_{i}_weight")
                weights.append(attr() if callable(attr) else attr)
            weights = torch.softmax(torch.tensor(weights, dtype=torch.float), 0)

            def sub_sampler():
                choice_idx = torch.multinomial(weights, 1).item()
                return self.choice_values[choice_idx]()

            return lambda: sub_sampler

        return sampler

    def forward(self):
        return self.sampler()


class HpSamplerList(nn.Module):
    """
    A container for multiple hyperparameter samplers that handles batch sampling.

    Parameters:
        hyperparameters (dict): Dictionary mapping parameter names to their sampling configurations
        device (str): Device to use for tensor operations

    Example:
        hp_config = {
            'learning_rate': {
                'distribution': 'meta_trunc_norm_log_scaled',
                'min_mean': 1e-4,
                'max_mean': 1e-1
            },
            'num_layers': {
                'distribution': 'uniform_int',
                'min': 2,
                'max': 10
            }
        }
        sampler = HpSamplerList(hp_config, device='cuda')
        params = sampler.sample()  # Returns dict with sampled values
    """

    def __init__(self, hyperparameters, device):
        super().__init__()
        self.device = device
        self.hyperparameters = nn.ModuleDict(
            {name: HpSampler(device=device, **params) for name, params in hyperparameters.items() if params}
        )

    def sample(self):
        return {name: hp() for name, hp in self.hyperparameters.items()}

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn
import numpy as np


class StdScaleLayer(nn.Module):
    """Standard scaling layer that normalizes input features.

    Computes mean and standard deviation on the first batch and uses these
    statistics to normalize subsequent inputs using (x - mean) / std.
    The statistics are computed along dimension 0.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fit the info on the first batch
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + 1e-6

        return (x - self.mean) / self.std


class SignActivation(nn.Module):
    """Sign function as an activation layer.

    Returns 1.0 for inputs >= 0, and -1.0 otherwise.
    Implemented as a binary step function using float values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (x >= 0.0).float() - 1.0


class Heaviside(nn.Module):
    """Heaviside function as an activation layer.

    Returns 1.0 for inputs >= 0, and 0.0 otherwise.
    Implemented as a binary step function using float values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0.0).float()


class RBFActivation(nn.Module):
    """Radial Basis Function (RBF) activation layer.

    Implements the Gaussian RBF: f(x) = exp(-x^2)
    Useful for localized feature representations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2))


class RandomFreqSineActivation(nn.Module):
    """Random frequency sine activation with fixed random scale and bias.

    Applies sine activation with randomly initialized (but fixed) frequency scaling and phase shift:
    f(x) = sin(scale * standardize(x) + bias)

    The scale and bias parameters are initialized randomly but remain constant during training
    (requires_grad=False).

    Args:
        min_scale (float): Minimum value for random frequency scaling (default: 0.1)
        max_scale (float): Maximum value for random frequency scaling (default: 100)
    """

    def __init__(self, min_scale=0.1, max_scale=100):
        super().__init__()
        log_min_scale = np.log(min_scale)
        log_max_scale = np.log(max_scale)
        self.scale = nn.Parameter(
            torch.exp(log_min_scale + (log_max_scale - log_min_scale) * torch.rand(1)), requires_grad=False
        )
        self.bias = nn.Parameter(2 * np.pi * torch.rand(1), requires_grad=False)
        self.stdscaler = StdScaleLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.scale * self.stdscaler(x) + self.bias)


class RandomFunctionActivation(nn.Module):
    """Random Fourier feature based activation function.

    Generates a random periodic function by combining multiple sine waves with
    different frequencies, phases and weights. The input is first standardized.

    Args:
        n_frequencies (int): Number of frequency components to use (default: 256)
    """

    def __init__(self, n_frequencies: int = 256):
        super().__init__()

        self.freqs = nn.Parameter(n_frequencies * torch.rand(n_frequencies), requires_grad=False)
        self.bias = nn.Parameter(2 * np.pi * torch.rand(n_frequencies), requires_grad=False)
        self.stdscaler = StdScaleLayer()

        decay_exponent = -np.exp(np.random.uniform(np.log(0.7), np.log(3.0)))
        with torch.no_grad():
            freq_factors = self.freqs**decay_exponent
            freq_factors = freq_factors / (freq_factors**2).sum().sqrt()
        self.l2_weights = nn.Parameter(freq_factors * torch.randn(n_frequencies), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stdscaler(x)
        x = torch.sin(self.freqs * x[..., None] + self.bias)
        x = (self.l2_weights * x).sum(dim=-1)
        return x


class FunctionActivation(nn.Module):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class RandomScaleLayer(nn.Module):
    """Random scaling layer with optional per-feature parameters.

    Applies random scaling and bias: f(x) = scale * (x + bias)

    Args:
        individual (bool, optional): If True, uses different parameters for each
            input feature. Defaults to False.
    """

    def __init__(self, individual: bool = False):
        super().__init__()
        self.individual = individual
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        n_out = x.shape[-1] if self.individual else 1
        self.scale = torch.exp(np.log(1.0) + 2 * torch.randn(1, n_out, device=x.device))
        # use uniform on [0, 1] since we round to integers anyway
        self.bias = torch.randn(1, n_out, device=x.device)
        self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.initialize(x)

        return self.scale * (x + self.bias)


class ExpActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class SqrtAbsActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.abs(x))


class UnitIntervalIndicator(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.abs(x) <= 1.0).float()


class SineActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


class AbsActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class StdRandomScaleFactory:
    def __init__(self, act_class, individual: bool = False):
        self.act_class = act_class
        self.individual = individual

    def __call__(self):
        return nn.Sequential(StdScaleLayer(), RandomScaleLayer(individual=self.individual), self.act_class())


class RandomChoiceActivation(nn.Module):
    """Randomly selects and instantiates one activation function from a list.

    Args:
        act_list: List of activation function constructors to choose from.

    Attributes:
        act: The randomly selected activation function instance
    """

    def __init__(self, act_list: List[nn.Module]):
        super().__init__()
        self.act = act_list[np.random.randint(len(act_list))]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class RandomChoiceFactory:
    """Factory class for creating RandomChoice activations"""

    def __init__(self, act_classes):
        self.act_classes = act_classes

    def __call__(self):
        return RandomChoiceActivation(self.act_classes)


def get_activations(random: bool = True, scale: bool = True, diverse: bool = True):
    """Generate a list of activation functions with various configurations.

    This function creates a list of activation functions by combining simple activations
    with optional random functions, scaling, and diversity options.

    Args:
        random: If True, adds RandomFunctionActivation to the list and samples it multiple
            times to increase probability of selection. Defaults to True.

        scale: If True, wraps activations with StdRandomScaleFactory to add standardization
            and random scaling. Defaults to True.

        diverse: If True, adds RandomChoiceFactory instances to allow different activation
            functions in each layer. Defaults to True.
    """
    # Start with a set of simple activations
    simple_activations = [
        nn.Tanh,
        nn.LeakyReLU,
        nn.ELU,
        nn.Identity,
        nn.SELU,
        nn.SiLU,
        nn.ReLU,
        nn.Softplus,
        nn.ReLU6,
        nn.Hardtanh,
        SignActivation,
        RBFActivation,
        ExpActivation,
        SqrtAbsActivation,
        UnitIntervalIndicator,
        SineActivation,
        SquareActivation,
        AbsActivation,
    ]
    activations = simple_activations
    if random:
        # Add random activation and sample it more often
        activations += [RandomFunctionActivation] * 10

    if scale:
        # Create scaled versions using StdRandomScaleFactory
        activations = [StdRandomScaleFactory(act) for act in activations]

    if diverse:
        # Add possibility to have different activation functions in each layer
        activations += [RandomChoiceFactory(activations)] * len(activations)

    return activations

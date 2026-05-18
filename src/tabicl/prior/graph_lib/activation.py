import functools

import torch
import torch.nn.functional as F

from tabiclv2.prior.graph_lib.base import RandomTensorTransformer, RandomTensorSequential


class RandomActivation(RandomTensorTransformer):
    """
    Base class for random activation functions.
    By activation function, we mean any funtion that preserves the dimension (number of features),
    not just elementwise functions.
    """
    def _fit(self, x: torch.Tensor):
        # don't use RandomGPActivation because it's slow
        # todo: could use other activations (neg power and int power) here, like in the other activation types
        act_types = [RandomTorchActivation] * 4 + [RandomPowerReluActivation, RandomPowerActivation]
        base_act = self.sampler.choice("act_type", act_types)(self.context)
        self.act_ = RandomTensorSequential(
            self.context,
            [
                Standardize(self.context),
                RandomRescale(self.context),
                base_act,
                Standardize(self.context),
            ],
        )

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_(x)


# ----- some specific activations -----


def sgn(x):
    return 2 * (x >= 0.0).float() - 1.0


def heaviside(x):
    return (x >= 0.0).float()


def unit_interval_indicator(x):
    return (torch.abs(x) <= 1.0).float()


def rbf(x):
    return torch.exp(-(x**2))


def identity(x):
    return x


def log_clip_abs(x):
    return torch.log(torch.clamp(torch.abs(x), min=1e-6))


def is_max(x):
    return (x == torch.max(x, dim=-1, keepdim=True).values).float()


def rank(x):
    perm = torch.argsort(x, dim=-1)
    # from https://discuss.pytorch.org/t/how-to-quickly-inverse-a-permutation-by-using-pytorch/116205/7
    src = torch.arange(x.shape[-1], device=x.device)[None].expand(x.shape[0], -1)
    perms_inv = torch.empty_like(perm, device=x.device)
    perms_inv = torch.scatter(perms_inv, dim=1, index=perm, src=src)
    return perms_inv.float()


def argsort_float(x):
    return torch.argsort(x, dim=-1).float()


def modulo_act(x):
    return x - torch.floor(x)


acts = [
    # TabICLv1
    torch.tanh,
    F.leaky_relu,
    F.elu,
    identity,
    F.selu,
    F.silu,
    F.relu,
    F.softplus,
    F.relu6,
    F.hardtanh,
    sgn,
    heaviside,
    rbf,
    torch.exp,
    unit_interval_indicator,
    torch.sin,
    torch.square,
    torch.abs,
    # TabPFNv2
    # unclear: power?, smooth relu is unclear but it might be softplus,
    log_clip_abs,
    rank,
    F.sigmoid,
    torch.round,
    modulo_act,
    # extra
    functools.partial(F.softmax, dim=-1),
    is_max,
    argsort_float,
    F.logsigmoid,
]


class RandomActivationPlain(RandomActivation):
    """
    Random activation, without the standardization / rescaling
    """
    def _fit(self, x: torch.Tensor):
        act_types = [RandomTorchActivation] * 8 + [
            RandomPowerReluActivation,
            RandomPowerActivation,
            RandomNegPowerActivation,
            RandomIntPowerActivation,
        ]
        # don't use RandomGPActivation because it's slow
        base_act = self.sampler.choice("act_type_plain", act_types)(self.context)
        self.act_ = base_act

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_(x)


class RandomTorchActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.act_ = self.sampler.choice("acts_torch", acts)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_(x)


class RandomPowerReluActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = self.sampler.numerical("relu_random_power_exponent", 0.1, 10.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** self.exponent_


class RandomPowerActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = self.sampler.numerical("random_power_exponent", 0.1, 10.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return sgn(x) * (x.abs() ** self.exponent_)


class RandomNegPowerActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = -self.sampler.numerical("neg_random_power_exponent", 0.1, 10.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs() + 1e-3) ** self.exponent_


class RandomIntPowerActivation(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.exponent_ = self.sampler.randint("rand_int_power_exponent", 2, 6)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return x**self.exponent_


class Standardize(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.inv_std_ = 1.0 / (x.std(dim=0, keepdim=True, correction=0) + 1e-4)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) * self.inv_std_


class L2Normalize(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.factor_ = 1.0 / (x.square().sum(dim=-1).mean().sqrt() + 1e-8)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.factor_ * x


class RandomRescale(RandomActivation):
    def _fit(self, x: torch.Tensor):
        self.scale_ = self.sampler.numerical("rr_scale", 1e-0, 1e1, use_log=True)
        # take random datapoints for shifts, so that activations like ReLU are not always zero
        self.bias_ = -x[
            torch.randint(x.shape[0], size=(x.shape[1],), device=x.device), torch.arange(x.shape[1], device=x.device)
        ][None, :]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale_ * (x + self.bias_)


class KumaraswamyWarping(RandomActivation):
    """
    Performs min-max scaling and then the Kumaraswamy transform.
    """
    def _fit(self, x: torch.Tensor):
        self.min_ = x.min(dim=0).values
        self.max_ = x.max(dim=0).values
        self.a_ = self.sampler.numerical("kumaraswamy_a", 0.2, 5, use_log=True)
        self.b_ = self.sampler.numerical("kumaraswamy_b", 0.2, 5, use_log=True)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((x - self.min_) / (self.max_ - self.min_ + 1e-30), 0.0, 1.0)
        return 1.0 - (1.0 - x ** self.a_) ** self.b_
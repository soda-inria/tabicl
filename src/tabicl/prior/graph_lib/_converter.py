from typing import Tuple, Any

import torch
import numpy as np

from tabiclv2.prior.graph_lib.base import PriorComponent, Context, RandomTensorTransformer, RandomTransformer
from tabiclv2.prior.graph_lib.activation import Standardize, RandomActivation, KumaraswamyWarping
from tabiclv2.prior.graph_lib.function import CheapRandomFunction
from tabiclv2.prior.graph_lib.points import RandomPoints
from tabiclv2.prior.graph_lib.weights import RandomWeights


class Converter(RandomTransformer):
    """
    A converter extracts a column for the dataset from a part of a node matrix,
    and can also modify this part of the node matrix.
    """
    def __init__(self, context: Context):
        super().__init__(context, allow_override_by_class=Converter)

    def get_n_features(self) -> int:
        # Return number of features that the converter needs to extract the data.
        raise NotImplementedError()

    def _fit(self, x: torch.Tensor):
        pass

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the converter.

        Parameters
        ----------
        x : torch.Tensor
            Values to convert of shape (n_samples, n_features). (Part of the node matrix.)

        Returns
        -------
            Tuple of (values to be further used in the graph, values to be used in the dataset).
        """
        return super().__call__(x)  # this function is only redefined here to redefine the type signatures


class NumericalConverter(Converter):
    def __init__(self, context: Context, disallow_warping: bool = False):
        super().__init__(context)
        self.disallow_warping = disallow_warping

    def get_n_features(self):
        return 1

    def _fit(self, x: torch.Tensor):
        modes = ["id"]
        if self.config.allow_kumaraswamy_warping and not self.disallow_warping:
            modes.append("kumar")
        if self.config.allow_act_warping and not self.disallow_warping:
            modes.append("act")
        self.mode_ = self.sampler.choice("num_conv_mode", modes)
        self.tfm_ = RandomActivation(self.context) if self.mode_ == 'act' else KumaraswamyWarping(self.context)

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode_ == "id":
            return x, x
        elif self.mode_ in ["kumar", "act"]:
            if self.config.use_corrected_num_converters:
                return x, self.tfm_(x)
            else:
                # this is the wrong version, but it was used in the experiments
                return self.tfm_(x), x
        else:
            raise NotImplementedError()


class CategoricalNeighborDiscretizer(Converter):
    def __init__(self, context: Context, n_values: int):
        super().__init__(context)
        self.n_values = n_values

    def _fit(self, x: torch.Tensor):
        # Create permutation on same device as x
        perm = torch.randperm(len(x), device=x.device)
        # Store centers using correct device
        self.centers_ = x[perm[: self.n_values]]
        self.p_ = self.sampler.numerical("cat_disc_p", 0.5, 4.0, use_log=True)

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # All operations happen on the same device as x and centers
        dists = torch.cdist(x, self.centers_, p=self.p_)  # this is faster than manually calculating the distance matrix
        closest_idx = dists.argmin(dim=-1)
        return self.centers_[closest_idx], closest_idx[:, None]


class CategoricalSoftmaxDiscretizer(Converter):
    def __init__(self, context: Context, n_values: int):
        super().__init__(context)
        self.n_values = n_values

    def _fit(self, x: torch.Tensor):
        self.embeddings_ = RandomPoints(self.context).sample(self.n_values, self.n_values)
        self.standardize_ = Standardize(self.context)
        self.bias_ = torch.log(RandomWeights(self.context).sample(1, x.shape[1]) + 1e-4)
        self.factor_ = self.sampler.numerical("softmax_disc_factor", 0.1, 10, use_log=True)

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.factor_ * self.standardize_(x) + self.bias_
        x[~torch.isfinite(x)] = 0.0  # this is only a quick fix, but whatever
        idxs = torch.multinomial(torch.softmax(x, dim=-1), num_samples=1, replacement=True).squeeze(-1)
        return self.embeddings_[idxs], idxs[:, None]


class CategoricalConverter(Converter):
    def __init__(self, context: Context, n_values: int):
        super().__init__(context)
        self.n_values = n_values

        all_cat_modes = ["neighbor_id", "neighbor_disc", "neighbor_func", "neighbor_int", "softmax_id", "softmax_disc",
                         "softmax_int"]

        if self.config.cat_modes == 'default':
            cat_modes = all_cat_modes
        elif self.config.cat_modes in all_cat_modes:
            cat_modes = [self.config.cat_modes]
        elif self.config.cat_modes == 'neighbor':
            cat_modes = [cm for cm in all_cat_modes if cm.startswith('neighbor')]
        elif self.config.cat_modes == 'softmax':
            cat_modes = [cm for cm in all_cat_modes if cm.startswith('softmax')]
        else:
            raise ValueError(f'Unknown value {self.config.cat_modes=}')

        self.mode = self.sampler.choice(
            "cat_mode", cat_modes
        )
        # self.mode = "softmax_id"
        self.disc = (
            CategoricalSoftmaxDiscretizer(self.context, n_values=self.n_values)
            if self.mode.startswith("softmax")
            else CategoricalNeighborDiscretizer(self.context, n_values=self.n_values)
        )
        if self.mode.startswith("softmax"):
            self.n_features = n_values
        else:
            self.n_features = (
                n_values
                if self.sampler.boolean("cat_use_all_features")
                else self.sampler.randint("cat_n_features", 1, n_values, use_log=True, mode="local")
            )
        self.post_func = (
            CheapRandomFunction(self.context, self.n_features, self.n_features)
            if self.mode == "neighbor_func"
            else None
        )

    def get_n_features(self) -> int:
        return self.n_features

    def _fit(self, x: torch.Tensor):
        pass  # already fit in the constructor since we need it for n_features

    def _transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_disc, x_idxs = self.disc(x)

        if self.mode in ["neighbor_id", "softmax_id"]:
            return x, x_idxs
        elif self.mode in ["neighbor_disc", "softmax_disc"]:
            return x_disc, x_idxs
        elif self.mode == "neighbor_func":
            return self.post_func(x_disc), x_idxs
        elif self.mode in ["neighbor_int", "softmax_int"]:
            return x_idxs.float(), x_idxs  # x_idxs.float() is 1 column, it's going to be broadcasted
        else:
            raise ValueError(f'Unknown categorical mode "{self.mode}"')
from typing import List, Dict, Tuple
import torch

from tabicl.prior.graph_lib._base import RandomTransformer, Context, FeatureSpec
from tabicl.prior.graph_lib._activation import Standardize, L2Normalize
from tabicl.prior.graph_lib._converter import CategoricalConverter, NumericalConverter
from tabicl.prior.graph_lib._multi_function import RandomMultiFunction
from tabicl.prior.graph_lib._points import RandomPoints
from tabicl.prior.graph_lib._weights import RandomWeights


class RandomNodeFunction(RandomTransformer):
    """
    Computes the node matrix from the parents' matrices, and optionally extracts columns for the dataset.
    """
    def __init__(self, context: Context, feature_specs: Dict[str, FeatureSpec]):
        super().__init__(context)
        self.feature_specs = feature_specs

    def _fit(self, x: List[torch.Tensor], n_samples: int):
        self.converters_ = dict()
        for feature_name, feature_spec in self.feature_specs.items():
            cat_size = feature_spec.cat_size
            converter = (
                CategoricalConverter(self.context, n_values=cat_size)
                if cat_size > 0
                else NumericalConverter(self.context,
                                        disallow_warping=self.config.disallow_y_warping and feature_spec.group == 'y')
            )
            self.converters_[feature_name] = converter

        self.n_features_ = self.sampler.randint("node_n_latent_features", 1, 32, use_log=True)
        if len(self.converters_) > 0:
            self.n_features_ += sum(converter.get_n_features() for converter in self.converters_.values())
        self.random_points_ = RandomPoints(self.context)
        self.random_func_ = RandomMultiFunction(self.context, out_features=self.n_features_)

        self.std_ = Standardize(self.context)
        self.l2_norm_ = L2Normalize(self.context)
        self.random_weights_ = RandomWeights(self.context)

    def _transform(self, x: List[torch.Tensor], n_samples: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(x) == 0:
            x = self.random_points_.sample(n_samples, self.n_features_)
        else:
            x = self.random_func_(x)

        if self.config.use_node_feature_importance:
            x = self.std_(x)
            weights = self.random_weights_.sample(1, x.shape[1]).squeeze(0)
            weights = weights / weights.square().mean().sqrt()
            x = x * weights[None, :]
        if self.config.use_node_l2_norm:
            x = self.l2_norm_(x)

        out_features = dict()
        start_idx = 0
        for name, converter in self.converters_.items():
            end_idx = start_idx + converter.get_n_features()
            assert end_idx <= self.n_features_
            tens, feat = converter(x[:, start_idx:end_idx])
            x[:, start_idx:end_idx] = tens
            out_features[name] = feat
            start_idx = end_idx

        if self.config.add_gaussian_noise:
            noise_level = self.sampler.numerical("gaussian_noise_sigma", low=1e-5, high=1e0, use_log=True)
            x = x + noise_level * torch.randn_like(x)

        if self.config.use_node_importance:
            x = x * self.sampler.numerical("node_importance", low=0.1, high=10.0, use_log=True)

        return x, out_features

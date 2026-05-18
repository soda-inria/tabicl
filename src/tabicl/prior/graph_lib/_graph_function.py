from typing import List, Dict

import torch

from tabiclv2.prior.graph_lib.base import RandomTransformer, Context, FeatureSpec
from tabiclv2.prior.graph_lib.node_function import RandomNodeFunction


class RandomGraphFunction(RandomTransformer):
    """
    Samples a dataset by propagating data through a graph.
    """
    def __init__(self, context: Context, dag: List[List[int]], node_feature_specs: List[Dict[str, FeatureSpec]]):
        super().__init__(context=context)
        self.dag = dag
        self.node_feature_specs = node_feature_specs

    def _fit(self, n_samples: int):
        self.nodes_ = [
            RandomNodeFunction(self.context, feature_specs=feature_specs)
            for feature_specs in self.node_feature_specs
        ]
        # for efficiency, prune nodes whose values don't need to be computed
        self.should_compute_ = [False for _ in range(len(self.nodes_))]
        for node_idx in reversed(range(len(self.nodes_))):
            if len(self.node_feature_specs[node_idx]) >= 1:
                self.should_compute_[node_idx] = True
            if self.should_compute_[node_idx]:  # could have been set by successors or by itself
                for parent in self.dag[node_idx]:
                    self.should_compute_[parent] = True

    def _transform(self, n_samples: int) -> Dict[str, torch.Tensor]:
        n_nodes = len(self.node_feature_specs)
        node_values = [None for _ in range(n_nodes)]
        features = dict()
        for node_idx in range(len(self.node_feature_specs)):
            if self.should_compute_[node_idx]:
                node_values[node_idx], out_features = self.nodes_[node_idx](
                    [node_values[parent] for parent in self.dag[node_idx]], n_samples
                )
                features = features | out_features
        return features

from typing import List

import torch
import numpy as np

from tabiclv2.prior.graph_lib.base import PriorComponent, Context


class RandomDAG(PriorComponent):
    """
    Random Directed Acyclic Graph (DAG) generator.
    """

    def __init__(self, context: Context):
        super().__init__(context)

    def sample(self, n_nodes: int) -> List[List[int]]:
        # Generate a random DAG with the specified number of nodes
        # return List[List[int]], where parents[i] contains the list of parent node indices for node i
        return RandomCauchyDAG(self.context).sample(n_nodes)


class RandomCauchyDAG(RandomDAG):
    """
    Samples a random DAG based on Cauchy random variables modeling connectivities.
    """
    def sample(self, n_nodes: int) -> List[List[int]]:
        offset = np.random.standard_cauchy() + self.config.cauchy_dag_offset
        output_importances = torch.tensor(np.random.standard_cauchy(n_nodes))
        input_importances = torch.tensor(np.random.standard_cauchy(n_nodes))
        logits = offset + output_importances[None, :] + input_importances[:, None]
        adjacency_matrix = torch.rand_like(logits) <= torch.sigmoid(logits)
        return [[i_in for i_in in range(i_out) if adjacency_matrix[i_in, i_out]] for i_out in range(n_nodes)]


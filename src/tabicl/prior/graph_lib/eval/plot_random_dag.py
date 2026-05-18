from typing import List

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

from tabiclv2.prior.graph_lib.base import Context
from tabiclv2.prior.graph_lib.graph import RandomCauchyDAG


def plot_graph_ax(ax: plt.Axes, graph: List[List[int]]):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges based on predecessors list
    for node, preds in enumerate(graph):
        for pred in preds:
            G.add_edge(pred, node)

    # Draw the graph
    pos = nx.spring_layout(G, iterations=1000)  # Layout for better visualization
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue", edge_color="gray", arrows=True, node_size=150)


def plot_random_graphs(cls, n_nodes: int = 15, n_rows: int = 4, n_cols: int = 5):
    np.random.seed(1)
    torch.manual_seed(1)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            plot_graph_ax(axs[row_idx, col_idx], cls(Context()).sample(n_nodes))

    plt.suptitle(f"{cls.__name__}")
    plt.tight_layout()
    plt.savefig(f"plots/{cls.__name__}.pdf", dpi=300)
    # plt.show()


if __name__ == "__main__":
    plot_random_graphs(RandomCauchyDAG)

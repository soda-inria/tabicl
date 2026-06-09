import matplotlib.pyplot as plt
import torch
import numpy as np

from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._points import (
    RandomPoints,
    RandomGaussianPoints,
    RandomGaussianMixturePoints, RandomCovariancePoints,
)


def plot_random_points_ax(ax: plt.Axes, points: torch.Tensor):
    ax.set_xticks([])
    ax.set_yticks([])
    if points.shape[1] >= 3:
        ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=3, cmap="viridis")
    else:
        ax.plot(points[:, 0], points[:, 1], ".")


def plot_random_points(cls):
    np.random.seed(0)
    torch.manual_seed(0)
    nrows, ncols = 9, 12
    n_points = 300
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 12))
    for row in range(nrows):
        for col in range(ncols):
            plot_random_points_ax(axs[row, col], cls(Context()).sample(n_points, 32))

    plt.suptitle(cls.__name__)
    plt.tight_layout()
    plt.savefig(f"plots/{cls.__name__}.pdf", dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_random_points(RandomCovariancePoints)
    plot_random_points(RandomPoints)
    plot_random_points(RandomGaussianMixturePoints)
    plot_random_points(RandomGaussianPoints)

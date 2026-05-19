import matplotlib.pyplot as plt
import torch

from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._weights import SimpleRandomWeights


def plot_random_weights_ax(ax: plt.Axes, weights_class, n_weights: int, n_repeats: int = 20):
    ctx = Context()
    for _ in range(n_repeats):
        weights = weights_class(ctx).sample(1, n_weights).squeeze(0).sort(descending=True)[0]
        weights /= weights.max()
        ax.plot(torch.arange(len(weights)), weights, ".-")


def plot_random_weights(weights_class):
    nrows = 3
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 12))
    for row in range(nrows):
        for col in range(ncols):
            plot_random_weights_ax(axs[row, col], weights_class, n_weights=2 + 2 * col + 10 * row)

    plt.suptitle(weights_class.__name__)
    plt.tight_layout()
    plt.savefig(f"plots/{weights_class.__name__}.pdf", dpi=300)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    for weights_class in [SimpleRandomWeights]:
        plot_random_weights(weights_class)

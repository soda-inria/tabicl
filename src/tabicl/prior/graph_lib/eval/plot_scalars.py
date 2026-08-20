import numpy as np
import matplotlib.pyplot as plt

from tabicl.prior.graph_lib._base import Context


def plot_histograms(data):
    """
    Plots a 6x4 grid of histograms for each row of the input array.

    Parameters:
    data (numpy.ndarray): A (24, n) numpy array.
    """
    if data.shape[0] != 24:
        raise ValueError("Input array must have 24 rows")

    fig, axes = plt.subplots(6, 4, figsize=(12, 18))
    axes = axes.ravel()  # Flatten the 2D array of axes

    for i in range(24):
        axes[i].hist(data[i, :], bins=20, color="blue", edgecolor="black", alpha=0.7)
        # axes[i].set_title(f'Row {i}')
        # axes[i].set_xlabel('Value')
        # axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_scalars():
    ctx = Context()
    values = np.array(
        [[ctx.sampler.numerical("a" * i, 0.0, 1.0, boundary_mass=False) for _ in range(1000)] for i in range(24)]
    )
    plot_histograms(values)


if __name__ == "__main__":
    plot_scalars()
    pass

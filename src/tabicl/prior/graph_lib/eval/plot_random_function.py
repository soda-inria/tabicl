import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._function import (
    RandomMLPFunction,
    RandomFunction,
    RandomTreeFunction,
    RandomProductFunction,
    RandomDiscretizationFunction,
    RandomGPFunction,
)


def plot_function_surface_ax(ax, f, xlim, ylim, resolution=50, dim=2):
    """
    Plots the surface of a function f over a square domain.

    Parameters:
    - f: function that maps a (n_samples, 2) torch tensor to a (n_samples, 1) torch tensor
    - xlim: tuple (xmin, xmax) defining the x-axis limits
    - ylim: tuple (ymin, ymax) defining the y-axis limits
    - resolution: number of points per axis for the grid (default: 50)
    """
    x = torch.linspace(xlim[0], xlim[1], resolution)
    y = torch.linspace(ylim[0], ylim[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    if dim > 2:
        new_points = torch.zeros(points.shape[0], dim)
        new_points[:, :2] = points
        points = new_points

    Z = f(points).reshape(resolution, resolution).detach().numpy()
    Z /= 1e-6 + np.sqrt(np.mean(np.square(Z.flatten())))

    X, Y = X.numpy(), Y.numpy()

    ax.plot_surface(X, Y, Z, cmap="viridis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(X, Y)")


def plot_random_function_surface(fct_class):
    print(f"Plotting {fct_class.__name__}")
    nrows, ncols = 4, 5
    fig, axs = plt.subplots(nrows, ncols, subplot_kw={"projection": "3d"}, figsize=(12, 10))
    dim = 2

    for row in range(nrows):
        for col in range(ncols):
            plot_function_surface_ax(
                axs[row, col], fct_class(Context(), dim, 1).__call__, (-3, 3), (-3, 3),
                resolution=200, dim=dim
            )

    plt.suptitle(f"{fct_class.__name__}")
    plt.savefig(f"plots/{fct_class.__name__}.pdf", dpi=300)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)
    plot_random_function_surface(RandomGPFunction)
    plot_random_function_surface(RandomDiscretizationFunction)
    plot_random_function_surface(RandomFunction)
    plot_random_function_surface(RandomMLPFunction)
    plot_random_function_surface(RandomProductFunction)
    plot_random_function_surface(RandomTreeFunction)

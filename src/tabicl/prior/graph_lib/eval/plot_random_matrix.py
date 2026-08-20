import matplotlib.pyplot as plt
import torch
import numpy as np

from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._matrix import (
    RandomMatrix,
    RandomGaussianMatrix,
    RandomWeightsMatrix,
    RandomKernelMatrix,
    RandomActivationMatrix,
    RandomSingularValuesMatrix,
)


def plot_matrix_ax(ax: plt.Axes, mat: torch.Tensor):
    # print(mat.shape)
    # print(mat)
    ax.set_aspect("equal")
    # print(mat)
    # ax.imshow(mat.abs().cpu().numpy(), interpolation='None', aspect='equal', origin='upper', vmin=0, vmax=1,
    #           extent=[0, mat.shape[1], 0, mat.shape[0]])
    U, S, Vh = torch.linalg.svd(mat.abs())
    u = U[:, 0]
    v = Vh[0, :]
    # u, v can be interpreted as the cos(angle of coordinates wrt first singular vector)
    input_perm = torch.argsort(v.abs(), descending=True)
    output_perm = torch.argsort(u.abs(), descending=True)
    # this somehow fails sometimes for matplotlib==3.7.1 for the kernel matrix.
    ax.imshow(mat[output_perm][:, input_perm].abs(), interpolation="None", vmin=0)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_random_matrices(cls, n: int, m: int, n_rows: int = 6, n_cols: int = 8):
    np.random.seed(1)
    torch.manual_seed(1)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            print(f"##### sampling new matrix #####")
            plot_matrix_ax(axs[row_idx, col_idx], cls(Context()).sample(1, n, m).squeeze(0))

    plt.suptitle(f"{cls.__name__}")
    plt.tight_layout()
    plt.savefig(f"plots/{cls.__name__}.pdf", dpi=300)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    n, m = 30, 30
    plot_random_matrices(RandomMatrix, n, m)
    plot_random_matrices(RandomGaussianMatrix, n, m)
    plot_random_matrices(RandomWeightsMatrix, n, m)
    plot_random_matrices(RandomKernelMatrix, n, m)
    plot_random_matrices(RandomActivationMatrix, n, m)
    plot_random_matrices(RandomSingularValuesMatrix, n, m)

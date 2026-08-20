from typing import Dict

import matplotlib.pyplot as plt
import torch
import numpy as np

from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._dataset import RandomDataset


def plot_random_dataset_ax(ax: plt.Axes, ds_tensors: Dict[str, torch.Tensor]):
    ax.set_xticks([])
    ax.set_yticks([])

    x_num = ds_tensors["x_num"]
    y_cat = ds_tensors["y_cat"].squeeze(-1)
    classes = torch.unique(y_cat)
    # print(f"{classes=}")

    for c in classes:
        # print(f"class size: {(y_cat==c).float().sum().item()}")
        ax.plot(x_num[y_cat == c, 0], x_num[y_cat == c, 1], ".")
        # ax.scatter(x_num[:, 0], x_num[:, 1], c=y_cat, s=3)


def plot_random_dataset(cls):
    np.random.seed(1)
    torch.manual_seed(1)
    # nrows, ncols = 6, 8
    nrows, ncols = 9, 12
    n_points = 300
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 12))
    for row in range(nrows):
        for col in range(ncols):
            print(f"Generating random dataset for {row=}, {col=}")
            n_classes = 2 if np.random.rand() < 0.5 else np.random.randint(3, 11)
            ds_prop = DatasetProperties(n_train=n_points, n_test=0, cat_sizes={"x": [0]*20, "y": [n_classes]})
            while True:
                ds_tensors = cls(Context()).sample(ds_prop).get_concat_tensors()
                # print(list(ds_tensors.keys()))
                if len(torch.unique(ds_tensors["y_cat"])) == n_classes:
                    break
            plot_random_dataset_ax(axs[row, col], ds_tensors)

    plt.suptitle(cls.__name__)
    plt.tight_layout()
    plt.savefig(f"plots/{cls.__name__}.pdf", dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_random_dataset(RandomDataset)

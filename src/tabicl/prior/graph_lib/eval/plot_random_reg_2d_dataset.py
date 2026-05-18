import matplotlib.pyplot as plt
import torch
import numpy as np

from tabiclv2.prior.graph_lib.base import Context
from tabiclv2.prior.graph_lib.dataset import RandomDataset, Dataset, DatasetProperties


def plot_random_dataset_ax(ax: plt.Axes, dataset: Dataset):
    ax.set_xticks([])
    ax.set_yticks([])

    x_num = dataset.tensors["x_num"]
    y_num = dataset.tensors["y_num"].squeeze(-1)
    y_num = y_num - y_num.min() / (y_num.max() - y_num.min() + 1e-8)

    print(y_num[:5])

    ax.scatter(x_num[:, 0], x_num[:, 1], c=y_num, s=3)


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
            ds_prop = DatasetProperties(n_train=n_points, n_test=0, cat_sizes={"x": [0]*2, "y": [0]})
            while True:
                ds = cls(Context()).sample(ds_prop)
                if len(torch.unique(ds.tensors["y_num"])) > 1:
                    break
            plot_random_dataset_ax(axs[row, col], ds)

    plt.suptitle(cls.__name__)
    plt.tight_layout()
    plt.savefig(f"plots/{cls.__name__}_reg_2d.pdf", dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_random_dataset(RandomDataset)

import time

import numpy as np
import threadpoolctl
import torch

from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._properties import sample_categorical_sizes
from tabicl.prior.graph_lib._dataset import RandomDataset
from tabicl.prior.graph_lib._function import (
    RandomMLPFunction,
    RandomTreeFunction,
    RandomDiscretizationFunction,
    RandomFunction,
    RandomLinearFunction,
    RandomGPFunction, RandomQuadraticFunction, RandomProductFunction, RandomEMAssignmentFunction,
)
from tabicl.prior.graph_lib._matrix import (
    RandomGaussianMatrix,
    RandomWeightsMatrix,
    RandomKernelMatrix,
    RandomActivationMatrix,
    RandomMatrix,
    RandomSingularValuesMatrix,
)
from tabicl.prior.graph_lib._points import RandomGaussianPoints
from tabicl.prior.graph_lib._weights import SimpleRandomWeights


# could have multiple settings: RandomFunction / full dataset generation
# then sequential generation or multiprocessing
# call __call__ multiple times to see the difference between fit+transform and transform


def measure_function_runtime(fct_class, n_reps: int = 32, input_dim: int = 32, output_dim: int = 8):
    print(f"Measuring runtime for {fct_class.__name__}:")
    np.random.seed(0)
    torch.manual_seed(0)

    for n_samples in [1000, 10_000]:
        print(f"\tResults for {n_samples} samples:")

        full_times = []
        transform_times = []

        points = RandomGaussianPoints(Context()).sample(n_samples, input_dim)

        # warmup
        fct_class(Context(), input_dim, output_dim)(points)

        for _ in range(n_reps):
            start_time = time.time()
            fun = fct_class(Context(), input_dim, output_dim)
            fun(points)
            end_time = time.time()
            full_times.append(end_time - start_time)

            start_time = time.time()
            fun(points)
            end_time = time.time()
            transform_times.append(end_time - start_time)

        print(f"\t\tFull: {np.mean(full_times):g}s +- {np.std(full_times) / np.sqrt(n_reps):g}s")
        print(f"\t\tTransform: {np.mean(transform_times):g}s +- {np.std(transform_times) / np.sqrt(n_reps):g}s")


def measure_matrix_runtime(mat_class, n_reps: int = 16, n_batch: int = 1, n: int = 32, m: int = 8):
    print(f"Measuring runtime for {mat_class.__name__}:")
    np.random.seed(0)
    torch.manual_seed(0)

    full_times = []

    # warmup
    mat_class(Context()).sample(n_batch, n, m)

    for _ in range(n_reps):
        start_time = time.time()
        mat_class(Context()).sample(n_batch, n, m)
        end_time = time.time()
        full_times.append(end_time - start_time)

    print(f"\tTime: {np.mean(full_times):g}s +- {np.std(full_times) / np.sqrt(n_reps):g}s")


def measure_weights_runtime(weight_class, n_reps: int = 16, n_batch: int = 32, n: int = 32):
    print(f"Measuring runtime for {weight_class.__name__}:")
    np.random.seed(0)
    torch.manual_seed(0)

    ctx = Context()

    full_times = []

    # warmup
    weight_class(ctx).sample(n_batch, n)

    for _ in range(n_reps):
        start_time = time.time()
        weight_class(ctx).sample(n_batch, n)
        end_time = time.time()
        full_times.append(end_time - start_time)

    print(f"\tTime: {np.mean(full_times):g}s +- {np.std(full_times) / np.sqrt(n_reps):g}s")


def measure_prior_runtime(n_reps: int = 128):
    print(f"Measuring runtime for the prior:")

    np.random.seed(0)
    torch.manual_seed(0)

    full_times = []

    # data_prop = DatasetProperties(n_train=1024, n_test=0, cat_sizes={'x': [0]*10 + [2, 2, 2, 3, 4, 7, 10, 20], 'y': [3]})
    # data_prop = DatasetProperties(
    #     n_train=1024, n_test=0, cat_sizes={"x": [0] * 50 + [2, 2, 2, 3, 4, 7, 10, 20], "y": [3]}
    # )
    data_prop = DatasetProperties(
        n_train=1024, n_test=0, cat_sizes={"x": sample_categorical_sizes(context=Context(), n_features=50, max_cat_size=200), "y": [3]}
    )

    # warmup
    RandomDataset(Context()).sample(data_prop)

    for _ in range(n_reps):
        start_time = time.time()
        RandomDataset(Context()).sample(data_prop)
        end_time = time.time()
        full_times.append(end_time - start_time)

    print(f"\tTime: {np.mean(full_times):g}s +- {np.std(full_times) / np.sqrt(n_reps):g}s")


if __name__ == "__main__":
    with threadpoolctl.threadpool_limits(limits=1):
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)

        measure_prior_runtime()

        for weights_class in [SimpleRandomWeights]:
            measure_weights_runtime(weights_class)

        for fct_class in [
            RandomMLPFunction,
            RandomTreeFunction,
            RandomDiscretizationFunction,
            RandomGPFunction,
            RandomLinearFunction,
            RandomQuadraticFunction,
            RandomEMAssignmentFunction,
            RandomProductFunction,
            RandomFunction,
        ]:
            measure_function_runtime(fct_class)

        for mat_class in [
            RandomWeightsMatrix,
            RandomGaussianMatrix,
            RandomSingularValuesMatrix,
            RandomKernelMatrix,
            RandomActivationMatrix,
            RandomMatrix,
        ]:
            measure_matrix_runtime(mat_class)

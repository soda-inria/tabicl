"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import functools
import os

import sys
import math
import warnings
from typing import Dict, Tuple, Union, Optional, Any, List, Callable

import numpy as np
import psutil
import threadpoolctl
from scipy.stats import loguniform
import multiprocessing as mp

import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset

from .graph_lib._config import PriorConfig
from ._mlp_scm import MLPSCM
from ._tree_scm import TreeSCM
from ._graph_scm import GraphSCM

from ._hp_sampling import HpSamplerList
from ._reg2cls import Reg2Cls
from ._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


def run_parallel(func: Callable, args: List[Any], n_jobs: int = -1) -> List[Any]:
    """
    Uses a multiprocessing.Pool to evaluate func on all values in args, with n_jobs processes running in parallel.

    :param func: Function to be called.
    :param args: List of arguments to call the function with.
    :param n_jobs: Number of jobs (if -1, set to the number of physical cores).
    :return: Returns the list of values returned by the function evaluated on different values from args.
    """
    ctx = mp.get_context(method='fork')  # not sure if this is necessary
    with ctx.Pool(processes=n_jobs if n_jobs >= 1 else psutil.cpu_count(logical=False),
                 initializer=functools.partial(torch.set_num_threads, 1)) as pool:
        return pool.map(func, args)


class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    regression : bool, default=False
        If True, generates regression datasets. Otherwise, generates classification datasets.

    batch_size : int, default=256
        Total number of datasets to generate per batch

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
    """

    def __init__(
            self,
            regression: bool = False,
            batch_size: int = 256,
            min_features: int = 2,
            max_features: int = 100,
            max_classes: int = 10,
            min_seq_len: Optional[int] = None,
            max_seq_len: int = 1024,
            log_seq_len: bool = False,
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            replay_small: bool = False,
    ):
        self.regression = regression
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = 0 if regression else max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
        """
        Checks if the training size range is valid.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size (position or ratio)

        max_train_size : int|float
            Maximum training size (position or ratio)

        Raises
        ------
        AssertionError
            If training size range is invalid
        ValueError
            If training size types are mismatched or invalid
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert 0 < min_train_size < max_train_size < 1, "0 < min_train_size < max_train_size < 1"
        else:
            raise ValueError("Both training sizes must be of the same type (int or float)")

    @staticmethod
    def sample_seq_len(
            min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(min_train_size: Union[int, float], max_train_size: Union[int, float], seq_len: int) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return max_features
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        return X_new, d_new

    @staticmethod
    def cls_sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2) -> bool:
        """
        Verifies that both train and test sets contain all classes for classification datasets.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target Targets tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """

        def is_valid_split(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True


class SCMPrior(Prior):
    """
    Generates synthetic datasets using Structural Causal Models (SCM).

    The data generation process follows a hierarchical structure:
    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary transformations and checks.

    Parameters
    ----------
    regression : bool, default=False
        If True, generates regression datasets. Otherwise, generates classification datasets.

    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', or 'mix_scm'
        'mix_scm' randomly selects between 'mlp_scm' and 'tree_scm' based on probabilities.

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
            self,
            regression: bool = False,
            batch_size: int = 256,
            batch_size_per_gp: int = 4,
            batch_size_per_subgp: Optional[int] = None,
            min_features: int = 2,
            max_features: int = 100,
            max_classes: int = 10,
            min_seq_len: Optional[int] = None,
            max_seq_len: int = 1024,
            log_seq_len: bool = False,
            log_n_features: bool = False,
            seq_len_per_gp: bool = False,
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            replay_small: bool = False,
            prior_type: str = "mlp_scm",
            fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
            sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
            n_jobs: int = -1,
            num_threads_per_generate: int = 1,
            device: str = "cpu",
            config: Optional[PriorConfig] = None,
    ):
        super().__init__(
            regression=regression,
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.prior_type = prior_type
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device
        self.log_n_features = log_n_features
        self.config = config or PriorConfig()

    def hp_sampling(self) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device)
        return hp_sampler.sample()

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Targets tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """

        # print(f'{params["prior_type"]=}')
        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        for idx in range(100):
            X, y = prior_cls(**params)()
            X, y = Reg2Cls(params)(X, y)

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            # Only keep valid datasets with sufficient features and balanced classes
            X, d = self.delete_unique_features(X, d)
            if not (d > 0).all():
                continue

            if not (self.regression or self.cls_sanity_check(X, y, params["train_size"])):
                continue

            #
            # if idx <= 5 and self.config.filter_unpredictable_datasets \
            #         and compute_predictability_pvalue(X[0], y[0], is_classif=not self.regression) >= 0.05:
            if idx <= 5 and should_filter(X[0], y[0], self.config, is_classif=not self.regression):
                # print(f'filter', flush=True)
                continue

            # print(f'no filter', flush=True)
            break

        return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Targets tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
            )
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
                )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            if self.config.seq_lens_multiple_of > 1:
                k = self.config.seq_lens_multiple_of
                gp_seq_len = k * round(gp_seq_len / k)
                gp_train_size = k * round(gp_train_size / k)

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type = self.get_prior()
                if self.log_n_features:
                    subgp_num_features = int(loguniform.rvs(self.min_features, gp_max_features))
                else:
                    subgp_num_features = round(np.random.uniform(self.min_features, gp_max_features))
                subgp_sampled_hp = {k: v() if callable(v) else v for k, v in group_sampled_hp.items()}

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    if self.regression:
                        ds_num_classes = None
                    else:
                        # Each dataset has its own number of classes
                        if np.random.random() > 0.5:
                            ds_num_classes = np.random.randint(2, self.max_classes + 1)
                        else:
                            ds_num_classes = 2

                    # Create parameters dictionary for this dataset
                    params = {
                        # -----------------
                        # Global parameters
                        # -----------------
                        "regression": self.regression,
                        "device": self.device,
                        **self.fixed_hp,  # Fixed HPs
                        # -------------------------
                        # Group-specific parameters
                        # -------------------------
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": gp_max_features if self.seq_len_per_gp else self.max_features,
                        # ----------------------------
                        # Subgroup-specific parameters
                        # ----------------------------
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        # ---------------------------
                        # Dataset-specific parameters
                        # ---------------------------
                        "num_classes": ds_num_classes,
                    }
                    param_list.append(params)

        # Use joblib to generate datasets in parallel.
        # Note: the 'loky' backend does not support nested parallelism during DDP, whereas the 'threading' backend does.
        # However, 'threading' does not respect `inner_max_num_threads`.
        # Therefore, we stick with the 'loky' backend for parallelism, but this requires generating
        # the prior datasets separately from the training process and loading them from disk,
        # rather than generating them on-the-fly.
        # if self.n_jobs == 1:
        #     results = [self.generate_dataset(params) for params in param_list]
        # else:
        #     with joblib.parallel_config(
        #             n_jobs=self.n_jobs, backend="loky", inner_max_num_threads=self.num_threads_per_generate
        #     ):
        #         results = joblib.Parallel()(joblib.delayed(self.generate_dataset)(params) for params in param_list)

        with threadpoolctl.threadpool_limits(limits=1):
            if self.n_jobs == 1:
                n_threads_old = torch.get_num_threads()
                torch.set_num_threads(1)
                results = [self.generate_dataset(params) for params in param_list]
                torch.set_num_threads(n_threads_old)
            else:
                results = run_parallel(self.generate_dataset, args=param_list, n_jobs=self.n_jobs)

        X_list, y_list, d_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

        return X, y, d, seq_lens, train_sizes

    def get_prior(self) -> str:
        """
        Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities.

        Returns
        -------
        str
            The selected prior type name
        """
        if self.prior_type == "mix_scm":
            return np.random.choice(["mlp_scm", "tree_scm"], p=self.fixed_hp.get("mix_probas", [0.7, 0.3]))
        else:
            return self.prior_type


class GraphPrior(Prior):
    """
    Generates synthetic datasets using Graph-based Structural Causal Models (SCM).

    Parameters
    ----------
    regression : bool, default=False
        If True, generates regression datasets. Otherwise, generates classification datasets.

    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    config : PriorConfig | None, default=None
        Prior configuration.

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
            self,
            regression: bool = False,
            batch_size: int = 256,
            batch_size_per_gp: int = 4,
            batch_size_per_subgp: Optional[int] = None,
            min_features: int = 2,
            max_features: int = 100,
            max_classes: int = 10,
            min_seq_len: Optional[int] = None,
            max_seq_len: int = 1024,
            log_seq_len: bool = False,
            seq_len_per_gp: bool = False,
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            replay_small: bool = False,
            config: Optional[PriorConfig] = None,
            n_jobs: int = -1,
            num_threads_per_generate: int = 1,
            device: str = "cpu",
    ):
        super().__init__(
            regression=regression,
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.config = config
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a single valid dataset based on the provided parameters.

        This method implements a robust generation process with:
        1. Multiple retry attempts for generating valid datasets
        2. Feature filtering to remove non-informative columns
        3. Validation checks to ensure class balance in train/test splits

        Parameters
        ----------
        params : dict
            Parameters accepted by `GraphSCM` for dataset generation including:
            - regression: boolean flag for regression/classification
            - seq_len: number of samples
            - train_size: position to split train/test sets
            - num_features: number of features
            - max_features: maximum number of features
            - num_classes: number of target classes
            - device: computation device
            - config: PriorConfig object

        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Targets tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """

        while True:
            X, y = GraphSCM(**params)()

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            # Only keep valid datasets with sufficient features and balanced classes
            X, d = self.delete_unique_features(X, d)
            if not (d > 0).all():
                continue

            if not (self.regression or self.cls_sanity_check(X, y, params["train_size"])):
                continue

            if should_filter(X[0], y[0], self.config, is_classif=not self.regression):
                continue

            return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Targets tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
            )
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
                )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Each subgroup shares the same number of features
                subgp_num_features = round(np.random.uniform(self.min_features, gp_max_features))

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    if self.regression:
                        ds_num_classes = None
                    else:
                        # Each dataset has its own number of classes
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)

                    # Create parameters dictionary for this dataset
                    params = {
                        # -----------------
                        # Global parameters
                        # -----------------
                        "regression": self.regression,
                        "device": self.device,
                        "config": self.config,
                        # -------------------------
                        # Group-specific parameters
                        # -------------------------
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        "max_features": gp_max_features,
                        # ----------------------------
                        # Subgroup-specific parameters
                        # ----------------------------
                        "num_features": subgp_num_features,
                        # ---------------------------
                        # Dataset-specific parameters
                        # ---------------------------
                        "num_classes": ds_num_classes,
                        # ---------------------------
                        # DAG generation parameters
                        # ---------------------------
                    }

                    param_list.append(params)

        with threadpoolctl.threadpool_limits(limits=1):
            if self.n_jobs == 1:
                n_threads_old = torch.get_num_threads()
                torch.set_num_threads(1)
                results = [self.generate_dataset(params) for params in param_list]
                torch.set_num_threads(n_threads_old)
            else:
                results = run_parallel(self.generate_dataset, args=param_list, n_jobs=self.n_jobs)

        X_list, y_list, d_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

        return X, y, d, seq_lens, train_sizes


class DummyPrior(Prior):
    """This class creates purely random data. This is useful for testing and debugging
    without the computational overhead of SCM-based generation.

    Parameters
    ----------
    regression : bool, default=False
        If True, generates regression datasets. Otherwise, generates classification datasets.

    batch_size : int, default=256
        Number of datasets to generate

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    device : str, default="cpu"
        Computation device
    """

    def __init__(
            self,
            regression: bool = False,
            batch_size: int = 256,
            min_features: int = 2,
            max_features: int = 100,
            max_classes: int = 10,
            min_seq_len: Optional[int] = None,
            max_seq_len: int = 1024,
            log_seq_len: bool = False,
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            device: str = "cpu",
    ):
        super().__init__(
            regression=regression,
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Targets tensor of shape (batch_size, seq_len).
            Contains randomly assigned class Targets.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
            All datasets share the same split position.
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(self.min_seq_len, self.max_seq_len, log=self.log_seq_len)
        train_size = self.sample_train_size(self.min_train_size, self.max_train_size, seq_len)

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

        num_classes = np.random.randint(2, self.max_classes + 1)
        if self.regression:
            y = torch.randn(batch_size, seq_len, device=self.device)
        else:
            y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)
        train_sizes = torch.full((batch_size,), train_size, device=self.device)

        return X, y, d, seq_lens, train_sizes


class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    regression : bool, default=False
        If True, generates regression datasets. Otherwise, generates classification datasets.

    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm', or 'dummy'

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models

        2. Dummy: Randomly generated datasets for debugging

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    config : PriorConfig | None, default=None
        Prior configuration when using 'graph_scm' prior_type

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors)

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
            self,
            regression: bool = False,
            batch_size: int = 256,
            batch_size_per_gp: int = 4,
            batch_size_per_subgp: Optional[int] = None,
            min_features: int = 2,
            max_features: int = 100,
            max_classes: int = 10,
            min_seq_len: Optional[int] = None,
            max_seq_len: int = 1024,
            log_seq_len: bool = False,
            log_n_features: bool = False,
            seq_len_per_gp: bool = False,
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            replay_small: bool = False,
            prior_type: str = "mlp_scm",
            scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
            scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
            config: Optional[PriorConfig] = None,
            n_jobs: int = -1,
            num_threads_per_generate: int = 1,
            device: str = "cpu",
    ):
        super().__init__()
        if prior_type == "dummy":
            self.prior = DummyPrior(
                regression=regression,
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
            )
        elif prior_type in ["mlp_scm", "tree_scm", "mix_scm"]:
            self.prior = SCMPrior(
                regression=regression,
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                log_n_features=log_n_features,
                seq_len_per_gp=seq_len_per_gp,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                prior_type=prior_type,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
                config=config,
            )
        elif prior_type == "graph_scm":
            self.prior = GraphPrior(
                regression=regression,
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                seq_len_per_gp=seq_len_per_gp,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                config=config,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', 'graph_scm', or 'dummy'."
            )

        self.regression = regression
        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = 0 if regression else max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        y : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class Targets of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
        """
        return self.prior.get_batch(batch_size or self.batch_size)

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns the next batch from the iterator. Since this is an infinite
        iterator, it never raises StopIteration and instead continuously generates
        new synthetic data batches.
        """
        # with DisablePrinting():
        #     return self.get_batch()
        return self.get_batch()

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        return (
            f"PriorDataset(\n"
            f"  task: {'regression' if self.regression else 'classification'}\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout


def should_filter(
        X: torch.Tensor,
        y: torch.Tensor,
        config: PriorConfig,
        *,
        is_classif: bool,
        n_estimators: int = 25,
        n_boot: int = 200,
) -> bool:
    if not (config.remove_trivial_datasets or config.filter_unpredictable_datasets):
        return False

    if is_classif:
        y_int = y.to(torch.long)
        C = int(y_int.max().item() + 1)
        Y = F.one_hot(y_int, num_classes=C).to(torch.float32)  # (n, C)
        if C == 2:
            # drop one dimension to make it faster
            Y = Y[:, :1]
    else:
        Y = y.reshape(-1, 1).to(torch.float32)  # (n, 1)

    # sklearn needs numpy
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # do the squeeze to avoid a sklearn warning
    et = ExtraTreesRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        random_state=1,  # don't set to 0 because it doesn't give OOB scores for all samples for some 133 <= n <= 257
        max_depth=6,  # make it faster
    ).fit(X_np, np.squeeze(Y_np, axis=-1) if Y_np.shape[-1] == 1 else Y_np)

    Yhat = et.oob_prediction_  # (n, d)
    if len(Yhat.shape) == 1:
        Yhat = Yhat[:, None]
    mask = ~np.isnan(Yhat).any(axis=1)  # keep valid OOB rows
    Yv, Yhatv = Y_np[mask], Yhat[mask]

    baseline = Y_np.mean(axis=0, keepdims=True)  # (1, d)
    imp = ((Yv - baseline) ** 2).sum(axis=1) - ((Yv - Yhatv) ** 2).sum(axis=1)

    baseline_mse = ((Yv - baseline) ** 2).sum(axis=1).mean()
    extratrees_mse = ((Yv - Yhatv) ** 2).sum(axis=1).mean()

    # Vectorized bootstrap (no retraining)
    rng = np.random.default_rng(0)
    n = imp.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))

    if config.remove_trivial_datasets and np.sqrt(extratrees_mse) <= config.trivial_dataset_threshold * np.sqrt(baseline_mse):
        return True
    if config.filter_unpredictable_datasets and pval >= 0.05:
        return True
    return False


def compute_predictability_pvalue(
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        is_classif: bool,
        n_estimators: int = 25,
        n_boot: int = 200,
) -> float:
    """
    Single-fit ExtraTreesRegressor with OOB predictions.
    Metric: MSE (== multiclass Brier; binary up to a constant factor).
    Baseline: mean target (== class-prior vector for classification).
    Returns a one-sided bootstrap p-value for 'model > baseline'.
    """
    if is_classif:
        y_int = y.to(torch.long)
        C = int(y_int.max().item() + 1)
        Y = F.one_hot(y_int, num_classes=C).to(torch.float32)  # (n, C)
        if C == 2:
            # drop one dimension to make it faster
            Y = Y[:, :1]
    else:
        Y = y.reshape(-1, 1).to(torch.float32)  # (n, 1)

    # sklearn needs numpy
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # do the squeeze to avoid a sklearn warning
    et = ExtraTreesRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        random_state=1,  # don't set to 0 because it doesn't give OOB scores for all samples for some 133 <= n <= 257
        max_depth=6,  # make it faster
    ).fit(X_np, np.squeeze(Y_np, axis=-1) if Y_np.shape[-1] == 1 else Y_np)

    Yhat = et.oob_prediction_  # (n, d)
    if len(Yhat.shape) == 1:
        Yhat = Yhat[:, None]
    mask = ~np.isnan(Yhat).any(axis=1)  # keep valid OOB rows
    Yv, Yhatv = Y_np[mask], Yhat[mask]

    baseline = Y_np.mean(axis=0, keepdims=True)  # (1, d)
    imp = ((Yv - baseline) ** 2).sum(axis=1) - ((Yv - Yhatv) ** 2).sum(axis=1)

    # Vectorized bootstrap (no retraining)
    rng = np.random.default_rng(0)
    n = imp.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))
    return pval

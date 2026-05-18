"""Meta-dataset construction for TabICL single-dataset fine-tuning.

A fine-tuning *meta-batch* corresponds to one (context, query) split of a chunk
of the real training dataset, preprocessed into ``n_estimators_finetune``
ensemble variants (different normalizations, feature shuffles, and — for
classification — class-label shuffles). The meta-batch is the unit a single
forward pass consumes: the ensemble dimension is treated as the batch dimension
of :class:`tabicl._model.tabicl.TabICL`, so one meta-batch ≈ one
``TabICL.forward`` call.

Per-epoch chunking with fresh context/query splits keeps the in-context
signal diverse across epochs while reusing the pretrained preprocessing
pipeline from :class:`tabicl._sklearn.preprocessing.EnsembleGenerator`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from tabicl._sklearn.preprocessing import EnsembleGenerator


@dataclass
class MetaBatch:
    """A single context+query meta-batch for fine-tuning.

    The ensemble dimension ``E`` is the outer (batch) dimension of the tensors.
    Each of the ``E`` ensemble members sees the same underlying (context, query)
    row split but a different preprocessing pipeline + feature/class shuffle.

    Attributes
    ----------
    X : Tensor, shape ``(E, T, H)``
        Concatenated context+query features, with ``X[:, :train_size]`` the
        context and ``X[:, train_size:]`` the query. Always float32.

    y_train : Tensor, shape ``(E, train_size)``
        Context labels (per-estimator, with class shuffle already applied for
        classification). Always float32 — the classifier forward casts to long
        where cross-entropy needs integer indices.

    y_query : Tensor, shape ``(E, test_size)``
        Ground-truth query labels aligned to each ensemble member's shuffle
        pattern. Long dtype for classification; float32 (z-normalized) for
        regression.

    train_size : int
        Number of context samples (``X[:, :train_size]``).

    y_scaler_mean, y_scaler_std : float or None
        Regression-only: per-chunk z-norm statistics computed on the context.
        Unused (``None``) for classification.
    """

    X: torch.Tensor
    y_train: torch.Tensor
    y_query: torch.Tensor
    train_size: int
    y_scaler_mean: Optional[float] = None
    y_scaler_std: Optional[float] = None


def count_chunks(
    n_samples: int,
    max_chunk_size: int,
    min_chunk_size: int = 50,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    """Return how many meta-batch chunks this rank will yield per epoch.

    Deterministic and cheap (no actual permutation / tensor work). Useful to
    size a tqdm progress bar and the LR scheduler before starting the epoch.

    Under DDP (``world_size > 1``) chunks are split across ranks with
    drop_last semantics, so every rank yields exactly
    ``global_n_chunks // world_size`` chunks; the tail remainder is dropped
    to keep per-rank step counts equal (required so DDP's per-iteration
    gradient all-reduce does not deadlock). If the global chunk count is
    smaller than ``world_size`` every rank falls back to the full list —
    no parallelism, but no hang either.

    With the defaults ``rank=0, world_size=1`` behavior is unchanged.
    """
    if n_samples <= 0:
        return 0
    if n_samples <= max_chunk_size:
        global_n = 1
    else:
        n_full = n_samples // max_chunk_size
        remainder = n_samples - n_full * max_chunk_size
        global_n = n_full + (1 if remainder >= min_chunk_size else 0)
    del rank  # shard count is the same on every rank under drop_last
    if world_size <= 1 or global_n < world_size:
        return global_n
    return global_n // world_size


def _chunk_indices(
    n_samples: int,
    max_chunk_size: int,
    rng: np.random.Generator,
    *,
    min_chunk_size: int = 50,
) -> List[np.ndarray]:
    """Split ``range(n_samples)`` into randomly-shuffled chunks of at most
    ``max_chunk_size`` samples.

    Tail chunks smaller than ``min_chunk_size`` are dropped unless the whole
    dataset is smaller than ``max_chunk_size`` (in which case we keep the
    single chunk).
    """
    perm = rng.permutation(n_samples)
    if n_samples <= max_chunk_size:
        return [perm]
    n_full = n_samples // max_chunk_size
    remainder = n_samples - n_full * max_chunk_size
    chunks = [perm[i * max_chunk_size : (i + 1) * max_chunk_size] for i in range(n_full)]
    if remainder >= min_chunk_size:
        chunks.append(perm[n_full * max_chunk_size :])
    return chunks


def _split_ctx_query(
    y_chunk: np.ndarray,
    *,
    query_size: int,
    seed: int,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(context_idx, query_idx)`` into the chunk for a single split."""
    n = len(y_chunk)
    query_size = max(1, min(query_size, n - 1))
    splitter_cls = StratifiedShuffleSplit if stratify else ShuffleSplit
    splitter = splitter_cls(n_splits=1, test_size=query_size, random_state=seed)
    # StratifiedShuffleSplit.split only uses X for its length — pass a dummy.
    dummy_X = np.zeros((n, 1))
    ctx_idx, qry_idx = next(splitter.split(dummy_X, y_chunk))
    return ctx_idx, qry_idx


def _build_ensemble_generator(
    *,
    classification: bool,
    n_estimators: int,
    norm_methods,
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    random_state: int,
) -> EnsembleGenerator:
    return EnsembleGenerator(
        classification=classification,
        n_estimators=n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        class_shuffle_method=class_shuffle_method,
        outlier_threshold=outlier_threshold,
        random_state=random_state,
    )


def _build_meta_batch(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    *,
    classification: bool,
    n_estimators: int,
    query_size: int,
    epoch_seed: int,
    chunk_idx: int,
    norm_methods,
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    preprocessing_seed: int,
) -> MetaBatch:
    """Build one MetaBatch from one chunk of the training set.

    Implements the context/query split + per-member preprocessing + class/feature
    shuffling + z-normalization (regression). Returns CPU tensors — the caller
    is responsible for moving them to the device.
    """
    split_seed = epoch_seed + chunk_idx * 7919  # prime offset to decorrelate chunks
    if classification:
        n_classes = int(y_chunk.max()) + 1
        query_size = max(query_size, n_classes)
    ctx_idx, qry_idx = _split_ctx_query(y_chunk, query_size=query_size, seed=split_seed, stratify=classification)

    X_ctx = X_chunk[ctx_idx]
    y_ctx = y_chunk[ctx_idx]
    X_qry = X_chunk[qry_idx]
    y_qry = y_chunk[qry_idx]

    y_mean: Optional[float] = None
    y_std: Optional[float] = None
    if not classification:
        y_mean = float(np.mean(y_ctx))
        y_std = float(np.std(y_ctx))
        if y_std < 1e-8:
            y_std = 1e-8
        y_ctx = (y_ctx - y_mean) / y_std
        y_qry = (y_qry - y_mean) / y_std

    gen = _build_ensemble_generator(
        classification=classification,
        n_estimators=n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=feat_shuffle_method,
        class_shuffle_method=class_shuffle_method,
        outlier_threshold=outlier_threshold,
        random_state=preprocessing_seed,
    )
    gen.fit(X_ctx, y_ctx)
    variants = gen.transform(X_qry, mode="both")

    X_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    y_query_list: list[np.ndarray] = []
    train_size = len(ctx_idx)

    for norm_method, (X_variant, y_variant) in variants.items():
        # X_variant: (E_m, T, H) with T = train_size + test_size
        # y_variant: (E_m, train_size) already class-shuffled for classification
        X_list.append(X_variant)
        y_train_list.append(y_variant)

        shuffle_configs = gen.ensemble_configs_[norm_method]
        for _feat_shuffle, y_pattern in shuffle_configs:
            if classification and y_pattern is not None:
                # Apply the same class remap to the query ground truth so that
                # cross-entropy compares logits to targets in the shuffled label
                # space.
                y_query_list.append(np.asarray(y_pattern)[y_qry.astype(int)])
            else:
                y_query_list.append(y_qry)

    X_tensor = torch.from_numpy(np.concatenate(X_list, axis=0)).float()
    # ``y_train`` is always float32: classification labels get cast to long
    # by the cross-entropy loss site, and the model's train-time forward
    # expects float ICL labels. ``y_query`` uses long for classification
    # (so CE can index) and float32 for regression (z-normalized targets).
    y_train_tensor = torch.from_numpy(np.concatenate(y_train_list, axis=0)).float()
    y_query_dtype = torch.long if classification else torch.float32
    y_query_tensor = torch.from_numpy(np.stack(y_query_list, axis=0)).to(dtype=y_query_dtype)

    return MetaBatch(
        X=X_tensor,
        y_train=y_train_tensor,
        y_query=y_query_tensor,
        train_size=train_size,
        y_scaler_mean=y_mean,
        y_scaler_std=y_std,
    )


def iter_epoch_meta_batches(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    n_estimators: int,
    max_chunk_size: int,
    query_ratio: float,
    epoch_seed: int,
    preprocessing_seed: int,
    norm_methods,
    feat_shuffle_method: str,
    class_shuffle_method: str,
    outlier_threshold: float,
    min_chunk_size: int = 50,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[MetaBatch]:
    """Yield one :class:`MetaBatch` per chunk of ``(X, y)`` for a single epoch.

    Each call regenerates the chunking permutation using ``epoch_seed``, so
    successive epochs see different random chunks / different (context, query)
    splits within each chunk. Preprocessors inside each chunk are seeded with
    ``preprocessing_seed`` (fixed across epochs) so normalization and shuffles
    are stable — only the *samples* seen as context vs query change.

    Under DDP (``world_size > 1``) the chunk list is split across ranks with
    drop_last semantics: rank ``r`` yields chunks ``[r * k, (r + 1) * k)``
    where ``k = global_n_chunks // world_size``. The per-chunk ``chunk_idx``
    passed into :func:`_build_meta_batch` is the *global* index, so every
    rank's sharded output is a bit-identical subset of the single-GPU
    stream (preprocessing seeds derive from ``chunk_idx`` and must not
    depend on rank). If ``global_n_chunks < world_size`` every rank falls
    back to the whole list (replication).
    """
    rng = np.random.default_rng(epoch_seed)
    chunks = _chunk_indices(len(y), max_chunk_size=max_chunk_size, rng=rng, min_chunk_size=min_chunk_size)

    if world_size > 1 and len(chunks) >= world_size:
        per_rank = len(chunks) // world_size
        start = rank * per_rank
        sharded = list(enumerate(chunks))[start : start + per_rank]
    else:
        sharded = list(enumerate(chunks))

    for chunk_idx, indices in sharded:
        X_chunk = X[indices]
        y_chunk = y[indices]
        query_size = max(1, int(len(indices) * query_ratio))
        yield _build_meta_batch(
            X_chunk,
            y_chunk,
            classification=classification,
            n_estimators=n_estimators,
            query_size=query_size,
            epoch_seed=epoch_seed,
            chunk_idx=chunk_idx,
            norm_methods=norm_methods,
            feat_shuffle_method=feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=outlier_threshold,
            preprocessing_seed=preprocessing_seed,
        )


def move_meta_batch(batch: MetaBatch, device: torch.device) -> MetaBatch:
    """Return a copy of ``batch`` with tensors on ``device``."""
    return MetaBatch(
        X=batch.X.to(device, non_blocking=True),
        y_train=batch.y_train.to(device, non_blocking=True),
        y_query=batch.y_query.to(device, non_blocking=True),
        train_size=batch.train_size,
        y_scaler_mean=batch.y_scaler_mean,
        y_scaler_std=batch.y_scaler_std,
    )

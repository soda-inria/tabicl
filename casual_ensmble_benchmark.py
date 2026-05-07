#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_ROOT = Path("data178")
DEFAULT_MODEL_PATH = "tabicl-classifier-v1.1-20250506.ckpt"
DEFAULT_CHECKPOINT_VERSION = "tabicl-classifier-v1.1-20250506.ckpt"
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"

np = None
pd = None


@dataclass
class ResultRow:
    dataset_name: str
    dataset_dir: str
    task_type: Optional[str]
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    n_classes: int
    accuracy: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]
    causal_shuffle_n_features: Optional[int] = None
    causal_shuffle_feature_indices: Optional[str] = None
    causal_selector_seconds: Optional[float] = None
    causal_selector_status: Optional[str] = None
    causal_graph_png: Optional[str] = None
    causal_graph_json: Optional[str] = None


@dataclass
class ModelSummaryRow:
    model_name: str
    model_path: str
    gpu_id: int
    datasets_discovered: int
    ok_count: int
    failed_count: int
    skipped_count: int
    avg_accuracy_ok: Optional[float]
    avg_fit_seconds_ok: Optional[float]
    avg_predict_seconds_ok: Optional[float]
    avg_dataset_seconds_ok: Optional[float]
    total_dataset_seconds_ok: float
    model_wall_seconds: float
    status: str
    error: Optional[str]
    failed_datasets: str


def ensure_runtime_deps() -> None:
    global np
    global pd

    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd

        np = _np
        pd = _pd


def parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    return int(value)


def parse_auto_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered == "auto":
        return None
    return int(value)


def parse_auto_bool(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be one of: auto, true, false")


def parse_kv_cache(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"kv", "repr"}:
        return lowered
    raise argparse.ArgumentTypeError("kv_cache must be one of: false, true, kv, repr")


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_dataset_dirs(data_root: Path) -> List[Path]:
    return [path for path in sorted(data_root.iterdir()) if path.is_dir()]


def collect_torch_diagnostics() -> Dict[str, object]:
    import torch

    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        device_count = f"error: {exc}"

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": device_count,
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def stable_feature_prefix(context: str, fallback: str) -> str:
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(
    values,
    *,
    kind: str,
    prefix: str,
) -> pd.DataFrame:
    ensure_runtime_deps()

    if isinstance(values, pd.DataFrame):
        df = values.copy()
    else:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr)

    if kind == "numeric":
        df = df.apply(pd.to_numeric, errors="coerce")
    elif kind == "categorical":
        df = df.apply(normalize_categorical_series)
    else:
        raise ValueError(f"Unsupported feature kind: {kind}")

    df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]
    return df


def make_target_array(values) -> np.ndarray:
    ensure_runtime_deps()

    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> np.ndarray:
    ensure_runtime_deps()

    suffix = file_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)

    raise ValueError(f"Unsupported split file type: {file_path}")


def find_by_suffix(files: List[Path], suffix: str) -> Path | None:
    lower_suffix = suffix.lower()
    for file_path in files:
        if file_path.name.lower().endswith(lower_suffix):
            return file_path
    return None


def find_split_files(dataset_dir: Path):
    files = [path for path in dataset_dir.iterdir() if path.is_file()]

    n_train = find_by_suffix(files, "n_train.npy")
    c_train = find_by_suffix(files, "c_train.npy")
    y_train = find_by_suffix(files, "y_train.npy")
    n_val = find_by_suffix(files, "n_val.npy")
    c_val = find_by_suffix(files, "c_val.npy")
    y_val = find_by_suffix(files, "y_val.npy")
    n_test = find_by_suffix(files, "n_test.npy")
    c_test = find_by_suffix(files, "c_test.npy")
    y_test = find_by_suffix(files, "y_test.npy")

    if y_train is None or y_test is None:
        raise FileNotFoundError("Missing y_train.npy or y_test.npy")
    if n_train is None and c_train is None:
        raise FileNotFoundError("Missing both N_train.npy and C_train.npy")
    if n_test is None and c_test is None:
        raise FileNotFoundError("Missing both N_test.npy and C_test.npy")

    train_split = (n_train, c_train, y_train)
    val_split = (n_val, c_val, y_val) if y_val is not None and (n_val is not None or c_val is not None) else None
    test_split = (n_test, c_test, y_test)
    return train_split, val_split, test_split


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    ensure_runtime_deps()

    features: List[pd.DataFrame] = []
    feature_prefix = stable_feature_prefix(context, y_path.stem)

    if num_path is not None:
        x_num = load_array(num_path)
        features.append(make_feature_frame(x_num, kind="numeric", prefix=f"{feature_prefix}_n"))
    if cat_path is not None:
        x_cat = load_array(cat_path)
        features.append(make_feature_frame(x_cat, kind="categorical", prefix=f"{feature_prefix}_c"))

    if not features:
        raise ValueError("No feature files found for split")

    n_samples = features[0].shape[0]
    for idx, feature_df in enumerate(features):
        if feature_df.shape[0] != n_samples:
            raise ValueError(
                f"Inconsistent number of rows across feature blocks: block {idx} has "
                f"{feature_df.shape[0]} rows but expected {n_samples}"
            )

    X = features[0] if len(features) == 1 else pd.concat(features, axis=1)
    y = make_target_array(load_array(y_path))

    if len(X) != len(y):
        raise ValueError(f"Feature/target row mismatch: X has {len(X)} rows while y has {len(y)}")

    return X, y


def normalize_model_path(model_path: str | None) -> str | None:
    if model_path is None:
        return None
    if str(model_path).strip().lower() == "none":
        return None

    path = Path(model_path).expanduser()
    try:
        if path.exists():
            path = path.resolve()
    except Exception:
        pass
    return str(path)


def extract_last_int(stem: str) -> int | None:
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    return int(numbers[-1])


def discover_model_paths(models_dir: Path, max_models: int | None = None) -> List[Path]:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")
    if not models_dir.is_dir():
        raise NotADirectoryError(f"Models directory is not a directory: {models_dir}")

    model_paths = [
        path
        for path in models_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".ckpt", ".pt", ".pth"}
    ]
    model_paths.sort(
        key=lambda path: (
            0 if extract_last_int(path.stem) is not None else 1,
            extract_last_int(path.stem) if extract_last_int(path.stem) is not None else path.stem,
            path.stem,
        )
    )
    if max_models is not None:
        model_paths = model_paths[:max_models]
    return model_paths


def preload_model_once(classifier: Any, worker_label: str, verbose: bool) -> None:
    load_fn = getattr(classifier, "_load_model", None)
    if not callable(load_fn):
        return

    t0 = time.perf_counter()
    load_fn()

    if hasattr(classifier, "_get_model_load_key"):
        try:
            classifier._loaded_model_key = classifier._get_model_load_key()
        except Exception:
            pass

    def _skip_reloading():
        return None

    try:
        classifier._load_model = _skip_reloading
    except Exception:
        pass

    if verbose:
        print(
            f"[{worker_label}] model preloaded once in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )


def release_classifier_resources(classifier: Any) -> None:
    if classifier is None:
        return

    try:
        model = getattr(classifier, "model_", None)
        if model is not None:
            clear_cache = getattr(model, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
    except Exception:
        pass

    for attr_name in ("model_kv_cache_", "ensemble_generator_", "X_encoder_", "y_encoder_", "model_"):
        if hasattr(classifier, attr_name):
            try:
                setattr(classifier, attr_name, None)
            except Exception:
                pass


def force_memory_cleanup(device_str: str) -> None:
    gc.collect()

    try:
        import torch
    except Exception:
        return

    if not device_str.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return

    try:
        device = torch.device(device_str)
        with torch.cuda.device(device):
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        pass


class PCMarkovBlanketFeatureSelector:
    """Select target-adjacent features with a lightweight PC/MB-style procedure."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        max_samples: int = 5000,
        max_candidates: int = 128,
        max_cond_set: int = 2,
        min_features: int = 4,
        top_k: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.alpha = alpha
        self.max_samples = max_samples
        self.max_candidates = max_candidates
        self.max_cond_set = max_cond_set
        self.min_features = min_features
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X, y):
        ensure_runtime_deps()

        started_at = time.time()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X/y row mismatch: {X_arr.shape[0]} != {y_arr.shape[0]}")

        n_features = int(X_arr.shape[1])
        if n_features == 0:
            self.selected_indices_ = np.asarray([], dtype=np.int64)
            self.association_scores_ = np.asarray([], dtype=np.float64)
            self.selector_seconds_ = time.time() - started_at
            self.status_ = "empty_input"
            self.causal_graph_ = self._build_causal_graph(
                selected_indices=[],
                candidate_indices=[],
                scores=self.association_scores_,
                n_features=0,
                sample_count=0,
            )
            return self

        sample_idx = self._sample_rows(y_arr)
        X_sample = self._standardize_matrix(X_arr[sample_idx])
        y_sample = self._standardize_vector(y_arr[sample_idx])

        scores = self._target_association_scores(X_sample, y_sample)
        self.association_scores_ = scores

        candidate_count = min(max(1, int(self.max_candidates)), n_features)
        candidate_indices = np.argsort(-scores, kind="mergesort")[:candidate_count]
        selected = self._pc_prune(X_sample, y_sample, candidate_indices, scores)

        min_features = min(max(0, int(self.min_features)), n_features)
        if len(selected) < min_features:
            selected_set = set(int(i) for i in selected)
            for idx in candidate_indices:
                selected_set.add(int(idx))
                if len(selected_set) >= min_features:
                    break
            selected = sorted(selected_set, key=lambda idx: (-scores[idx], idx))

        top_k = self._resolve_top_k(n_features)
        if top_k is not None and len(selected) > top_k:
            selected = sorted(selected, key=lambda idx: (-scores[idx], idx))[:top_k]

        selected = sorted(set(int(i) for i in selected))
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.selector_seconds_ = time.time() - started_at
        self.status_ = (
            f"ok selected={len(selected)} candidates={len(candidate_indices)} "
            f"samples={len(sample_idx)}"
        )
        self.causal_graph_ = self._build_causal_graph(
            selected_indices=selected,
            candidate_indices=candidate_indices,
            scores=scores,
            n_features=n_features,
            sample_count=len(sample_idx),
        )
        return self

    def _build_causal_graph(
        self,
        *,
        selected_indices,
        candidate_indices,
        scores,
        n_features: int,
        sample_count: int,
    ) -> dict[str, object]:
        selected = [int(i) for i in np.asarray(selected_indices, dtype=np.int64).reshape(-1).tolist()]
        candidates = [int(i) for i in np.asarray(candidate_indices, dtype=np.int64).reshape(-1).tolist()]
        scores_arr = np.asarray(scores, dtype=np.float64).reshape(-1)

        feature_nodes = []
        edges = []
        for idx in selected:
            score = float(scores_arr[idx]) if 0 <= idx < scores_arr.size else 0.0
            feature_node = f"feature_{idx}"
            feature_nodes.append(
                {
                    "id": feature_node,
                    "kind": "feature",
                    "feature_index": idx,
                    "association_score": score,
                    "selected": True,
                }
            )
            edges.append(
                {
                    "source": feature_node,
                    "target": "target",
                    "kind": "target_adjacency",
                    "association_score": score,
                }
            )

        return {
            "graph_type": "target_adjacency_skeleton",
            "target_node": "target",
            "nodes": [{"id": "target", "kind": "target"}] + feature_nodes,
            "edges": edges,
            "selected_feature_indices": selected,
            "candidate_feature_indices": candidates,
            "association_scores": {
                str(idx): float(scores_arr[idx])
                for idx in candidates
                if 0 <= idx < scores_arr.size
            },
            "parameters": {
                "alpha": float(self.alpha),
                "max_samples": int(self.max_samples),
                "max_candidates": int(self.max_candidates),
                "max_cond_set": int(self.max_cond_set),
                "min_features": int(self.min_features),
                "top_k": None if self.top_k is None else int(self.top_k),
                "random_state": self.random_state,
            },
            "n_features": int(n_features),
            "sample_count": int(sample_count),
            "selector_seconds": float(getattr(self, "selector_seconds_", 0.0)),
            "status": str(getattr(self, "status_", "unknown")),
        }

    def _resolve_top_k(self, n_features: int) -> int | None:
        if self.top_k is not None:
            return max(0, min(int(self.top_k), n_features))
        if n_features <= 0:
            return 0
        auto_k = max(int(self.min_features), int(math.ceil(2.0 * math.sqrt(n_features))))
        return min(auto_k, int(self.max_candidates), n_features)

    def _sample_rows(self, y_arr) -> np.ndarray:
        n_samples = int(y_arr.shape[0])
        max_samples = int(self.max_samples)
        if max_samples <= 0 or n_samples <= max_samples:
            return np.arange(n_samples, dtype=np.int64)

        rng = np.random.default_rng(self.random_state)
        selected_parts = []
        classes, inverse, counts = np.unique(y_arr, return_inverse=True, return_counts=True)
        remaining = max_samples
        for class_id, count in enumerate(counts):
            class_idx = np.flatnonzero(inverse == class_id)
            quota = max(1, int(round(max_samples * float(count) / float(n_samples))))
            quota = min(quota, int(class_idx.size), remaining)
            if quota > 0:
                selected_parts.append(rng.choice(class_idx, size=quota, replace=False))
                remaining -= quota
            if remaining <= 0:
                break

        if remaining > 0:
            already = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=np.int64)
            mask = np.ones(n_samples, dtype=bool)
            mask[already] = False
            pool = np.flatnonzero(mask)
            if pool.size:
                extra = rng.choice(pool, size=min(remaining, int(pool.size)), replace=False)
                selected_parts.append(extra)

        sampled = np.concatenate(selected_parts) if selected_parts else np.arange(n_samples, dtype=np.int64)
        sampled = np.asarray(sampled, dtype=np.int64)
        sampled.sort()
        return sampled

    @staticmethod
    def _standardize_matrix(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = np.isfinite(col)
            fill = float(np.nanmedian(col[finite])) if finite.any() else 0.0
            col[~finite] = fill
            mean = float(col.mean())
            std = float(col.std())
            if std <= 1e-12:
                X[:, j] = 0.0
            else:
                X[:, j] = (col - mean) / std
        return X

    @staticmethod
    def _standardize_vector(y) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y)
        fill = float(np.nanmedian(y[finite])) if finite.any() else 0.0
        y = y.copy()
        y[~finite] = fill
        std = float(y.std())
        if std <= 1e-12:
            return np.zeros_like(y, dtype=np.float64)
        return (y - float(y.mean())) / std

    @staticmethod
    def _target_association_scores(X, y) -> np.ndarray:
        n = max(1, int(X.shape[0]))
        corr = np.abs(np.dot(X.T, y) / float(n))
        corr[~np.isfinite(corr)] = 0.0
        return corr

    def _pc_prune(self, X, y, candidate_indices, scores) -> List[int]:
        selected = [int(i) for i in candidate_indices if scores[int(i)] > 0.0]
        if not selected:
            return []

        max_cond_set = max(0, int(self.max_cond_set))
        max_conditioning_pool = min(12, max(0, len(selected) - 1))

        for cond_size in range(max_cond_set + 1):
            removed: set[int] = set()
            for feat_idx in list(selected):
                if feat_idx in removed:
                    continue
                others = [idx for idx in selected if idx != feat_idx and idx not in removed]
                others.sort(key=lambda idx: (-scores[idx], idx))

                if cond_size == 0:
                    conditioning_sets = [()]
                else:
                    pool = others[:max_conditioning_pool]
                    if len(pool) < cond_size:
                        continue
                    conditioning_sets = itertools.combinations(pool, cond_size)

                for cond_set in conditioning_sets:
                    p_value = self._partial_corr_pvalue(X[:, feat_idx], y, X[:, list(cond_set)] if cond_set else None)
                    if p_value > self.alpha:
                        removed.add(feat_idx)
                        break

            if removed:
                selected = [idx for idx in selected if idx not in removed]
            if len(selected) <= self.min_features:
                break

        selected.sort(key=lambda idx: (-scores[idx], idx))
        return selected

    @staticmethod
    def _partial_corr_pvalue(x, y, Z=None) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(x.shape[0])
        cond_dim = 0

        if Z is not None and np.asarray(Z).size:
            Z_arr = np.asarray(Z, dtype=np.float64)
            if Z_arr.ndim == 1:
                Z_arr = Z_arr.reshape(-1, 1)
            cond_dim = int(Z_arr.shape[1])
            design = np.column_stack([np.ones(n), Z_arr])
            try:
                beta_x = np.linalg.lstsq(design, x, rcond=None)[0]
                beta_y = np.linalg.lstsq(design, y, rcond=None)[0]
                x = x - design @ beta_x
                y = y - design @ beta_y
            except np.linalg.LinAlgError:
                return 0.0

        x_std = float(x.std())
        y_std = float(y.std())
        if x_std <= 1e-12 or y_std <= 1e-12:
            return 1.0

        r = float(np.dot((x - x.mean()) / x_std, (y - y.mean()) / y_std) / max(1, n))
        r = max(min(r, 0.999999), -0.999999)
        dof = max(1, n - cond_dim - 3)
        z_value = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(dof)
        return math.erfc(abs(z_value) / math.sqrt(2.0))


class PartialFeatureShuffleEnsembleGenerator:
    """Ensemble generator that only permutes a selected subset of columns."""

    def __init__(
        self,
        *,
        classification: bool,
        n_estimators: int,
        shuffle_feature_indices,
        norm_methods: str | List[str] | None = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.classification = classification
        self.n_estimators = n_estimators
        self.shuffle_feature_indices = shuffle_feature_indices
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y):
        from tabicl.sklearn.preprocessing import PreprocessingPipeline, Shuffler, UniqueFeatureFilter
        from tabicl.sklearn.sklearn_utils import validate_data

        ensure_runtime_deps()
        validate_data(self, X, y)

        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power"]
        elif isinstance(self.norm_methods, str):
            self.norm_methods_ = [self.norm_methods]
        else:
            self.norm_methods_ = list(self.norm_methods)

        self.unique_filter_ = UniqueFeatureFilter()
        X = self.unique_filter_.fit_transform(X)

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = int(X.shape[1])
        if self.classification:
            self.n_classes_ = len(np.unique(y))

        kept_original_indices = np.flatnonzero(self.unique_filter_.features_to_keep_)
        requested = set(int(i) for i in np.asarray(self.shuffle_feature_indices, dtype=np.int64).reshape(-1))
        self.shuffle_feature_indices_filtered_ = np.asarray(
            [new_idx for new_idx, old_idx in enumerate(kept_original_indices) if int(old_idx) in requested],
            dtype=np.int64,
        )

        self.ensemble_configs_, self.feature_shuffles_, y_patterns = self._generate_ensemble()
        if self.classification:
            self.class_shuffles_ = y_patterns

        self.preprocessors_ = {}
        for norm_method in self.ensemble_configs_:
            if norm_method not in self.preprocessors_:
                preprocessor = PreprocessingPipeline(
                    normalization_method=norm_method,
                    outlier_threshold=self.outlier_threshold,
                    random_state=self.random_state,
                )
                preprocessor.fit(X)
                self.preprocessors_[norm_method] = preprocessor

        return self

    def _generate_feature_shuffles(self) -> List[List[int]]:
        from tabicl.sklearn.preprocessing import Shuffler

        identity = list(range(self.n_features_in_))
        selected = [int(i) for i in self.shuffle_feature_indices_filtered_.tolist()]
        selected = sorted(set(i for i in selected if 0 <= i < self.n_features_in_))

        if len(selected) <= 1 or self.feat_shuffle_method == "none" or self.n_estimators == 1:
            return [identity]

        sub_shuffler = Shuffler(
            n_elements=len(selected),
            method=self.feat_shuffle_method,
            random_state=self.random_state,
        )
        sub_patterns = sub_shuffler.shuffle(self.n_estimators)
        feature_shuffles = []
        for sub_pattern in sub_patterns:
            pattern = identity.copy()
            for target_pos, selected_offset in zip(selected, sub_pattern):
                pattern[target_pos] = selected[int(selected_offset)]
            feature_shuffles.append(pattern)
        return feature_shuffles

    def _generate_ensemble(self):
        from tabicl.sklearn.preprocessing import Shuffler

        X_shuffles = self._generate_feature_shuffles()

        if self.classification:
            class_shuffler = Shuffler(
                n_elements=self.n_classes_,
                method=self.class_shuffle_method,
                random_state=self.random_state,
            )
            y_patterns = class_shuffler.shuffle(self.n_estimators)
        else:
            y_patterns = [None]

        rng = np.random.default_rng(self.random_state)
        shuffle_configs = list(itertools.product(X_shuffles, y_patterns))
        if shuffle_configs:
            order = rng.permutation(len(shuffle_configs))
            shuffle_configs = [shuffle_configs[int(i)] for i in order]

        shuffle_norm_configs = list(itertools.product(shuffle_configs, self.norm_methods_))
        shuffle_norm_configs = shuffle_norm_configs[: self.n_estimators]

        ensemble_configs = OrderedDict()
        X_shuffle_dict = OrderedDict()
        y_pattern_dict = OrderedDict()
        for method in self.norm_methods_:
            method_shuffle_configs = [config[0] for config in shuffle_norm_configs if config[1] == method]
            if not method_shuffle_configs:
                continue
            X_shuffle_dict[method] = [config[0] for config in method_shuffle_configs]
            y_pattern_dict[method] = [config[1] for config in method_shuffle_configs]
            ensemble_configs[method] = method_shuffle_configs

        return ensemble_configs, X_shuffle_dict, y_pattern_dict

    def transform(self, X=None, mode="both", feature_mask=None):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["ensemble_configs_"])
        assert mode in ("both", "train", "test"), f"Invalid mode: {mode}"

        if feature_mask is not None:
            filtered_mask = feature_mask[self.unique_filter_.features_to_keep_]
            kept_cols = ~filtered_mask
            idx_map = {}
            new_idx = 0
            for old_idx in range(len(filtered_mask)):
                if kept_cols[old_idx]:
                    idx_map[old_idx] = new_idx
                    new_idx += 1

            self.masked_feature_shuffles_ = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                remapped = []
                for feat_shuffle, _ in shuffle_configs:
                    remapped.append([idx_map[i] for i in feat_shuffle if i in idx_map])
                self.masked_feature_shuffles_[norm_method] = remapped

        if mode == "train":
            y = self.y_
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                X_preprocessed = self.preprocessors_[norm_method].X_transformed_
                if feature_mask is not None:
                    X_preprocessed = X_preprocessed[:, kept_cols]
                X_ensemble = []
                y_ensemble = []
                for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                    if feature_mask is not None:
                        feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                    X_ensemble.append(X_preprocessed[:, feat_shuffle])
                    if self.classification:
                        y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                    else:
                        y_ensemble.append(y)
                data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))
            return data

        assert X is not None, "X is required when mode is 'test' or 'both'"
        X = self.unique_filter_.transform(X)

        if feature_mask is not None:
            X = np.array(X, dtype=np.float64)
            X[:, filtered_mask] = 0.0

        if mode == "test":
            data = OrderedDict()
            for norm_method, shuffle_configs in self.ensemble_configs_.items():
                X_test_preprocessed = self.preprocessors_[norm_method].transform(X)
                if feature_mask is not None:
                    X_test_preprocessed = X_test_preprocessed[:, kept_cols]
                X_ensemble = []
                for i, (feat_shuffle, _) in enumerate(shuffle_configs):
                    if feature_mask is not None:
                        feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                    X_ensemble.append(X_test_preprocessed[:, feat_shuffle])
                data[norm_method] = (np.stack(X_ensemble, axis=0),)
            return data

        y = self.y_
        data = OrderedDict()
        for norm_method, shuffle_configs in self.ensemble_configs_.items():
            preprocessor = self.preprocessors_[norm_method]
            X_train_pp = preprocessor.X_transformed_
            X_test_pp = preprocessor.transform(X)
            if feature_mask is not None:
                X_train_pp = X_train_pp[:, kept_cols]
                X_test_pp = X_test_pp[:, kept_cols]
            X_variant = np.concatenate([X_train_pp, X_test_pp], axis=0)
            X_ensemble = []
            y_ensemble = []
            for i, (feat_shuffle, y_pattern) in enumerate(shuffle_configs):
                if feature_mask is not None:
                    feat_shuffle = self.masked_feature_shuffles_[norm_method][i]
                X_ensemble.append(X_variant[:, feat_shuffle])
                if self.classification:
                    y_ensemble.append(np.array(y_pattern)[y.astype(int)])
                else:
                    y_ensemble.append(y)
            data[norm_method] = (np.stack(X_ensemble, axis=0), np.stack(y_ensemble, axis=0))

        return data


class CausalFeatureShuffleTabICLClassifier:
    """TabICL classifier wrapper that restricts feature shuffling to selected columns."""

    def __init__(
        self,
        *,
        causal_alpha: float = 0.05,
        causal_max_samples: int = 5000,
        causal_max_candidates: int = 128,
        causal_max_cond_set: int = 2,
        causal_min_features: int = 4,
        causal_top_k: int | None = None,
        **tabicl_kwargs,
    ) -> None:
        from tabicl import TabICLClassifier

        self._base = TabICLClassifier(**tabicl_kwargs)
        self.causal_alpha = causal_alpha
        self.causal_max_samples = causal_max_samples
        self.causal_max_candidates = causal_max_candidates
        self.causal_max_cond_set = causal_max_cond_set
        self.causal_min_features = causal_min_features
        self.causal_top_k = causal_top_k

    def __getattr__(self, name: str):
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)

    def __setattr__(self, name: str, value):
        if name == "_base" or name.startswith("causal_"):
            object.__setattr__(self, name, value)
        elif "_base" in self.__dict__:
            setattr(self._base, name, value)
        else:
            object.__setattr__(self, name, value)

    def fit(self, X, y):
        import torch
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.multiclass import check_classification_targets
        from tabicl.sklearn.classifier import TabICLClassifier
        from tabicl.sklearn.preprocessing import TransformToNumerical
        from tabicl.sklearn.sklearn_utils import _num_samples, validate_data

        base: TabICLClassifier = self._base

        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")

        X, y = validate_data(base, X, y, dtype=None, skip_check_array=True)
        check_classification_targets(y)

        base._resolve_device()
        base.n_samples_in_ = _num_samples(X)
        base._build_inference_config()

        base._load_model()
        base.model_.to(base.device_)

        base.y_encoder_ = LabelEncoder()
        y_encoded = base.y_encoder_.fit_transform(y)
        base.classes_ = base.y_encoder_.classes_
        base.n_classes_ = len(base.y_encoder_.classes_)

        if base.n_classes_ > base.model_.max_classes:
            if base.kv_cache:
                raise ValueError(
                    f"KV caching is not supported when the number of classes ({base.n_classes_}) exceeds the max number "
                    f"of classes ({base.model_.max_classes}) natively supported by the model."
                )
            if not base.support_many_classes:
                raise ValueError(
                    f"The number of classes ({base.n_classes_}) exceeds the max number of classes ({base.model_.max_classes}) "
                    f"natively supported by the model. Consider enabling many-class support which performs mixed-radix "
                    f"ensembling during column-wise embedding and hierarchical classification during in-context learning."
                )
            if base.verbose:
                print(
                    f"The number of classes ({base.n_classes_}) exceeds the max number of classes ({base.model_.max_classes}) "
                    f"natively supported by the model. Therefore, many-class strategy is enabled."
                )

        base.X_encoder_ = TransformToNumerical(verbose=base.verbose)
        X_encoded = base.X_encoder_.fit_transform(X)

        selector = PCMarkovBlanketFeatureSelector(
            alpha=self.causal_alpha,
            max_samples=self.causal_max_samples,
            max_candidates=self.causal_max_candidates,
            max_cond_set=self.causal_max_cond_set,
            min_features=self.causal_min_features,
            top_k=self.causal_top_k,
            random_state=base.random_state,
        )
        selector.fit(X_encoded, y_encoded)

        self.causal_selector_ = selector
        self.causal_shuffle_feature_indices_ = selector.selected_indices_
        self.causal_shuffle_feature_count_ = int(selector.selected_indices_.size)
        self.causal_selector_seconds_ = float(selector.selector_seconds_)
        self.causal_selector_status_ = str(selector.status_)
        self.causal_graph_ = selector.causal_graph_

        base.ensemble_generator_ = PartialFeatureShuffleEnsembleGenerator(
            classification=True,
            n_estimators=base.n_estimators,
            norm_methods=base.norm_methods or ["none", "power"],
            feat_shuffle_method=base.feat_shuffle_method,
            class_shuffle_method=base.class_shuffle_method,
            outlier_threshold=base.outlier_threshold,
            random_state=base.random_state,
            shuffle_feature_indices=selector.selected_indices_,
        )
        base.ensemble_generator_.fit(X_encoded, y_encoded)

        base.model_kv_cache_ = None
        if base.kv_cache:
            if base.kv_cache is True or base.kv_cache == "kv":
                base.cache_mode_ = "kv"
            elif base.kv_cache == "repr":
                base.cache_mode_ = "repr"
            else:
                raise ValueError(f"Invalid kv_cache value '{base.kv_cache}'. Expected False, True, 'kv', or 'repr'.")
            base._build_kv_cache()

        return self

    def predict(self, X):
        return self._base.predict(X)

    def predict_proba(self, X):
        return self._base.predict_proba(X)


def build_model_summary_row(
    model_path: Path,
    gpu_id: int,
    dataset_dirs: List[Path],
    rows: List[ResultRow],
    model_wall_seconds: float,
    error: str | None = None,
) -> ModelSummaryRow:
    ensure_runtime_deps()

    result_df = (
        pd.DataFrame([asdict(row) for row in rows])
        if rows
        else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    )
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()

    avg_fit_seconds_ok = float(ok_df["fit_seconds"].mean()) if len(ok_df) else None
    avg_predict_seconds_ok = float(ok_df["predict_seconds"].mean()) if len(ok_df) else None
    total_dataset_seconds_ok = (
        float((ok_df["fit_seconds"] + ok_df["predict_seconds"]).sum())
        if len(ok_df)
        else 0.0
    )
    avg_dataset_seconds_ok = (
        float((ok_df["fit_seconds"] + ok_df["predict_seconds"]).mean())
        if len(ok_df)
        else None
    )
    failed_datasets = ",".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else ""

    if error is not None:
        return ModelSummaryRow(
            model_name=model_path.stem,
            model_path=model_path.as_posix(),
            gpu_id=gpu_id,
            datasets_discovered=len(dataset_dirs),
            ok_count=0,
            failed_count=len(dataset_dirs),
            skipped_count=0,
            avg_accuracy_ok=None,
            avg_fit_seconds_ok=None,
            avg_predict_seconds_ok=None,
            avg_dataset_seconds_ok=None,
            total_dataset_seconds_ok=0.0,
            model_wall_seconds=float(model_wall_seconds),
            status="fail",
            error=error,
            failed_datasets=",".join(path.name for path in dataset_dirs),
        )

    return ModelSummaryRow(
        model_name=model_path.stem,
        model_path=model_path.as_posix(),
        gpu_id=gpu_id,
        datasets_discovered=len(dataset_dirs),
        ok_count=int(len(ok_df)),
        failed_count=int(len(failed_df)),
        skipped_count=int(len(skipped_df)),
        avg_accuracy_ok=(float(ok_df["accuracy"].mean()) if len(ok_df) else None),
        avg_fit_seconds_ok=avg_fit_seconds_ok,
        avg_predict_seconds_ok=avg_predict_seconds_ok,
        avg_dataset_seconds_ok=avg_dataset_seconds_ok,
        total_dataset_seconds_ok=total_dataset_seconds_ok,
        model_wall_seconds=float(model_wall_seconds),
        status="ok" if len(ok_df) else "fail",
        error=None if len(ok_df) else "No successful datasets processed",
        failed_datasets=failed_datasets,
    )


class BackgroundPrefetcher:
    def __init__(self, enabled: bool, verbose: bool) -> None:
        self.enabled = bool(enabled)
        self.verbose = verbose
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._seen: set[str] = set()
        self._thread: threading.Thread | None = None

        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def schedule(self, model_paths: List[Path | str]) -> None:
        if not self.enabled:
            return

        for model_path in model_paths:
            key = str(Path(model_path).resolve())
            if key in self._seen:
                continue
            self._seen.add(key)
            self._queue.put(key)

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            model_path = self._queue.get()
            if model_path is None:
                return

            try:
                with open(model_path, "rb") as handle:
                    while handle.read(8 * 1024 * 1024):
                        pass
                if self.verbose:
                    print(f"[prefetch] warmed page cache for {model_path}", flush=True)
            except Exception as exc:
                print(f"[prefetch] warning: failed to warm {model_path}: {exc}", flush=True)


def get_causal_diagnostics(classifier) -> dict[str, object]:
    indices = getattr(classifier, "causal_shuffle_feature_indices_", None)
    if indices is None:
        indices_text = None
    else:
        indices_text = ",".join(str(int(i)) for i in np.asarray(indices).reshape(-1).tolist())

    return {
        "causal_shuffle_n_features": getattr(classifier, "causal_shuffle_feature_count_", None),
        "causal_shuffle_feature_indices": indices_text,
        "causal_selector_seconds": getattr(classifier, "causal_selector_seconds_", None),
        "causal_selector_status": getattr(classifier, "causal_selector_status_", None),
        "causal_graph_png": None,
        "causal_graph_json": None,
    }


def safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return stem or "causal_graph"


def render_causal_graph_png(causal_graph: dict[str, object], png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import networkx as nx

    nodes = list(causal_graph.get("nodes", []))
    edges = list(causal_graph.get("edges", []))
    target_node = str(causal_graph.get("target_node", "target"))
    feature_nodes = [node for node in nodes if node.get("kind") == "feature"]

    graph = nx.Graph()
    graph.add_node(target_node)
    for node in feature_nodes:
        graph.add_node(str(node["id"]))
    for edge in edges:
        graph.add_edge(str(edge["source"]), str(edge["target"]))

    pos = {target_node: (0.0, 0.0)}
    count = max(1, len(feature_nodes))
    radius = 2.5 if count > 1 else 1.8
    for item_idx, node in enumerate(feature_nodes):
        angle = 2.0 * math.pi * item_idx / count
        pos[str(node["id"])] = (radius * math.cos(angle), radius * math.sin(angle))

    width = min(18.0, max(7.0, 1.1 * max(4, count)))
    height = min(18.0, max(6.0, 0.9 * max(4, count)))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()

    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.8, alpha=0.55, edge_color="#586070")
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[target_node],
        node_color="#d95f02",
        node_size=1700,
        linewidths=1.4,
        edgecolors="#4c2700",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[str(node["id"]) for node in feature_nodes],
        node_color="#1b9e77",
        node_size=1050,
        linewidths=1.0,
        edgecolors="#0b3d2e",
        ax=ax,
    )

    labels = {target_node: "target"}
    for node in feature_nodes:
        feature_index = int(node.get("feature_index", -1))
        score = float(node.get("association_score", 0.0))
        labels[str(node["id"])] = f"f{feature_index}\nr={score:.3f}"
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, font_color="#20242a", ax=ax)

    dataset_name = causal_graph.get("dataset_name")
    title = "target adjacency causal graph"
    if dataset_name:
        title = f"{dataset_name} - {title}"
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_causal_graph_files(
    classifier,
    graph_dir: Path | None,
    dataset_name: str,
) -> tuple[str | None, str | None]:
    if graph_dir is None:
        return None, None

    causal_graph = getattr(classifier, "causal_graph_", None)
    if not isinstance(causal_graph, dict):
        return None, None

    graph_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_file_stem(dataset_name)
    png_path = graph_dir / f"{stem}.png"
    json_path = graph_dir / f"{stem}.json"

    payload = dict(causal_graph)
    payload["dataset_name"] = dataset_name

    json_result = None
    png_result = None
    try:
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        json_result = json_path.as_posix()
    except Exception:
        json_result = None

    try:
        render_causal_graph_png(payload, png_path)
        png_result = png_path.as_posix()
    except Exception:
        png_result = None

    return png_result, json_result


def evaluate_one_dataset(classifier, dataset_dir: Path, causal_graph_dir: Path | None = None) -> ResultRow:
    ensure_runtime_deps()

    task_type: Optional[str] = None
    try:
        info = load_dataset_info(dataset_dir)
        task_type = str(info.get("task_type", "")).lower() if info else None
        if task_type not in CLASSIFICATION_TASKS:
            return ResultRow(
                dataset_name=dataset_dir.name,
                dataset_dir=dataset_dir.as_posix(),
                task_type=task_type,
                n_train=0,
                n_val=0,
                n_test=0,
                n_features=0,
                n_classes=0,
                accuracy=None,
                fit_seconds=0.0,
                predict_seconds=0.0,
                status="skip",
                error=f"Skipped due to task_type={task_type!r}",
            )

        train_split, val_split, test_split = find_split_files(dataset_dir)

        X_train, y_train = load_split(
            train_split[0],
            train_split[1],
            train_split[2],
            context=f"{dataset_dir.name}-train",
        )
        train_count = int(len(y_train))

        val_count = 0
        if val_split is not None:
            X_val, y_val = load_split(
                val_split[0],
                val_split[1],
                val_split[2],
                context=f"{dataset_dir.name}-val",
            )
            val_count = int(len(y_val))
            X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)

        X_test, y_test = load_split(
            test_split[0],
            test_split[1],
            test_split[2],
            context=f"{dataset_dir.name}-test",
        )

        classes = pd.unique(pd.Series(np.concatenate([np.asarray(y_train), np.asarray(y_test)], axis=0)))

        t0 = time.time()
        classifier.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_pred = classifier.predict(X_test)
        predict_seconds = time.time() - t1

        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
        causal_graph_png, causal_graph_json = write_causal_graph_files(
            classifier,
            causal_graph_dir,
            dataset_dir.name,
        )
        causal_diag = get_causal_diagnostics(classifier)
        causal_diag["causal_graph_png"] = causal_graph_png
        causal_diag["causal_graph_json"] = causal_graph_json

        return ResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=int(len(y_train)),
            n_val=val_count,
            n_test=int(len(y_test)),
            n_features=int(X_train.shape[1]),
            n_classes=int(len(classes)),
            accuracy=accuracy,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
            **causal_diag,
        )
    except Exception as exc:
        return ResultRow(
            dataset_name=dataset_dir.name,
            dataset_dir=dataset_dir.as_posix(),
            task_type=task_type,
            n_train=0,
            n_val=0,
            n_test=0,
            n_features=0,
            n_classes=0,
            accuracy=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def worker_main(
    worker_id: int,
    gpu_id: int,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    causal_graph_dir: str | None,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        gpu_id_str = str(gpu_id)
        # Restrict visibility for both ROCm and CUDA hosts. On AMD boxes
        # ROCR_VISIBLE_DEVICES is the effective selector, while on NVIDIA/CUDA
        # hosts we must also set CUDA_VISIBLE_DEVICES to bind each worker to
        # its assigned physical GPU before using device="cuda:0" internally.
        os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import torch
        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = "cuda:0"
        classifier = CausalFeatureShuffleTabICLClassifier(**worker_kwargs)

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "status": "ready",
                "assigned_count": len(assigned_dataset_dirs),
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        graph_dir = Path(causal_graph_dir) if causal_graph_dir is not None else None
        for dataset_dir in assigned_dataset_dirs:
            row = evaluate_one_dataset(classifier, Path(dataset_dir), graph_dir)
            rows.append(row)

            if verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[ok] {row.dataset_name} accuracy={row.accuracy:.6f}"
                    )
                elif row.status == "skip":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[skip] {row.dataset_name} reason={row.error}"
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[fail] {row.dataset_name} error={row.error}"
                    )

        pd.DataFrame([asdict(row) for row in rows]).to_csv(worker_out_csv, index=False)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass

        ensure_runtime_deps()
        crash_row = pd.DataFrame(
            [
                {
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "dataset_dir": "__worker__",
                    "task_type": None,
                    "n_train": 0,
                    "n_val": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "n_classes": 0,
                    "accuracy": None,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def model_pool_worker_main(
    worker_id: int,
    gpu_id: int,
    dataset_dirs: List[str],
    ready_queue,
    task_queue,
    result_queue,
    base_model_kwargs: Dict,
    causal_graph_root: str | None,
    verbose: bool,
) -> None:
    device_str = "cuda:0"
    worker_label = f"worker {worker_id} | gpu {gpu_id}"

    try:
        ensure_runtime_deps()
        # Keep the per-worker device mapping consistent across ROCm and CUDA.
        os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import torch
        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "status": "ready",
                "assigned_count": len(dataset_dirs),
            }
        )

        resolved_dataset_dirs = [Path(item) for item in dataset_dirs]
        while True:
            task = task_queue.get()
            if task is None:
                return

            model_path = Path(str(task["model_path"]))
            graph_dir = (
                Path(causal_graph_root) / safe_file_stem(model_path.stem)
                if causal_graph_root is not None
                else None
            )
            started_at = time.time()
            classifier = None

            try:
                worker_kwargs = dict(base_model_kwargs)
                worker_kwargs["device"] = device_str
                worker_kwargs["model_path"] = normalize_model_path(str(model_path))
                classifier = CausalFeatureShuffleTabICLClassifier(**worker_kwargs)
                preload_model_once(classifier, worker_label, verbose)

                rows: List[ResultRow] = []
                for dataset_dir in resolved_dataset_dirs:
                    row = evaluate_one_dataset(classifier, dataset_dir, graph_dir)
                    rows.append(row)

                    if verbose:
                        if row.status == "ok":
                            print(
                                f"[{worker_label}] [{model_path.stem}] "
                                f"[ok] {row.dataset_name} accuracy={row.accuracy:.6f}",
                                flush=True,
                            )
                        elif row.status == "skip":
                            print(
                                f"[{worker_label}] [{model_path.stem}] "
                                f"[skip] {row.dataset_name} reason={row.error}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[{worker_label}] [{model_path.stem}] "
                                f"[fail] {row.dataset_name} error={row.error}",
                                flush=True,
                            )

                summary = build_model_summary_row(
                    model_path=model_path,
                    gpu_id=gpu_id,
                    dataset_dirs=resolved_dataset_dirs,
                    rows=rows,
                    model_wall_seconds=time.time() - started_at,
                )
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "status": "model_done",
                        "summary": asdict(summary),
                    }
                )
            except Exception as exc:
                summary = build_model_summary_row(
                    model_path=model_path,
                    gpu_id=gpu_id,
                    dataset_dirs=resolved_dataset_dirs,
                    rows=[],
                    model_wall_seconds=time.time() - started_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "status": "model_done",
                        "summary": asdict(summary),
                    }
                )
            finally:
                release_classifier_resources(classifier)
                force_memory_cleanup(device_str)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass

        try:
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "worker_crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass


def write_summary(
    summary_path: Path,
    result_df: pd.DataFrame,
    dataset_dirs: List[Path],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()

    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        (
            f"avg_accuracy_ok: {ok_df['accuracy'].mean():.6f}"
            if len(ok_df)
            else "avg_accuracy_ok: (none)"
        ),
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(failed_df["dataset_name"].astype(str).tolist())
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    if len(skipped_df):
        skipped_names = ", ".join(skipped_df["dataset_name"].astype(str).tolist())
        lines.append(f"skipped_datasets: {skipped_names}")
    else:
        lines.append("skipped_datasets: (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_pool_outputs(
    out_dir: Path,
    model_summaries: List[dict[str, Any]],
    wall_seconds: float,
) -> None:
    ensure_runtime_deps()

    summary_df = (
        pd.DataFrame(model_summaries)
        if model_summaries
        else pd.DataFrame(columns=ModelSummaryRow.__annotations__.keys())
    )
    for column in (
        "avg_accuracy_ok",
        "avg_fit_seconds_ok",
        "avg_predict_seconds_ok",
        "avg_dataset_seconds_ok",
        "total_dataset_seconds_ok",
        "model_wall_seconds",
    ):
        if column in summary_df.columns:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")
    all_csv = out_dir / "all_models_summary.csv"
    summary_txt = out_dir / "summary.txt"
    summary_df.to_csv(all_csv, index=False)

    ok_df = summary_df[summary_df["status"] == "ok"].copy() if len(summary_df) else pd.DataFrame()
    failed_df = summary_df[summary_df["status"] == "fail"].copy() if len(summary_df) else pd.DataFrame()

    lines = [
        f"total_models: {len(summary_df)}",
        f"successful_models: {len(ok_df)}",
        f"failed_models: {len(failed_df)}",
        (
            f"average_avg_dataset_seconds_ok: {ok_df['avg_dataset_seconds_ok'].mean():.6f}"
            if len(ok_df)
            else "average_avg_dataset_seconds_ok: (none)"
        ),
        (
            f"average_avg_accuracy_ok: {ok_df['avg_accuracy_ok'].mean():.6f}"
            if len(ok_df)
            else "average_avg_accuracy_ok: (none)"
        ),
        f"global_wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        lines.append(
            "failed_models_list: "
            + ", ".join(failed_df["model_name"].astype(str).tolist())
        )
    else:
        lines.append("failed_models_list: (none)")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TabICLv1.1 classification benchmarks with causal feature "
            "selection restricted feature-shuffle ensembles."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--out-dir", default="baseline/iclv1.1_casual_ensmble32")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument(
        "--n-estimators",
        "--ensemble",
        "--ensmble",
        dest="n_estimators",
        type=int,
        default=32,
        help="Number of TabICL ensemble estimators. --ensemble/--ensmble are aliases for this value.",
    )
    parser.add_argument("--batch-size", type=parse_optional_int, default=8)
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--causal-alpha", type=float, default=0.05)
    parser.add_argument("--causal-max-samples", type=int, default=1000000)
    parser.add_argument("--causal-max-candidates", type=int, default=128)
    parser.add_argument("--causal-max-cond-set", type=int, default=2)
    parser.add_argument("--causal-min-features", type=int, default=5)
    parser.add_argument("--causal-top-k", type=parse_auto_int, default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--prefetch-models", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")
    return gpu_ids


def build_common_model_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "kv_cache": args.kv_cache,
        "allow_auto_download": True,
        "checkpoint_version": args.checkpoint_version,
        "use_amp": args.use_amp,
        "use_fa3": args.use_fa3,
        "offload_mode": args.offload_mode,
        "random_state": args.random_state,
        "causal_alpha": args.causal_alpha,
        "causal_max_samples": args.causal_max_samples,
        "causal_max_candidates": args.causal_max_candidates,
        "causal_max_cond_set": args.causal_max_cond_set,
        "causal_min_features": args.causal_min_features,
        "causal_top_k": args.causal_top_k,
    }


def run_single_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    out_dir: Path,
) -> None:
    model_kwargs = build_common_model_kwargs(args)
    model_kwargs["model_path"] = normalize_model_path(args.model_path or DEFAULT_MODEL_PATH)

    start_time = time.time()
    ready_queue: mp.Queue = mp.Queue()
    start_event = mp.Event()

    worker_csv_paths: List[Path] = []
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        assigned_dirs = [str(path.resolve()) for path in dataset_dirs[worker_id::args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)

        proc = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                assigned_dirs,
                ready_queue,
                start_event,
                str(worker_csv),
                dict(model_kwargs),
                str(out_dir / "causal_graphs"),
                args.verbose,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    ready_workers: set[int] = set()
    while len(ready_workers) < args.workers:
        try:
            message = ready_queue.get(timeout=10)
        except Exception:
            dead_workers = [
                str(idx)
                for idx, proc in enumerate(processes)
                if not proc.is_alive() and idx not in ready_workers
            ]
            if dead_workers:
                raise RuntimeError(
                    "Some workers exited before initialization completed: "
                    + ", ".join(dead_workers)
                )
            continue

        if message.get("status") == "ready":
            ready_workers.add(int(message["worker_id"]))
            if args.verbose:
                print(
                    f"[worker {message['worker_id']} | gpu {message['gpu_id']}] "
                    f"ready assigned={message.get('assigned_count', '?')}"
                )
            continue

        if message.get("status") == "crash":
            raise RuntimeError(
                f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                f"during initialization:\n{message.get('error', '(no traceback)')}"
            )

    start_event.set()

    for proc in processes:
        proc.join()

    dfs: List[pd.DataFrame] = []
    for worker_csv in worker_csv_paths:
        if worker_csv.exists():
            dfs.append(pd.read_csv(worker_csv))

    all_df = (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    )
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(summary_txt, all_df, dataset_dirs, wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))


def run_multi_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    out_dir: Path,
) -> None:
    model_paths = discover_model_paths(Path(args.models_dir), max_models=args.max_models)
    if not model_paths:
        raise FileNotFoundError(f"No checkpoint files found under {args.models_dir}")

    worker_count = min(args.workers, len(model_paths))
    base_model_kwargs = build_common_model_kwargs(args)
    ready_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    task_queues: List[mp.Queue] = []
    processes: List[mp.Process] = []
    prefetcher = BackgroundPrefetcher(enabled=args.prefetch_models > 0, verbose=args.verbose)
    closed_workers: set[int] = set()

    try:
        for worker_id in range(worker_count):
            task_queue: mp.Queue = mp.Queue()
            task_queues.append(task_queue)
            proc = mp.Process(
                target=model_pool_worker_main,
                args=(
                    worker_id,
                    gpu_ids[worker_id],
                    [str(path.resolve()) for path in dataset_dirs],
                    ready_queue,
                    task_queue,
                    result_queue,
                    dict(base_model_kwargs),
                    str(out_dir / "causal_graphs"),
                    args.verbose,
                ),
                daemon=False,
            )
            proc.start()
            processes.append(proc)

        ready_workers: set[int] = set()
        while len(ready_workers) < worker_count:
            try:
                message = ready_queue.get(timeout=10)
            except Exception:
                dead_workers = [
                    str(idx)
                    for idx, proc in enumerate(processes)
                    if not proc.is_alive() and idx not in ready_workers
                ]
                if dead_workers:
                    raise RuntimeError(
                        "Some workers exited before initialization completed: "
                        + ", ".join(dead_workers)
                    )
                continue

            if message.get("status") == "ready":
                ready_workers.add(int(message["worker_id"]))
                if args.verbose:
                    print(
                        f"[worker {message['worker_id']} | gpu {message['gpu_id']}] "
                        f"model-pool ready datasets={message.get('assigned_count', '?')}",
                        flush=True,
                    )
                continue

            if message.get("status") == "crash":
                raise RuntimeError(
                    f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                    f"during initialization:\n{message.get('error', '(no traceback)')}"
                )

        next_model_idx = 0
        completed_models = 0
        start_time = time.time()
        collected_summaries: List[dict[str, Any]] = []
        prefetcher.schedule(model_paths[: min(len(model_paths), worker_count + max(0, args.prefetch_models))])

        for worker_id in range(worker_count):
            if next_model_idx >= len(model_paths):
                break
            task_queues[worker_id].put({"model_path": str(model_paths[next_model_idx])})
            next_model_idx += 1
            prefetcher.schedule(model_paths[next_model_idx : next_model_idx + max(0, args.prefetch_models)])

        while completed_models < len(model_paths):
            message = result_queue.get()
            status = message.get("status")

            if status == "worker_crash":
                raise RuntimeError(
                    f"Worker {message.get('worker_id')} on gpu {message.get('gpu_id')} crashed:\n"
                    f"{message.get('error', '(no traceback)')}"
                )

            if status != "model_done":
                continue

            worker_id = int(message["worker_id"])
            summary = dict(message["summary"])
            collected_summaries.append(summary)
            completed_models += 1

            if args.verbose:
                print(
                    f"[worker {worker_id} | gpu {summary['gpu_id']}] "
                    f"finished model={summary['model_name']} status={summary['status']} "
                    f"ok={summary['ok_count']} fail={summary['failed_count']} "
                    f"wall={summary['model_wall_seconds']:.2f}s",
                    flush=True,
                )

            if next_model_idx < len(model_paths):
                task_queues[worker_id].put({"model_path": str(model_paths[next_model_idx])})
                next_model_idx += 1
                prefetcher.schedule(model_paths[next_model_idx : next_model_idx + max(0, args.prefetch_models)])
            elif worker_id not in closed_workers:
                task_queues[worker_id].put(None)
                closed_workers.add(worker_id)

        for worker_id in range(worker_count):
            if worker_id not in closed_workers:
                task_queues[worker_id].put(None)

        for proc in processes:
            proc.join()

        wall_seconds = time.time() - start_time
        write_model_pool_outputs(out_dir, collected_summaries, wall_seconds)

        print(f"saved_all_models_csv: {out_dir / 'all_models_summary.csv'}")
        print(f"saved_summary: {out_dir / 'summary.txt'}")
        print("base_model_kwargs:")
        print(json.dumps(base_model_kwargs, indent=2, ensure_ascii=False))
    finally:
        for worker_id in range(len(task_queues)):
            if worker_id not in closed_workers:
                try:
                    task_queues[worker_id].put(None)
                except Exception:
                    pass
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        prefetcher.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_runtime_deps()

    if args.models_dir is not None and args.model_path is not None:
        raise ValueError("--models-dir and --model-path are mutually exclusive")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    gpu_ids = resolve_gpu_ids(args)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.models_dir is not None:
        run_multi_model_mode(args, dataset_dirs, gpu_ids, out_dir)
    else:
        run_single_model_mode(args, dataset_dirs, gpu_ids, out_dir)


if __name__ == "__main__":
    main()

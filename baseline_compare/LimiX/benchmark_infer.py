#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

CLASSIFICATION_TASKS = {"binclass", "multiclass"}
CATEGORICAL_MISSING_TOKEN = "__tabicl_missing__"


@dataclass
class LoadedDataset:
    dataset_name: str
    dataset_dir: Path
    task_type: Optional[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame | None
    y_val: np.ndarray | None
    X_test: pd.DataFrame
    y_test: np.ndarray
    categorical_feature_indices: list[int]

    @property
    def n_val(self) -> int:
        return 0 if self.y_val is None else int(len(self.y_val))

    @property
    def X_train_merged(self) -> pd.DataFrame:
        if self.X_val is None:
            return self.X_train
        return pd.concat([self.X_train, self.X_val], axis=0, ignore_index=True)

    @property
    def y_train_merged(self) -> np.ndarray:
        if self.y_val is None:
            return self.y_train
        return np.concatenate([np.asarray(self.y_train), np.asarray(self.y_val)], axis=0)

    @property
    def n_train_report(self) -> int:
        return int(len(self.y_train_merged))

    @property
    def n_classes(self) -> int:
        pieces = [np.asarray(self.y_train)]
        if self.y_val is not None:
            pieces.append(np.asarray(self.y_val))
        pieces.append(np.asarray(self.y_test))
        return int(len(pd.unique(pd.Series(np.concatenate(pieces, axis=0)))))


@dataclass
class PredictionResult:
    y_pred: Any
    y_proba: Any | None
    classes: Any | None
    fit_seconds: float
    predict_seconds: float


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
    f1: Optional[float]
    balanced_accuracy: Optional[float]
    roc_auc: Optional[float]
    log_loss: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


RESULT_COLUMNS = list(ResultRow.__annotations__.keys())
OOM_ERROR_MARKERS = (
    "out of memory",
    "oom",
    "cuda out of memory",
)
INVALID_CUDA_CONFIG_MARKERS = (
    "invalid configuration argument",
)


class SkipDataset(Exception):
    pass


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64, copy=False), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(values, -50.0, 50.0))
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), 1e-12, None)


def scores_to_proba(scores: Any, *, n_classes: int | None = None) -> np.ndarray | None:
    try:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim == 1:
            if n_classes != 2:
                return None
            positive = sigmoid(arr)
            return np.column_stack([1.0 - positive, positive])
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            return None
        if n_classes is not None and arr.shape[1] != n_classes:
            return None
        if not np.isfinite(arr).all():
            return None

        row_sums = arr.sum(axis=1, keepdims=True)
        looks_like_proba = (
            np.all(arr >= 0.0)
            and np.all(row_sums[:, 0] > 0.0)
            and np.allclose(row_sums[:, 0], 1.0, atol=1e-4)
        )
        if looks_like_proba:
            return np.clip(arr / np.clip(row_sums, 1e-12, None), 0.0, 1.0)
        return softmax(arr)
    except Exception:
        return None


def compute_weighted_f1(y_true: Any, y_pred: Any) -> Optional[float]:
    try:
        from sklearn.metrics import f1_score

        return float(
            f1_score(
                np.asarray(y_true),
                np.asarray(y_pred),
                average="weighted",
                zero_division=0,
            )
        )
    except Exception:
        return None


def compute_balanced_accuracy(y_true: Any, y_pred: Any) -> Optional[float]:
    try:
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))
    except Exception:
        return None


def compute_roc_auc(
    y_true: Any,
    y_proba: Any | None,
    classes: Any | None,
) -> Optional[float]:
    if y_proba is None:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != len(y_true_arr):
            return None
        if proba_arr.shape[1] < 2 or len(np.unique(y_true_arr)) < 2:
            return None

        class_arr = None if classes is None else np.asarray(classes)
        if class_arr is not None and (
            class_arr.ndim != 1 or len(class_arr) != proba_arr.shape[1]
        ):
            class_arr = None

        if proba_arr.shape[1] == 2:
            if class_arr is not None:
                y_binary = (y_true_arr == class_arr[1]).astype(int)
                if len(np.unique(y_binary)) < 2:
                    return None
                return float(roc_auc_score(y_binary, proba_arr[:, 1]))
            return float(roc_auc_score(y_true_arr, proba_arr[:, 1]))

        if class_arr is not None:
            return float(
                roc_auc_score(
                    y_true_arr,
                    proba_arr,
                    labels=list(class_arr),
                    multi_class="ovr",
                )
            )
        return float(roc_auc_score(y_true_arr, proba_arr, multi_class="ovr"))
    except Exception:
        return None


def compute_log_loss(
    y_true: Any,
    y_proba: Any | None,
    classes: Any | None,
) -> Optional[float]:
    if y_proba is None:
        return None
    try:
        from sklearn.metrics import log_loss

        y_true_arr = np.asarray(y_true)
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim != 2 or proba_arr.shape[0] != len(y_true_arr):
            return None

        class_arr = None if classes is None else np.asarray(classes)
        if class_arr is not None and (
            class_arr.ndim != 1 or len(class_arr) != proba_arr.shape[1]
        ):
            class_arr = None
        if class_arr is not None:
            return float(log_loss(y_true_arr, proba_arr, labels=list(class_arr)))
        return float(log_loss(y_true_arr, proba_arr))
    except Exception:
        return None


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    try:
        if not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
    except Exception:
        return "nan"


def path_candidates(value: str | Path) -> list[Path]:
    path = Path(value).expanduser()
    if path.is_absolute():
        return [path]
    return [
        Path.cwd() / path,
        SCRIPT_DIR / path,
        SCRIPT_DIR.parent.parent / path,
    ]


def resolve_flexible_path(value: str | Path, *, must_exist: bool = False) -> Path:
    candidates = path_candidates(value)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    resolved = (candidates[1] if len(candidates) > 1 else candidates[0]).resolve()
    if must_exist:
        checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError(f"Path does not exist: {value} (checked: {checked})")
    return resolved


def resolve_script_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    script_path = SCRIPT_DIR / path
    repo_path = SCRIPT_DIR.parent.parent / path

    first_part = path.parts[0] if path.parts else ""
    if first_part and first_part not in {".", ".."}:
        if (Path.cwd() / first_part).exists():
            return cwd_path.resolve()
        if (SCRIPT_DIR / first_part).exists():
            return script_path.resolve()
        if (SCRIPT_DIR.parent.parent / first_part).exists():
            return repo_path.resolve()

    if cwd_path.exists():
        return cwd_path.resolve()
    if script_path.exists():
        return script_path.resolve()
    if repo_path.exists():
        return repo_path.resolve()
    return script_path.resolve()


def _uses_socks_proxy(value: str) -> bool:
    return value.lower().startswith(("socks://", "socks4://", "socks4a://", "socks5://", "socks5h://"))


def prepare_hf_proxy_environment() -> None:
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    socks_keys = [key for key in proxy_keys if os.environ.get(key, "").strip() and _uses_socks_proxy(os.environ[key].strip())]
    if not socks_keys or importlib.util.find_spec("socksio") is not None:
        return

    has_http_proxy = any(
        os.environ.get(key, "").strip() and not _uses_socks_proxy(os.environ[key].strip())
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    )
    if has_http_proxy:
        for key in socks_keys:
            os.environ.pop(key, None)
        print(
            "warning: removed SOCKS proxy env vars for Hugging Face download because socksio is not installed; "
            "using HTTP(S)_PROXY instead",
            file=sys.stderr,
            flush=True,
        )
        return

    raise RuntimeError(
        "Hugging Face download is configured to use a SOCKS proxy, but socksio is not installed. "
        "Install socksio/httpx[socks], or pass --model-path pointing to a local LimiX checkpoint."
    )


class LimiXAdapter:
    def __init__(self, args: argparse.Namespace, device: str) -> None:
        self.args = args
        self.device = device
        self._predictor = None
        self._load_error: Exception | None = None
        if args.hf_endpoint:
            os.environ["HF_ENDPOINT"] = args.hf_endpoint

    @staticmethod
    def _script_relative_path(value: str | Path) -> Path:
        return resolve_flexible_path(value)

    @staticmethod
    def _disable_flash_attention_runtime() -> bool:
        try:
            from model import layer as limix_layer
        except Exception:
            return False

        disabled = False
        if hasattr(limix_layer, "FLASH_ATTN_RUNTIME_DISABLED"):
            limix_layer.FLASH_ATTN_RUNTIME_DISABLED = True
            disabled = True
        if getattr(limix_layer, "HAVE_FLASH_ATTN", False):
            limix_layer.HAVE_FLASH_ATTN = False
            disabled = True
        return disabled

    def reset_predictor(self) -> None:
        self._predictor = None
        self._load_error = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def fit_predict_with_flash_attention_disabled(
        self,
        loaded: LoadedDataset,
    ) -> PredictionResult:
        disabled = self._disable_flash_attention_runtime()
        self.reset_predictor()
        if self.args.verbose:
            if disabled:
                print(
                    f"[flash_attn_retry] {loaded.dataset_name}: retrying after disabling flash attention",
                    flush=True,
                )
            else:
                print(
                    f"[flash_attn_retry] {loaded.dataset_name}: disable hook unavailable, retrying with fresh predictor",
                    flush=True,
                )
        return self.fit_predict(loaded)

    def _find_local_model_path(self) -> Path | None:
        candidates: list[Path] = []
        for cache_dir in path_candidates(self.args.model_cache_dir):
            candidates.append(cache_dir / self.args.hf_filename)
        candidates.append(SCRIPT_DIR / self.args.hf_filename)

        repo_cache_name = f"models--{self.args.hf_repo.replace('/', '--')}"
        cache_roots: list[Path] = []
        for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
            if os.environ.get(env_name):
                cache_roots.append(Path(os.environ[env_name]).expanduser())
        if os.environ.get("HF_HOME"):
            cache_roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
        cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        for cache_root in cache_roots:
            pattern = cache_root / repo_cache_name / "snapshots" / "*" / self.args.hf_filename
            candidates.extend(Path(path) for path in glob.glob(str(pattern)))

        existing = [path.resolve() for path in candidates if path.is_file()]
        if not existing:
            return None
        return max(existing, key=lambda path: path.stat().st_mtime)

    def _resolve_model_path(self) -> str:
        if self.args.model_path:
            raw_model_path = str(self.args.model_path).strip()
            if raw_model_path.lower() not in {"auto", "none", "null"}:
                return str(resolve_flexible_path(raw_model_path, must_exist=True))

        local_model = self._find_local_model_path()
        if local_model is not None:
            return str(local_model)

        prepare_hf_proxy_environment()

        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "Failed to import huggingface_hub for LimiX model download. "
                "Install the benchmark requirements, or pass --model-path pointing to a local checkpoint. "
                f"Original error: {exc}"
            ) from exc

        cache_dir = self._script_relative_path(self.args.model_cache_dir)
        return hf_hub_download(
            repo_id=self.args.hf_repo,
            filename=self.args.hf_filename,
            local_dir=str(cache_dir),
            local_files_only=self.args.local_files_only,
        )

    def _resolve_config_path(self) -> Path:
        path = self._script_relative_path(self.args.config_path)
        if not path.exists():
            raise FileNotFoundError(f"LimiX config does not exist: {path}")
        return path

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        if self._load_error is not None:
            raise self._load_error
        try:
            import torch
            from inference.predictor import LimiXPredictor
        except Exception as exc:
            self._load_error = RuntimeError(f"Failed to import LimiX predictor: {exc}")
            raise self._load_error from exc

        try:
            use_ddp_inference = bool(getattr(self.args, "internal_ddp_dataset_dir", None))
            self._predictor = LimiXPredictor(
                device=torch.device(self.device),
                model_path=self._resolve_model_path(),
                inference_config=str(self._resolve_config_path()),
                mix_precision=not self.args.disable_mixed_precision,
                inference_with_DDP=use_ddp_inference,
                seed=self.args.random_state,
            )
        except Exception as exc:
            self._load_error = RuntimeError(f"Failed to load LimiX predictor/model: {exc}")
            raise self._load_error from exc
        return self._predictor

    def _model_max_classes(self, predictor) -> int:
        if self.args.direct_max_classes is not None and self.args.direct_max_classes > 0:
            return int(self.args.direct_max_classes)
        decoder_config = getattr(getattr(predictor, "model", None), "decoder_config", None)
        if isinstance(decoder_config, dict) and "num_classes" in decoder_config:
            return int(decoder_config["num_classes"])
        return 10

    def _predict_direct(
        self,
        predictor,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        prediction = predictor.predict(
            X_train.to_numpy(),
            np.asarray(y_train),
            X_test.to_numpy(),
            task_type="Classification",
        )
        prediction = np.asarray(prediction)
        if prediction.ndim == 1:
            return prediction, None, None

        classes = getattr(predictor, "classes", None)
        if classes is None:
            classes = pd.unique(pd.Series(np.asarray(y_train)))
        classes = np.asarray(classes)
        if classes.ndim != 1 or len(classes) != prediction.shape[1]:
            classes = np.arange(prediction.shape[1])
        y_proba = scores_to_proba(prediction, n_classes=len(classes))
        pred_idx = np.argmax(prediction, axis=1)
        return classes[pred_idx], y_proba, classes

    def _predict_one_vs_rest(
        self,
        predictor,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        train_classes = pd.unique(pd.Series(np.asarray(y_train)))
        scores: list[np.ndarray] = []
        for class_value in train_classes:
            y_binary = (np.asarray(y_train) == class_value).astype(np.int64)
            if len(np.unique(y_binary)) < 2:
                scores.append(np.zeros(len(X_test), dtype=np.float32))
                continue
            prediction = predictor.predict(
                X_train.to_numpy(),
                y_binary,
                X_test.to_numpy(),
                task_type="Classification",
            )
            prediction = np.asarray(prediction)
            if prediction.ndim == 1:
                positive_scores = (prediction == 1).astype(np.float32)
            else:
                classes = np.asarray(getattr(predictor, "classes", np.arange(prediction.shape[1])))
                positive_cols = np.where(classes == 1)[0]
                positive_col = int(positive_cols[0]) if len(positive_cols) else min(1, prediction.shape[1] - 1)
                positive_scores = prediction[:, positive_col]
            scores.append(np.asarray(positive_scores, dtype=np.float32))
        score_matrix = np.column_stack(scores)
        y_proba = scores_to_proba(score_matrix, n_classes=len(train_classes))
        pred_source = y_proba if y_proba is not None else score_matrix
        classes = np.asarray(train_classes)
        return classes[np.argmax(pred_source, axis=1)], y_proba, classes

    def _predict_with_strategy(
        self,
        predictor,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        *,
        model_max_classes: int,
        test_batch_rows: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        n_train_classes = int(len(pd.unique(pd.Series(np.asarray(y_train)))))
        use_ovr = n_train_classes > model_max_classes
        predict_fn = self._predict_one_vs_rest if use_ovr else self._predict_direct

        if test_batch_rows <= 0 or len(X_test) <= test_batch_rows:
            return predict_fn(predictor, X_train, y_train, X_test)

        predictions = []
        probas = []
        classes: np.ndarray | None = None
        for start in range(0, len(X_test), test_batch_rows):
            batch = X_test.iloc[start : start + test_batch_rows].reset_index(drop=True)
            y_pred, y_proba, batch_classes = predict_fn(predictor, X_train, y_train, batch)
            predictions.append(y_pred)
            probas.append(y_proba)
            if classes is None and batch_classes is not None:
                classes = np.asarray(batch_classes)
        if probas and all(proba is not None for proba in probas):
            y_proba = np.concatenate([np.asarray(proba) for proba in probas], axis=0)
        else:
            y_proba = None
        return np.concatenate([np.asarray(pred) for pred in predictions], axis=0), y_proba, classes

    def fit_predict(self, loaded: LoadedDataset) -> PredictionResult:
        y_train = np.asarray(loaded.y_train_merged)
        n_train_classes = len(pd.unique(pd.Series(y_train)))
        n_dataset_classes = loaded.n_classes
        if n_train_classes < 2:
            raise SkipDataset(f"LimiX supports at least 2 train classes; got {n_train_classes}")
        if self.args.max_classes and n_dataset_classes > self.args.max_classes:
            raise SkipDataset(
                f"LimiX supports up to {self.args.max_classes} dataset classes; got {n_dataset_classes}"
            )
        if self.args.max_train_rows > 0 and len(y_train) >= self.args.max_train_rows:
            raise SkipDataset(
                f"LimiX skips train size >= {self.args.max_train_rows}; got {len(y_train)}"
            )

        predictor = self._load_predictor()
        X_train = loaded.X_train_merged
        X_test = loaded.X_test
        model_max_classes = self._model_max_classes(predictor)
        fit_seconds = 0.0

        predict_started = time.time()
        y_pred, y_proba, classes = self._predict_with_strategy(
            predictor,
            X_train,
            y_train,
            X_test,
            model_max_classes=model_max_classes,
            test_batch_rows=self.args.test_batch_rows,
        )
        predict_seconds = time.time() - predict_started

        return PredictionResult(
            y_pred=y_pred,
            y_proba=y_proba,
            classes=classes,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
        )


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_dataset_dirs(data_root: Path) -> list[Path]:
    return [path for path in sorted(data_root.iterdir()) if path.is_dir()]


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    string_series = series.astype("string")
    string_series = string_series.fillna(CATEGORICAL_MISSING_TOKEN)
    return string_series.astype(str)


def make_feature_frame(values, *, kind: str, prefix: str) -> pd.DataFrame:
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
    y = np.asarray(values)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    elif y.ndim > 1 and y.shape[0] == 1:
        y = y.squeeze(0)
    return pd.Series(y).values


def load_array(file_path: Path) -> np.ndarray:
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


def find_by_suffix(files: list[Path], suffix: str) -> Path | None:
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


def stable_feature_prefix(context: str, fallback: str) -> str:
    stem = Path(context or fallback).stem
    for suffix in ("-train", "-test", "-val", "-single"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or fallback


def load_split(
    num_path: Path | None,
    cat_path: Path | None,
    y_path: Path,
    *,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    features: list[pd.DataFrame] = []
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


def categorical_indices_from_split(train_split) -> list[int]:
    num_path, cat_path, _ = train_split
    n_num = 0
    if num_path is not None:
        arr = load_array(Path(num_path))
        n_num = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    if cat_path is None:
        return []
    arr = load_array(Path(cat_path))
    n_cat = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    return list(range(n_num, n_num + n_cat))


def load_classification_dataset(dataset_dir: Path) -> LoadedDataset:
    info = load_dataset_info(dataset_dir)
    task_type = str(info.get("task_type", "")).lower() if info else None
    if task_type not in CLASSIFICATION_TASKS:
        raise SkipDataset(f"Skipped due to task_type={task_type!r}")

    train_split, val_split, test_split = find_split_files(dataset_dir)
    categorical_feature_indices = categorical_indices_from_split(train_split)
    X_train, y_train = load_split(
        train_split[0],
        train_split[1],
        train_split[2],
        context=f"{dataset_dir.name}-train",
    )

    X_val = None
    y_val = None
    if val_split is not None:
        X_val, y_val = load_split(
            val_split[0],
            val_split[1],
            val_split[2],
            context=f"{dataset_dir.name}-val",
        )

    X_test, y_test = load_split(
        test_split[0],
        test_split[1],
        test_split[2],
        context=f"{dataset_dir.name}-test",
    )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}"
        )
    if X_val is not None and X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Feature count mismatch: train has {X_train.shape[1]}, val has {X_val.shape[1]}"
        )

    return LoadedDataset(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        task_type=task_type,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        categorical_feature_indices=categorical_feature_indices,
    )


def empty_row_for_dataset(
    dataset_dir: Path,
    status: str,
    error: str,
    *,
    task_type: Optional[str] = None,
) -> ResultRow:
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
        f1=None,
        balanced_accuracy=None,
        roc_auc=None,
        log_loss=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
    )


def worker_crash_row(dataset_name: str, error: str) -> ResultRow:
    return ResultRow(
        dataset_name=dataset_name,
        dataset_dir="__worker__",
        task_type=None,
        n_train=0,
        n_val=0,
        n_test=0,
        n_features=0,
        n_classes=0,
        accuracy=None,
        f1=None,
        balanced_accuracy=None,
        roc_auc=None,
        log_loss=None,
        fit_seconds=0.0,
        predict_seconds=0.0,
        status="fail",
        error=error,
    )


def evaluate_one_dataset(adapter: LimiXAdapter, dataset_dir: Path) -> ResultRow:
    task_type: Optional[str] = None
    try:
        loaded = load_classification_dataset(dataset_dir)
        task_type = loaded.task_type
        try:
            result = adapter.fit_predict(loaded)
        except Exception as exc:
            if is_invalid_cuda_config_error(exc):
                result = adapter.fit_predict_with_flash_attention_disabled(loaded)
            else:
                raise
        y_pred = np.asarray(result.y_pred)
        y_test = np.asarray(loaded.y_test)
        if len(y_pred) != len(y_test):
            raise ValueError(f"Prediction length mismatch: got {len(y_pred)}, expected {len(y_test)}")
        accuracy = float(np.mean(y_pred == y_test))
        f1 = compute_weighted_f1(y_test, y_pred)
        balanced_accuracy = compute_balanced_accuracy(y_test, y_pred)
        roc_auc = compute_roc_auc(y_test, result.y_proba, result.classes)
        log_loss_score = compute_log_loss(y_test, result.y_proba, result.classes)
        return ResultRow(
            dataset_name=loaded.dataset_name,
            dataset_dir=loaded.dataset_dir.as_posix(),
            task_type=loaded.task_type,
            n_train=loaded.n_train_report,
            n_val=loaded.n_val,
            n_test=int(len(y_test)),
            n_features=int(loaded.X_train.shape[1]),
            n_classes=loaded.n_classes,
            accuracy=accuracy,
            f1=f1,
            balanced_accuracy=balanced_accuracy,
            roc_auc=roc_auc,
            log_loss=log_loss_score,
            fit_seconds=float(result.fit_seconds),
            predict_seconds=float(result.predict_seconds),
            status="ok",
            error=None,
        )
    except SkipDataset as exc:
        return empty_row_for_dataset(dataset_dir, "skip", str(exc), task_type=task_type)
    except Exception as exc:
        return empty_row_for_dataset(
            dataset_dir,
            "fail",
            f"{type(exc).__name__}: {exc}",
            task_type=task_type,
        )


def _evaluate_one_dataset_child(
    dataset_dir: str,
    gpu_id: int,
    args_dict: dict[str, Any],
    disable_flash_attention: bool,
    result_queue,
) -> None:
    bind_worker_gpu(gpu_id)
    args = argparse.Namespace(**args_dict)
    try:
        if disable_flash_attention:
            LimiXAdapter._disable_flash_attention_runtime()
        adapter = LimiXAdapter(args, device="cuda:0")
        row = evaluate_one_dataset(adapter, Path(dataset_dir))
    except Exception:
        row = worker_crash_row(Path(dataset_dir).name, traceback.format_exc())
    result_queue.put(asdict(row))


def evaluate_one_dataset_in_fresh_process(
    dataset_dir: Path,
    gpu_id: int,
    args_dict: dict[str, Any],
    *,
    disable_flash_attention: bool,
) -> ResultRow:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_evaluate_one_dataset_child,
        args=(str(dataset_dir), gpu_id, args_dict, disable_flash_attention, result_queue),
        daemon=False,
    )
    proc.start()
    proc.join()

    row_dict: dict[str, Any] | None = None
    if not result_queue.empty():
        row_dict = result_queue.get()
    result_queue.close()
    result_queue.join_thread()

    if row_dict is None:
        return worker_crash_row(
            dataset_dir.name,
            f"Child dataset retry exited without a result (exitcode={proc.exitcode})",
        )
    return ResultRow(**row_dict)


def rows_to_frame(rows: Iterable[ResultRow]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[RESULT_COLUMNS]


def ensure_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    normalized = frame.copy()
    for column in RESULT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[RESULT_COLUMNS]


def is_oom_error_message(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in OOM_ERROR_MARKERS)


def is_invalid_cuda_config_error(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in INVALID_CUDA_CONFIG_MARKERS)


def extract_dataset_names_from_frame(frame: pd.DataFrame) -> list[str]:
    dataset_names: list[str] = []
    seen_names: set[str] = set()
    for _, row in frame.iterrows():
        raw_name = str(row.get("dataset_name", "") or "").strip()
        if not raw_name and "dataset_dir" in frame.columns:
            raw_name = Path(str(row.get("dataset_dir", "") or "")).name.strip()
        if not raw_name or raw_name in seen_names:
            continue
        dataset_names.append(raw_name)
        seen_names.add(raw_name)
    return dataset_names


def load_failed_dataset_names_from_results_csv(
    results_csv: Path,
    *,
    include_oom_failures: bool,
) -> list[str]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Retry results CSV does not exist: {results_csv}")

    frame = pd.read_csv(results_csv)
    required_columns = {"dataset_name", "status"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Retry results CSV is missing required columns: {', '.join(missing_columns)}"
        )

    failed_df = frame[frame["status"].astype(str).str.lower() == "fail"].copy()
    if not include_oom_failures and "error" in failed_df.columns:
        failed_df = failed_df[~failed_df["error"].map(is_oom_error_message)]

    return extract_dataset_names_from_frame(failed_df)


def load_oom_failed_dataset_names_from_results_csv(
    results_csv: Path,
) -> tuple[list[str], list[str]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Retry results CSV does not exist: {results_csv}")

    frame = pd.read_csv(results_csv)
    required_columns = {"dataset_name", "status", "error"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Retry results CSV is missing required columns: {', '.join(missing_columns)}"
        )

    failed_df = frame[frame["status"].astype(str).str.lower() == "fail"].copy()
    oom_df = failed_df[failed_df["error"].map(is_oom_error_message)].copy()
    non_oom_df = failed_df[~failed_df["error"].map(is_oom_error_message)].copy()
    return extract_dataset_names_from_frame(oom_df), extract_dataset_names_from_frame(non_oom_df)


def filter_dataset_dirs_by_name(
    dataset_dirs: list[Path],
    dataset_names: Iterable[str],
) -> tuple[list[Path], list[str]]:
    wanted_names = {str(name).strip() for name in dataset_names if str(name).strip()}
    filtered = [path for path in dataset_dirs if path.name in wanted_names]
    found_names = {path.name for path in filtered}
    missing_names = [name for name in dataset_names if name not in found_names]
    return filtered, missing_names


def merge_result_frames(base_df: pd.DataFrame, updated_df: pd.DataFrame) -> pd.DataFrame:
    base_df = ensure_result_columns(base_df)
    updated_df = ensure_result_columns(updated_df)
    if base_df.empty:
        return updated_df
    if updated_df.empty:
        return base_df

    base_df = base_df.copy()
    updated_df = updated_df.copy()
    order_map = {
        str(dataset_name): idx
        for idx, dataset_name in enumerate(base_df["dataset_name"].astype(str).tolist())
    }
    replacement_names = set(updated_df["dataset_name"].astype(str).tolist())
    base_keep = base_df[~base_df["dataset_name"].astype(str).isin(replacement_names)].copy()

    base_keep["_merge_order"] = base_keep["dataset_name"].astype(str).map(order_map)
    appended_start = len(order_map)
    updated_df["_merge_order"] = [
        order_map.get(str(dataset_name), appended_start + idx)
        for idx, dataset_name in enumerate(updated_df["dataset_name"].tolist())
    ]

    merged = pd.concat([base_keep, updated_df], ignore_index=True)
    merged = merged.sort_values("_merge_order", kind="stable").drop(columns="_merge_order")
    merged = merged.reset_index(drop=True)
    return merged[RESULT_COLUMNS]


def write_summary(summary_path: Path, result_df: pd.DataFrame, dataset_dirs: list[Path], wall_seconds: float) -> None:
    result_df = result_df.copy()
    for metric_column in ("accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"):
        if metric_column in result_df.columns:
            result_df[metric_column] = pd.to_numeric(
                result_df[metric_column],
                errors="coerce",
            )
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()

    def mean_line(label: str, column: str) -> str:
        if len(ok_df) and column in ok_df.columns and ok_df[column].notna().any():
            return f"{label}: {ok_df[column].mean():.6f}"
        return f"{label}: (none)"

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        mean_line("avg_accuracy_ok", "accuracy"),
        mean_line("avg_f1_ok", "f1"),
        mean_line("avg_balanced_accuracy_ok", "balanced_accuracy"),
        mean_line("avg_roc_auc_ok", "roc_auc"),
        mean_line("avg_log_loss_ok", "log_loss"),
        f"wall_seconds: {wall_seconds:.3f}",
        "failed_datasets: "
        + (", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)"),
        "skipped_datasets: "
        + (", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)"),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_gpu_id_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def detect_env_gpu_ids() -> list[int]:
    for env_name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        raw_value = os.environ.get(env_name, "").strip()
        if not raw_value:
            continue
        try:
            gpu_ids = parse_gpu_id_list(raw_value)
        except ValueError:
            gpu_ids = []
        if gpu_ids:
            return gpu_ids
    return []


def detect_nvidia_gpu_ids() -> list[int]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    gpu_ids: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            gpu_ids.append(int(line.split(",")[0].strip()))
        except ValueError:
            continue
    return gpu_ids


def detect_torch_gpu_ids() -> list[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(int(torch.cuda.device_count())))
    except Exception:
        pass
    return []


def resolve_requested_gpu_ids(raw_gpus: Any) -> list[int]:
    if raw_gpus is None or str(raw_gpus).strip().lower() == "auto":
        gpu_ids = detect_env_gpu_ids() or detect_nvidia_gpu_ids() or detect_torch_gpu_ids()
        if not gpu_ids:
            raise RuntimeError("No visible GPU detected. Pass --gpus explicitly if needed.")
        return gpu_ids

    gpu_ids = parse_gpu_id_list(str(raw_gpus))
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id or use 'auto'")
    return gpu_ids


def resolve_workers_and_gpu_ids(args: argparse.Namespace) -> tuple[int, list[int]]:
    gpu_ids = resolve_requested_gpu_ids(args.gpus)
    workers = len(gpu_ids) if args.workers is None else int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if len(gpu_ids) != workers:
        raise ValueError(
            f"--gpus must contain exactly --workers ids; got {len(gpu_ids)} ids for {workers} workers"
        )
    return workers, gpu_ids


def bind_worker_gpu(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def bind_ddp_gpu_group(gpu_ids: list[int]) -> None:
    visible_devices = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ["ROCR_VISIBLE_DEVICES"] = visible_devices
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def worker_main(
    worker_id: int,
    gpu_id: int,
    dataset_dirs: list[str],
    worker_csv: str,
    args_dict: dict[str, Any],
) -> None:
    bind_worker_gpu(gpu_id)
    args = argparse.Namespace(**args_dict)
    rows: list[ResultRow] = []
    try:
        adapter = LimiXAdapter(args, device="cuda:0")
        for dataset_dir in dataset_dirs:
            row = evaluate_one_dataset(adapter, Path(dataset_dir))
            if row.status == "fail" and is_invalid_cuda_config_error(row.error):
                row = evaluate_one_dataset_in_fresh_process(
                    Path(dataset_dir),
                    gpu_id,
                    args_dict,
                    disable_flash_attention=True,
                )
                LimiXAdapter._disable_flash_attention_runtime()
                adapter = LimiXAdapter(args, device="cuda:0")
            rows.append(row)
            if args.verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [ok] "
                        f"{row.dataset_name} "
                        f"accuracy={format_optional_float(row.accuracy)} "
                        f"f1={format_optional_float(row.f1)} "
                        f"balanced_accuracy={format_optional_float(row.balanced_accuracy)} "
                        f"roc_auc={format_optional_float(row.roc_auc)} "
                        f"log_loss={format_optional_float(row.log_loss)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [{row.status}] "
                        f"{row.dataset_name} error={row.error}",
                        flush=True,
                    )
    except Exception:
        rows.append(worker_crash_row(f"__WORKER_CRASH__{worker_id}", traceback.format_exc()))
    rows_to_frame(rows).to_csv(worker_csv, index=False)


def format_subprocess_error(command: list[str], completed: subprocess.CompletedProcess[str]) -> str:
    details = [
        f"DDP subprocess failed with exit code {completed.returncode}",
        "command: " + " ".join(command),
    ]
    if completed.stdout:
        details.append("stdout:\n" + completed.stdout.strip())
    if completed.stderr:
        details.append("stderr:\n" + completed.stderr.strip())
    return "\n".join(details)


def evaluate_one_dataset_with_ddp(
    args: argparse.Namespace,
    dataset_dir: Path,
    gpu_ids: list[int],
    *,
    out_dir: Path,
) -> ResultRow:
    with tempfile.NamedTemporaryFile(
        prefix=f"{dataset_dir.name}_",
        suffix="_ddp_result.json",
        dir=out_dir,
        delete=False,
    ) as tmp_file:
        result_path = Path(tmp_file.name)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        str(Path(__file__).resolve()),
        "--internal-ddp-dataset-dir",
        str(dataset_dir.resolve()),
        "--internal-ddp-result-path",
        str(result_path),
        "--model-cache-dir",
        str(args.model_cache_dir),
        "--config-path",
        str(args.config_path),
        "--hf-repo",
        str(args.hf_repo),
        "--hf-filename",
        str(args.hf_filename),
        "--hf-endpoint",
        str(args.hf_endpoint),
        "--random-state",
        str(args.random_state),
        "--max-classes",
        str(args.max_classes),
        "--max-train-rows",
        str(args.max_train_rows),
        "--test-batch-rows",
        str(args.test_batch_rows),
    ]
    if args.direct_max_classes is not None:
        command.extend(["--direct-max-classes", str(args.direct_max_classes)])
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.disable_mixed_precision:
        command.append("--disable-mixed-precision")
    if args.verbose:
        command.append("--verbose")

    env = os.environ.copy()
    bind_ddp_gpu_group(gpu_ids)
    env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    env["ROCR_VISIBLE_DEVICES"] = os.environ["ROCR_VISIBLE_DEVICES"]
    env.pop("HIP_VISIBLE_DEVICES", None)
    env.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))

    try:
        completed = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if args.verbose and completed.stdout.strip():
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if args.verbose and completed.stderr.strip():
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)

        if completed.returncode == 0 and result_path.exists():
            row_dict = json.loads(result_path.read_text(encoding="utf-8"))
            return ResultRow(**row_dict)

        if completed.returncode == 0 and not result_path.exists():
            return empty_row_for_dataset(
                dataset_dir,
                "fail",
                "DDP subprocess finished successfully but did not produce a result row.",
            )

        return empty_row_for_dataset(
            dataset_dir,
            "fail",
            format_subprocess_error(command, completed),
        )
    finally:
        try:
            result_path.unlink(missing_ok=True)
        except Exception:
            pass


def initialize_ddp_child() -> tuple[int, int]:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return int(dist.get_rank()), local_rank


def cleanup_ddp_child() -> None:
    try:
        import torch.distributed as dist
    except Exception:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_internal_ddp_dataset(args: argparse.Namespace) -> None:
    if not args.internal_ddp_dataset_dir or not args.internal_ddp_result_path:
        raise ValueError("Internal DDP mode requires --internal-ddp-dataset-dir and --internal-ddp-result-path")

    rank, local_rank = initialize_ddp_child()
    try:
        adapter = LimiXAdapter(args, device=f"cuda:{local_rank}")
        row = evaluate_one_dataset(adapter, Path(args.internal_ddp_dataset_dir))
        if rank == 0:
            result_path = Path(args.internal_ddp_result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(asdict(row), ensure_ascii=False), encoding="utf-8")
    finally:
        cleanup_ddp_child()


def run_retry_failed_datasets_with_ddp(args: argparse.Namespace) -> None:
    data_root = resolve_script_path(args.data_root)
    out_dir = resolve_script_path(args.out_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.workers is not None and int(args.workers) != 1:
        raise ValueError("--retry-failed-datasets-with-ddp requires --workers 1")

    gpu_ids = resolve_requested_gpu_ids(args.gpus)
    if len(gpu_ids) < 2:
        raise ValueError("--retry-failed-datasets-with-ddp requires at least 2 GPU ids")

    reference_results_csv = resolve_script_path(args.reference_results_csv)
    merge_results_csv = (
        resolve_script_path(args.merge_results_from_csv)
        if args.merge_results_from_csv
        else reference_results_csv
    )

    all_dataset_dirs = find_dataset_dirs(data_root)
    oom_dataset_names, skipped_non_oom_names = load_oom_failed_dataset_names_from_results_csv(reference_results_csv)
    dataset_dirs, missing_dataset_names = filter_dataset_dirs_by_name(all_dataset_dirs, oom_dataset_names)

    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No OOM retry target datasets found under {data_root} for {reference_results_csv}"
        )

    if missing_dataset_names:
        print(
            "warning: retry CSV referenced OOM datasets missing under data_root: "
            + ", ".join(missing_dataset_names),
            file=sys.stderr,
            flush=True,
        )
    if skipped_non_oom_names:
        print(
            "warning: retry-failed-datasets-with-ddp skipped non-OOM failed datasets from reference CSV: "
            + ", ".join(skipped_non_oom_names),
            file=sys.stderr,
            flush=True,
        )

    if args.verbose:
        print(
            f"retry_failed_datasets_with_ddp: selected {len(dataset_dirs)} OOM datasets "
            f"from {reference_results_csv}",
            flush=True,
        )
        print(f"resolved_gpus: {','.join(str(gpu_id) for gpu_id in gpu_ids)}", flush=True)
        print(f"merge_results_from_csv: {merge_results_csv}", flush=True)

    started = time.time()
    rows: list[ResultRow] = []
    worker_csv = out_dir / "worker_0.csv"
    for dataset_dir in dataset_dirs:
        row = evaluate_one_dataset_with_ddp(args, dataset_dir, gpu_ids, out_dir=out_dir)
        rows.append(row)
        if args.verbose:
            if row.status == "ok":
                print(
                    f"[ddp oom retry | gpus {','.join(str(gpu_id) for gpu_id in gpu_ids)}] "
                    f"[ok] {row.dataset_name} "
                    f"accuracy={format_optional_float(row.accuracy)} "
                    f"f1={format_optional_float(row.f1)} "
                    f"balanced_accuracy={format_optional_float(row.balanced_accuracy)} "
                    f"roc_auc={format_optional_float(row.roc_auc)} "
                    f"log_loss={format_optional_float(row.log_loss)}",
                    flush=True,
                )
            else:
                print(
                    f"[ddp oom retry | gpus {','.join(str(gpu_id) for gpu_id in gpu_ids)}] "
                    f"[{row.status}] {row.dataset_name} error={row.error}",
                    flush=True,
                )
        rows_to_frame(rows).to_csv(worker_csv, index=False)

    result_df = rows_to_frame(rows)
    base_df = ensure_result_columns(pd.read_csv(merge_results_csv))
    result_df = merge_result_frames(base_df, result_df)

    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, result_df, all_dataset_dirs, time.time() - started)
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")


def run_benchmark(args: argparse.Namespace) -> None:
    data_root = resolve_script_path(args.data_root)
    out_dir = resolve_script_path(args.out_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_dirs = find_dataset_dirs(data_root)
    dataset_dirs = list(all_dataset_dirs)
    retry_results_csv: Path | None = None
    merge_results_csv: Path | None = None

    if args.retry_failed_datasets_only:
        retry_results_csv = resolve_script_path(args.reference_results_csv)
        failed_dataset_names = load_failed_dataset_names_from_results_csv(
            retry_results_csv,
            include_oom_failures=args.retry_include_oom_failures,
        )
        dataset_dirs, missing_dataset_names = filter_dataset_dirs_by_name(
            dataset_dirs,
            failed_dataset_names,
        )
        if missing_dataset_names:
            print(
                "warning: retry CSV referenced datasets missing under data_root: "
                + ", ".join(missing_dataset_names),
                file=sys.stderr,
                flush=True,
            )
        if args.verbose:
            print(
                f"retry_failed_datasets_only: selected {len(dataset_dirs)} datasets "
                f"from {retry_results_csv}",
                flush=True,
            )

    if args.merge_results_from_csv:
        merge_results_csv = resolve_script_path(args.merge_results_from_csv)
    elif retry_results_csv is not None:
        merge_results_csv = retry_results_csv

    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        if retry_results_csv is not None:
            raise FileNotFoundError(
                f"No retry target datasets found under {data_root} for {retry_results_csv}"
            )
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    args.data_root = str(data_root)
    args.out_dir = str(out_dir)
    args.workers, gpu_ids = resolve_workers_and_gpu_ids(args)
    if args.verbose:
        print(f"resolved_workers: {args.workers}", flush=True)
        print(f"resolved_gpus: {','.join(str(gpu_id) for gpu_id in gpu_ids)}", flush=True)
        if merge_results_csv is not None:
            print(f"merge_results_from_csv: {merge_results_csv}", flush=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    started = time.time()
    worker_csvs: list[Path] = []
    processes: list[mp.Process] = []
    args_dict = vars(args).copy()

    for worker_id in range(args.workers):
        assigned = [str(path.resolve()) for path in dataset_dirs[worker_id :: args.workers]]
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csvs.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, gpu_ids[worker_id], assigned, str(worker_csv), args_dict),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    frames = [pd.read_csv(path) for path in worker_csvs if path.exists()]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    result_df = ensure_result_columns(result_df)
    summary_dataset_dirs = dataset_dirs
    if merge_results_csv is not None:
        base_df = ensure_result_columns(pd.read_csv(merge_results_csv))
        result_df = merge_result_frames(base_df, result_df)
        summary_dataset_dirs = all_dataset_dirs
    all_csv = out_dir / "all_classification_results.csv"
    summary_txt = out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)
    write_summary(summary_txt, result_df, summary_dataset_dirs, time.time() - started)
    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LimiX on data178 classification datasets."
    )
    parser.add_argument("--data-root", default="../../data178")
    parser.add_argument("--out-dir", default="limix-16m_results_178")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--gpus",
        default="2,3",
        help="Comma-separated physical GPU ids, or 'auto' to use detected GPUs.",
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config-path", default="config/cls_default_noretrieval.json")
    parser.add_argument("--hf-repo", default="stableai-org/LimiX-16M")
    parser.add_argument("--hf-filename", default="LimiX-16M.ckpt")
    parser.add_argument("--model-cache-dir", default="cache")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--direct-max-classes", type=int, default=None)
    parser.add_argument("--test-batch-rows", type=int, default=0)
    parser.add_argument("--disable-mixed-precision", action="store_true")
    parser.add_argument(
        "--retry-failed-datasets-with-ddp",
        action="store_true",
        help=(
            "When enabled, only rerun OOM failures from --reference-results-csv by using "
            "README-style multi-GPU DDP inference. Requires --workers 1 and at least 2 GPUs."
        ),
    )
    parser.add_argument(
        "--retry-failed-datasets-only",
        action="store_true",
        help=(
            "When enabled, only rerun datasets whose rows have status=fail in "
            "--reference-results-csv. Disabled by default so the script can run "
            "full benchmarks normally."
        ),
    )
    parser.add_argument(
        "--reference-results-csv",
        default="limix_results/all_classification_results.csv",
        help=(
            "Historical results CSV used by the retry-failed modes to pick "
            "failed datasets. Default points to the local limix results file."
        ),
    )
    parser.add_argument(
        "--retry-include-oom-failures",
        action="store_true",
        help="When retrying previous failures, also rerun failures whose error looks like OOM.",
    )
    parser.add_argument(
        "--merge-results-from-csv",
        default=None,
        help=(
            "Merge the current run into this existing results CSV before saving "
            "all_classification_results.csv. Defaults to --reference-results-csv "
            "when a retry-failed mode is enabled."
        ),
    )
    parser.add_argument("--internal-ddp-dataset-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-ddp-result-path", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.internal_ddp_dataset_dir or args.internal_ddp_result_path:
        run_internal_ddp_dataset(args)
        return
    if args.retry_failed_datasets_with_ddp and args.retry_failed_datasets_only:
        raise ValueError(
            "--retry-failed-datasets-with-ddp cannot be combined with "
            "--retry-failed-datasets-only"
        )
    if args.retry_failed_datasets_with_ddp:
        run_retry_failed_datasets_with_ddp(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()

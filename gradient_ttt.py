#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_ROOT = Path("data178")
DEFAULT_MODEL_PATH = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
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
    n_train_a: int = 0
    n_train_b: int = 0
    n_holdout_c: int = 0
    n_test_d: int = 0
    ttt_loss: Optional[float] = None
    ttt_steps: int = 0
    ttt_lr: Optional[float] = None
    ttt_applied: bool = False
    ttt_update_seconds: float = 0.0
    ttt_split_strategy: Optional[str] = None
    ttt_split_reason: Optional[str] = None
    grad_aug_eligible: bool = False
    grad_aug_applied: bool = False
    grad_aug_anchor_count: int = 0
    grad_aug_candidate_count: int = 0
    grad_aug_kept_count: int = 0
    grad_aug_support_added: int = 0
    grad_aug_mean_cosine: Optional[float] = None
    grad_aug_min_cosine: Optional[float] = None
    grad_aug_reason: Optional[str] = None


@dataclass
class TTTConfig:
    enabled: bool = False
    lr: float = 2e-6
    scheduler: str = "constant"
    grad_clip: float = 1.0
    dtype: str = "float32"
    micro_batch_size: int = 1
    weight_decay: float = 0.0
    steps: int = 1
    freeze_col: bool = True
    freeze_row: bool = True
    train_fraction: float = 0.75
    random_state: int = 42
    data_parallel: bool = False
    gpu_group: Optional[str] = None
    save_ckpt: bool = True
    save_ckpt_every: int = 2
    save_ckpt_start_step: Optional[int] = None
    ckpt_root: Optional[str] = None
    grad_aug: bool = True
    grad_aug_max_train_size: int = 2048
    grad_aug_max_anchor_rows: int = 64
    grad_aug_candidates_per_anchor: int = 4
    grad_aug_keep_per_anchor: int = 1
    grad_aug_max_kept: int = 64
    grad_aug_min_cosine: float = 0.80
    grad_aug_loss_weight: float = 0.5
    grad_aug_signature_views: int = 4


@dataclass
class GradAugDiagnostics:
    eligible: bool = False
    applied: bool = False
    anchor_count: int = 0
    candidate_count: int = 0
    kept_count: int = 0
    support_added: int = 0
    mean_cosine: Optional[float] = None
    min_cosine: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class TTTSplit:
    b_indices: Any
    c_indices: Any
    strategy: str
    reason: str


@dataclass
class TTTUpdateResult:
    applied: bool
    loss: Optional[float]
    steps: int
    update_seconds: float
    reason: Optional[str] = None
    support_aug_X: Any = None
    support_aug_y: Any = None
    grad_aug: GradAugDiagnostics | None = None


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


def format_optional_float(value: Optional[float], precision: int = 6) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{precision}f}"


def format_dataset_result_log(
    worker_label: str,
    row: ResultRow,
    *,
    model_name: str | None = None,
) -> str:
    prefix = f"[{worker_label}]"
    if model_name:
        prefix = f"{prefix} [{model_name}]"

    if row.status == "ok":
        return (
            f"{prefix} [ok] {row.dataset_name} "
            f"accuracy={format_optional_float(row.accuracy)} "
            f"fit={row.fit_seconds:.3f}s "
            f"predict={row.predict_seconds:.3f}s "
            f"ttt_applied={row.ttt_applied} "
            f"ttt_update={row.ttt_update_seconds:.3f}s "
            f"ttt_loss={format_optional_float(row.ttt_loss)} "
            f"ttt_steps={row.ttt_steps} "
            f"grad_aug={row.grad_aug_applied} "
            f"grad_aug_kept={row.grad_aug_kept_count}"
        )
    if row.status == "skip":
        return f"{prefix} [skip] {row.dataset_name} reason={row.error}"
    return f"{prefix} [fail] {row.dataset_name} error={row.error}"


def parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered == "none":
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


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be one of: true, false")


def parse_kv_cache(value: str) -> bool | str:
    lowered = value.strip().lower()
    if lowered in {"false", "0", "no", "off"}:
        return False
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"kv", "repr"}:
        return lowered
    raise argparse.ArgumentTypeError("kv_cache must be one of: false, true, kv, repr")


def normalize_gpu_group(value: int | str) -> str:
    if isinstance(value, int):
        return str(value)
    gpu_group = str(value).strip()
    gpu_ids = parse_gpu_id_list(gpu_group)
    if not gpu_ids:
        raise ValueError(f"GPU group must contain at least one GPU id: {value!r}")
    return ",".join(str(gpu_id) for gpu_id in gpu_ids)


def first_gpu_id_from_group(value: int | str) -> int:
    return parse_gpu_id_list(normalize_gpu_group(value))[0]


def apply_worker_environment_updates(gpu_id: int | str) -> str:
    gpu_id_str = normalize_gpu_group(gpu_id)
    # Set visibility vars for both CUDA and ROCm stacks so each worker sees one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
    os.environ["HIP_VISIBLE_DEVICES"] = gpu_id_str
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return "cuda:0"


def parse_gpu_id_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_gpu_group_list(value: str) -> List[str]:
    gpu_groups = []
    for raw_group in value.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        gpu_groups.append(normalize_gpu_group(raw_group))
    return gpu_groups


def detect_default_gpu_ids() -> List[int]:
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

    try:
        import torch

        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 0

    if device_count <= 0:
        raise RuntimeError("No visible GPU detected; please pass --gpus explicitly")
    return list(range(device_count))


def load_dataset_info(dataset_dir: Path) -> dict | None:
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def should_skip_ttt_for_dataset(dataset_dir: Path, info: dict | None) -> bool:
    dataset_names = {dataset_dir.name.strip().lower()}
    if info:
        info_name = str(info.get("name", "")).strip().lower()
        if info_name:
            dataset_names.add(info_name)
    return "volkert" in dataset_names


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


def build_ttt_config(args: argparse.Namespace) -> TTTConfig:
    if int(args.ttt_save_ckpt_every) < 1:
        raise ValueError("--ttt-save-ckpt-every must be >= 1")
    if args.ttt_save_ckpt_start_step is not None and int(args.ttt_save_ckpt_start_step) < 1:
        raise ValueError("--ttt-save-ckpt-start-step must be >= 1 or None")
    if int(args.grad_aug_signature_views) < 1:
        raise ValueError("--grad-aug-signature-views must be >= 1")
    if float(args.grad_aug_loss_weight) < 0:
        raise ValueError("--grad-aug-loss-weight must be >= 0")

    return TTTConfig(
        enabled=bool(args.ttt_holdout),
        lr=float(args.ttt_lr),
        scheduler=str(args.ttt_scheduler),
        grad_clip=float(args.ttt_grad_clip),
        dtype=str(args.ttt_dtype),
        micro_batch_size=int(args.ttt_micro_batch_size),
        weight_decay=float(args.ttt_weight_decay),
        steps=int(args.ttt_steps),
        freeze_col=bool(args.ttt_freeze_col),
        freeze_row=bool(args.ttt_freeze_row),
        random_state=int(args.random_state),
        data_parallel=bool(args.ttt_data_parallel),
        save_ckpt=bool(args.ttt_save_ckpt),
        save_ckpt_every=int(args.ttt_save_ckpt_every),
        save_ckpt_start_step=(
            int(args.ttt_save_ckpt_start_step) if args.ttt_save_ckpt_start_step is not None else None
        ),
        ckpt_root=str((Path(args.out_dir).expanduser() / "ttt_ckpts").resolve()),
        grad_aug=bool(args.grad_aug),
        grad_aug_max_train_size=int(args.grad_aug_max_train_size),
        grad_aug_max_anchor_rows=int(args.grad_aug_max_anchor_rows),
        grad_aug_candidates_per_anchor=int(args.grad_aug_candidates_per_anchor),
        grad_aug_keep_per_anchor=int(args.grad_aug_keep_per_anchor),
        grad_aug_max_kept=int(args.grad_aug_max_kept),
        grad_aug_min_cosine=float(args.grad_aug_min_cosine),
        grad_aug_loss_weight=float(args.grad_aug_loss_weight),
        grad_aug_signature_views=int(args.grad_aug_signature_views),
    )


def take_rows(X, indices):
    if hasattr(X, "iloc"):
        return X.iloc[indices].reset_index(drop=True)
    return np.asarray(X)[indices]


def split_ttt_holdout(y, config: TTTConfig) -> TTTSplit:
    ensure_runtime_deps()

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    if len(indices) < 2:
        return TTTSplit(
            b_indices=indices,
            c_indices=np.asarray([], dtype=int),
            strategy="none",
            reason="Need at least two samples for hold-out split.",
        )

    y_array = np.asarray(y)
    try:
        b_idx, c_idx = train_test_split(
            indices,
            train_size=config.train_fraction,
            random_state=config.random_state,
            shuffle=True,
            stratify=y_array,
        )
        return TTTSplit(
            b_indices=np.asarray(b_idx, dtype=int),
            c_indices=np.asarray(c_idx, dtype=int),
            strategy="stratified",
            reason="stratified split",
        )
    except Exception as exc:
        b_idx, c_idx = train_test_split(
            indices,
            train_size=config.train_fraction,
            random_state=config.random_state,
            shuffle=True,
            stratify=None,
        )
        return TTTSplit(
            b_indices=np.asarray(b_idx, dtype=int),
            c_indices=np.asarray(c_idx, dtype=int),
            strategy="random",
            reason=f"stratified split unavailable: {type(exc).__name__}: {exc}",
        )


def _torch_dtype(dtype_name: str):
    import torch

    normalized = dtype_name.strip().lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError("--ttt-dtype must be one of: float32, float16, bfloat16")


def _set_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _get_ttt_base_model(classifier):
    model = classifier.model_
    return getattr(model, "module", model)


def _resolve_ttt_data_parallel_device_ids(classifier, config: TTTConfig) -> List[int]:
    if not config.data_parallel or not config.gpu_group:
        return []

    device = getattr(classifier, "device_", None)
    device_type = getattr(device, "type", str(device).split(":")[0])
    if device_type != "cuda":
        return []

    try:
        gpu_ids = parse_gpu_id_list(normalize_gpu_group(config.gpu_group))
    except Exception:
        return []
    if len(gpu_ids) < 2:
        return []

    import torch

    if not torch.cuda.is_available():
        return []

    # Device visibility has already been rewritten per worker, so DataParallel
    # must use process-local logical ids rather than original global ids.
    return list(range(len(gpu_ids)))


def _build_ttt_forward_model(classifier, config: TTTConfig):
    base_model = _get_ttt_base_model(classifier)
    device_ids = _resolve_ttt_data_parallel_device_ids(classifier, config)
    if not device_ids:
        return base_model, 1

    import torch

    try:
        return torch.nn.DataParallel(base_model, device_ids=device_ids), len(device_ids)
    except Exception:
        return base_model, 1


def _configure_ttt_trainable_params(classifier, config: TTTConfig):
    model = _get_ttt_base_model(classifier)
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    # Keep frozen blocks in train mode so TabICL uses the training forward path.
    # Their parameters remain frozen by requires_grad=False.
    model.col_embedder.train()
    if not config.freeze_col:
        _set_requires_grad(model.col_embedder, True)

    model.row_interactor.train()
    if not config.freeze_row:
        _set_requires_grad(model.row_interactor, True)

    model.icl_predictor.train()
    _set_requires_grad(model.icl_predictor, True)

    return [param for param in model.parameters() if param.requires_grad]


def _fit_preserving_model_weights(classifier, X, y) -> None:
    original_load_model = getattr(classifier, "_load_model", None)

    def _skip_model_reload():
        return None

    try:
        classifier._load_model = _skip_model_reload
        classifier.fit(X, y)
    finally:
        if original_load_model is not None:
            classifier._load_model = original_load_model


def _iter_classification_ensemble_batches(classifier, X_c_encoded, y_c_encoded):
    ensemble_data = classifier.ensemble_generator_.transform(X_c_encoded, mode="both")
    class_shuffles_by_norm = classifier.ensemble_generator_.class_shuffles_

    for norm_method, (Xs, ys) in ensemble_data.items():
        class_shuffles = class_shuffles_by_norm[norm_method]
        if len(class_shuffles) != Xs.shape[0]:
            raise RuntimeError(
                f"Ensemble data/class shuffle mismatch for norm_method={norm_method!r}: "
                f"{Xs.shape[0]} views vs {len(class_shuffles)} class shuffles"
            )

        y_targets = []
        for class_shuffle in class_shuffles:
            y_targets.append(np.asarray(class_shuffle, dtype=np.int64)[y_c_encoded])
        yield Xs, ys, np.stack(y_targets, axis=0)


def _feature_frame(X) -> Any:
    ensure_runtime_deps()

    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True).copy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return pd.DataFrame(arr)


def _restore_feature_container(reference, rows_df):
    ensure_runtime_deps()

    rows_df = rows_df.reset_index(drop=True)
    if isinstance(reference, pd.DataFrame):
        return rows_df.loc[:, list(reference.columns)]
    return rows_df.to_numpy()


def _append_feature_rows(X, X_extra):
    ensure_runtime_deps()

    if X_extra is None:
        return X
    if len(X_extra) == 0:
        return X
    if isinstance(X, pd.DataFrame):
        return pd.concat([X.reset_index(drop=True), X_extra.reset_index(drop=True)], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(X), np.asarray(X_extra)], axis=0)


def _is_integer_like_series(series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return False
    return bool(np.all(np.isclose(values, np.round(values), rtol=0.0, atol=1e-6)))


def _choose_same_label_neighbor(
    X_b_df,
    y_b_array,
    anchor_row,
    label,
    numeric_cols: List[str],
    rng,
) -> int | None:
    same_label_indices = np.asarray([idx for idx, value in enumerate(y_b_array) if value == label], dtype=int)
    if same_label_indices.size == 0:
        return None
    if not numeric_cols:
        return int(rng.choice(same_label_indices))

    pool = X_b_df.iloc[same_label_indices][numeric_cols].apply(pd.to_numeric, errors="coerce")
    anchor_values = pd.to_numeric(anchor_row[numeric_cols], errors="coerce").to_numpy(dtype=float)
    scale = X_b_df[numeric_cols].apply(pd.to_numeric, errors="coerce").std(axis=0).to_numpy(dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    diffs = (pool.to_numpy(dtype=float) - anchor_values.reshape(1, -1)) / scale.reshape(1, -1)
    distances = np.nanmean(np.square(diffs), axis=1)
    distances = np.where(np.isfinite(distances), distances, np.inf)
    ranked = same_label_indices[np.argsort(distances)]
    nearest_pool = ranked[: min(16, ranked.size)]
    if nearest_pool.size == 0:
        return int(rng.choice(same_label_indices))
    return int(rng.choice(nearest_pool))


def _generate_gradient_aug_candidates(
    X_b,
    y_b,
    X_c,
    y_c,
    config: TTTConfig,
):
    ensure_runtime_deps()

    diagnostics = GradAugDiagnostics()
    n_train_total = int(len(y_b) + len(y_c))
    if not config.grad_aug:
        diagnostics.reason = "--grad-aug disabled"
        return None, None, None, diagnostics
    if n_train_total > config.grad_aug_max_train_size:
        diagnostics.reason = (
            f"n_train={n_train_total} exceeds grad_aug_max_train_size={config.grad_aug_max_train_size}"
        )
        return None, None, None, diagnostics
    if config.grad_aug_max_anchor_rows < 1:
        diagnostics.reason = "--grad-aug-max-anchor-rows must be >= 1"
        return None, None, None, diagnostics
    if config.grad_aug_candidates_per_anchor < 1:
        diagnostics.reason = "--grad-aug-candidates-per-anchor must be >= 1"
        return None, None, None, diagnostics
    if config.grad_aug_keep_per_anchor < 1:
        diagnostics.reason = "--grad-aug-keep-per-anchor must be >= 1"
        return None, None, None, diagnostics
    if config.grad_aug_max_kept < 1:
        diagnostics.reason = "--grad-aug-max-kept must be >= 1"
        return None, None, None, diagnostics
    if len(y_c) == 0:
        diagnostics.reason = "No hold-out C samples available for gradient augmentation"
        return None, None, None, diagnostics

    diagnostics.eligible = True
    rng = np.random.default_rng(config.random_state)
    X_b_df = _feature_frame(X_b)
    X_c_df = _feature_frame(X_c)
    y_b_array = np.asarray(y_b, dtype=object)
    y_c_array = np.asarray(y_c, dtype=object)

    numeric_cols = [col for col in X_b_df.columns if pd.api.types.is_numeric_dtype(X_b_df[col])]
    categorical_cols = [col for col in X_b_df.columns if col not in numeric_cols]
    train_context_df = pd.concat([X_b_df, X_c_df], axis=0, ignore_index=True)
    numeric_bounds = {}
    integer_like = {}
    for col in numeric_cols:
        numeric_values = pd.to_numeric(train_context_df[col], errors="coerce")
        numeric_bounds[col] = (numeric_values.min(skipna=True), numeric_values.max(skipna=True))
        integer_like[col] = _is_integer_like_series(numeric_values)

    anchor_indices = np.arange(len(y_c_array), dtype=int)
    if anchor_indices.size > config.grad_aug_max_anchor_rows:
        anchor_indices = np.asarray(
            rng.choice(anchor_indices, size=config.grad_aug_max_anchor_rows, replace=False),
            dtype=int,
        )
        anchor_indices.sort()
    diagnostics.anchor_count = int(anchor_indices.size)

    candidate_rows = []
    candidate_labels = []
    candidate_anchor_indices = []

    for anchor_idx in anchor_indices:
        anchor_row = X_c_df.iloc[int(anchor_idx)].copy()
        anchor_label = y_c_array[int(anchor_idx)]
        for _ in range(config.grad_aug_candidates_per_anchor):
            neighbor_idx = _choose_same_label_neighbor(
                X_b_df,
                y_b_array,
                anchor_row,
                anchor_label,
                numeric_cols,
                rng,
            )
            if neighbor_idx is None:
                continue
            neighbor_row = X_b_df.iloc[neighbor_idx]
            mixed_row = anchor_row.copy()
            alpha = float(rng.uniform(0.25, 0.75))

            for col in numeric_cols:
                anchor_value = pd.to_numeric(pd.Series([anchor_row[col]]), errors="coerce").iloc[0]
                neighbor_value = pd.to_numeric(pd.Series([neighbor_row[col]]), errors="coerce").iloc[0]
                if pd.isna(anchor_value) and pd.isna(neighbor_value):
                    value = anchor_value
                elif pd.isna(anchor_value):
                    value = neighbor_value
                elif pd.isna(neighbor_value):
                    value = anchor_value
                else:
                    value = alpha * float(anchor_value) + (1.0 - alpha) * float(neighbor_value)

                lower, upper = numeric_bounds[col]
                if pd.notna(value):
                    if pd.notna(lower):
                        value = max(float(lower), float(value))
                    if pd.notna(upper):
                        value = min(float(upper), float(value))
                    if integer_like[col]:
                        value = round(float(value))
                mixed_row[col] = value

            for col in categorical_cols:
                mixed_row[col] = anchor_row[col] if rng.random() < 0.5 else neighbor_row[col]

            candidate_rows.append(mixed_row)
            candidate_labels.append(anchor_label)
            candidate_anchor_indices.append(int(anchor_idx))

    diagnostics.candidate_count = int(len(candidate_rows))
    if not candidate_rows:
        diagnostics.reason = "No same-label B neighbors available for hold-out C anchors"
        return None, None, None, diagnostics

    candidate_df = pd.DataFrame(candidate_rows, columns=X_b_df.columns)
    candidate_X = _restore_feature_container(X_b, candidate_df)
    candidate_y = np.asarray(candidate_labels, dtype=np.asarray(y_c).dtype)
    candidate_anchor_indices = np.asarray(candidate_anchor_indices, dtype=int)
    diagnostics.reason = "candidates generated"
    return candidate_X, candidate_y, candidate_anchor_indices, diagnostics


def _select_gradient_signature_params(base_model) -> List[Any]:
    signature_params: List[Any] = []
    icl_predictor = getattr(base_model, "icl_predictor", None)
    if icl_predictor is None:
        return signature_params

    for module_name in ("decoder", "ln"):
        module = getattr(icl_predictor, module_name, None)
        if module is None:
            continue
        signature_params.extend(param for param in module.parameters() if param.requires_grad)
    return signature_params


def _clear_param_grads(params: List[Any]) -> None:
    for param in params:
        param.grad = None


def _compute_gradient_signature(
    classifier,
    X_sample,
    y_sample,
    config: TTTConfig,
    forward_context,
    trainable_params: List[Any],
    signature_params: List[Any],
):
    import torch
    import torch.nn.functional as F

    X_encoded = classifier.X_encoder_.transform(X_sample)
    y_encoded = classifier.y_encoder_.transform(y_sample)
    selected_batches = []
    remaining_views = max(1, int(config.grad_aug_signature_views))
    for Xs, ys, y_targets in _iter_classification_ensemble_batches(classifier, X_encoded, y_encoded):
        if remaining_views <= 0:
            break
        take_count = min(int(Xs.shape[0]), remaining_views)
        selected_batches.append((Xs[:take_count], ys[:take_count], y_targets[:take_count]))
        remaining_views -= take_count

    total_views = int(sum(Xs.shape[0] for Xs, _, _ in selected_batches))
    if total_views == 0:
        return None

    device = classifier.device_
    forward_model = _get_ttt_base_model(classifier)
    _clear_param_grads(trainable_params)

    try:
        for Xs, ys, y_targets in selected_batches:
            X_batch = torch.from_numpy(Xs).float().to(device)
            y_train_batch = torch.from_numpy(ys).float().to(device)
            y_target_batch = torch.from_numpy(y_targets).long().to(device)
            d_batch = torch.full(
                (X_batch.shape[0],),
                X_batch.shape[2],
                device=device,
                dtype=torch.long,
            )

            with forward_context():
                logits = forward_model(X_batch, y_train_batch, d_batch)
                loss = F.cross_entropy(logits.flatten(end_dim=-2), y_target_batch.flatten())
                scaled_loss = loss * (X_batch.shape[0] / total_views)
            scaled_loss.backward()

        grad_chunks = []
        for param in signature_params:
            if param.grad is not None:
                grad_chunks.append(param.grad.detach().flatten().float().cpu())
        if not grad_chunks:
            return None
        vector = torch.cat(grad_chunks)
        norm = torch.linalg.vector_norm(vector)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            return None
        return vector / norm
    finally:
        _clear_param_grads(trainable_params)


def _filter_gradient_matched_augments(
    classifier,
    X_b,
    y_b,
    X_c,
    y_c,
    config: TTTConfig,
    forward_context,
    trainable_params: List[Any],
):
    import torch

    candidate_X, candidate_y, candidate_anchor_indices, diagnostics = _generate_gradient_aug_candidates(
        X_b,
        y_b,
        X_c,
        y_c,
        config,
    )
    if candidate_X is None or candidate_y is None or candidate_anchor_indices is None:
        return None, None, diagnostics

    base_model = _get_ttt_base_model(classifier)
    signature_params = _select_gradient_signature_params(base_model)
    if not signature_params:
        diagnostics.reason = "No icl_predictor decoder/ln parameters available for gradient signatures"
        return None, None, diagnostics

    anchor_signature_cache = {}
    scored_candidates = []
    for candidate_idx, anchor_idx in enumerate(candidate_anchor_indices.tolist()):
        if anchor_idx not in anchor_signature_cache:
            X_anchor = take_rows(X_c, [anchor_idx])
            y_anchor = np.asarray(y_c, dtype=object)[[anchor_idx]]
            anchor_signature_cache[anchor_idx] = _compute_gradient_signature(
                classifier,
                X_anchor,
                y_anchor,
                config,
                forward_context,
                trainable_params,
                signature_params,
            )
        anchor_signature = anchor_signature_cache[anchor_idx]
        if anchor_signature is None:
            continue

        X_candidate = take_rows(candidate_X, [candidate_idx])
        y_candidate = np.asarray(candidate_y, dtype=object)[[candidate_idx]]
        candidate_signature = _compute_gradient_signature(
            classifier,
            X_candidate,
            y_candidate,
            config,
            forward_context,
            trainable_params,
            signature_params,
        )
        if candidate_signature is None:
            continue
        cosine = float(torch.dot(anchor_signature, candidate_signature).item())
        if np.isfinite(cosine) and cosine >= config.grad_aug_min_cosine:
            scored_candidates.append((anchor_idx, candidate_idx, cosine))

    if not scored_candidates:
        diagnostics.reason = f"No candidates passed cosine >= {config.grad_aug_min_cosine:.4f}"
        return None, None, diagnostics

    kept_by_anchor = []
    for anchor_idx in sorted({item[0] for item in scored_candidates}):
        anchor_items = [item for item in scored_candidates if item[0] == anchor_idx]
        anchor_items.sort(key=lambda item: item[2], reverse=True)
        kept_by_anchor.extend(anchor_items[: config.grad_aug_keep_per_anchor])

    kept_by_anchor.sort(key=lambda item: item[2], reverse=True)
    kept = kept_by_anchor[: config.grad_aug_max_kept]
    kept_indices = [item[1] for item in kept]
    kept_cosines = [item[2] for item in kept]
    kept_X = take_rows(candidate_X, kept_indices)
    kept_y = np.asarray(candidate_y, dtype=object)[kept_indices]

    diagnostics.applied = True
    diagnostics.kept_count = int(len(kept_indices))
    diagnostics.support_added = int(len(kept_indices))
    diagnostics.mean_cosine = float(np.mean(kept_cosines))
    diagnostics.min_cosine = float(np.min(kept_cosines))
    diagnostics.reason = (
        f"kept {diagnostics.kept_count}/{diagnostics.candidate_count} candidates "
        f"with cosine >= {config.grad_aug_min_cosine:.4f}"
    )
    return kept_X, kept_y, diagnostics


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"


def _build_ttt_ckpt_path(config: TTTConfig, model_name: str, dataset_name: str, step_idx: int) -> Path:
    if not config.ckpt_root:
        raise ValueError("TTT checkpoint root is not configured")
    return (
        Path(config.ckpt_root)
        / _sanitize_path_component(model_name)
        / _sanitize_path_component(dataset_name)
        / f"step_{step_idx}.ckpt"
    )


def _save_ttt_model_ckpt(classifier, config: TTTConfig, model_name: str, dataset_name: str, step_idx: int) -> Path:
    import torch

    model_config = getattr(classifier, "model_config_", None)
    if model_config is None:
        raise RuntimeError("TTT checkpoint save requested before model_config_ is available")

    base_model = _get_ttt_base_model(classifier)
    ckpt_path = _build_ttt_ckpt_path(config, model_name, dataset_name, step_idx)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "config": dict(model_config),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items()},
        "ttt_metadata": {
            "model_name": str(model_name),
            "dataset_name": str(dataset_name),
            "step": int(step_idx),
        },
    }
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def _should_save_ttt_ckpt_step(config: TTTConfig, step_idx: int) -> bool:
    if not config.save_ckpt:
        return False
    if config.save_ckpt_start_step is None:
        return step_idx % config.save_ckpt_every == 0
    return (
        step_idx >= config.save_ckpt_start_step
        and (step_idx - config.save_ckpt_start_step) % config.save_ckpt_every == 0
    )


def _should_save_ttt_final_ckpt(config: TTTConfig) -> bool:
    if not config.save_ckpt:
        return False
    if config.save_ckpt_start_step is None:
        return True
    return config.steps >= config.save_ckpt_start_step


def _derive_model_name(model_path: str | None, checkpoint_version: str | None) -> str:
    candidate = model_path if model_path else checkpoint_version
    if not candidate:
        return "tabicl_model"
    return Path(str(candidate)).stem


def run_ttt_holdout_update(
    classifier,
    X_b,
    y_b,
    X_c,
    y_c,
    config: TTTConfig,
    *,
    model_name: str,
    dataset_name: str,
) -> TTTUpdateResult:
    ensure_runtime_deps()

    if config.scheduler != "constant":
        raise ValueError("--ttt-scheduler currently supports only 'constant'")
    if config.steps < 1:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=0.0,
            reason="--ttt-steps must be >= 1",
        )
    if config.micro_batch_size < 1:
        raise ValueError("--ttt-micro-batch-size must be >= 1")

    y_b_values = set(pd.Series(np.asarray(y_b)).astype(object).tolist())
    y_c_values = set(pd.Series(np.asarray(y_c)).astype(object).tolist())
    missing_from_context = sorted(str(value) for value in (y_c_values - y_b_values))
    if missing_from_context:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=0.0,
            reason="Hold-out labels absent from B context: " + ",".join(missing_from_context),
        )

    update_start = time.time()
    classifier.fit(X_b, y_b)
    if classifier.n_classes_ > classifier.model_.max_classes:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            reason=(
                f"TTT training skipped because n_classes={classifier.n_classes_} "
                f"exceeds model max_classes={classifier.model_.max_classes}"
            ),
        )

    import torch
    import torch.nn.functional as F

    base_model = _get_ttt_base_model(classifier)
    torch_dtype = _torch_dtype(config.dtype)
    trainable_params = _configure_ttt_trainable_params(classifier, config)
    if not trainable_params:
        return TTTUpdateResult(
            applied=False,
            loss=None,
            steps=0,
            update_seconds=time.time() - update_start,
            reason="No trainable parameters selected for TTT",
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    forward_model, data_parallel_world_size = _build_ttt_forward_model(classifier, config)
    effective_micro_batch_size = config.micro_batch_size * data_parallel_world_size

    X_c_encoded = classifier.X_encoder_.transform(X_c)
    y_c_encoded = classifier.y_encoder_.transform(y_c)

    device = classifier.device_
    device_type = getattr(device, "type", str(device).split(":")[0])
    use_autocast = torch_dtype != torch.float32 and device_type in {"cuda", "cpu"}

    from contextlib import nullcontext

    def forward_context():
        if use_autocast:
            return torch.autocast(device_type=device_type, dtype=torch_dtype)
        return nullcontext()

    last_loss = None
    support_aug_X = None
    support_aug_y = None
    grad_aug_diagnostics = GradAugDiagnostics(reason="gradient augmentation not attempted")
    try:
        if X_c_encoded.shape[0] == 0:
            return TTTUpdateResult(
                applied=False,
                loss=None,
                steps=0,
                update_seconds=time.time() - update_start,
                reason="No hold-out samples available for TTT",
                grad_aug=GradAugDiagnostics(reason="No hold-out samples available for TTT"),
            )

        ensemble_batches = list(_iter_classification_ensemble_batches(classifier, X_c_encoded, y_c_encoded))
        total_views = int(sum(Xs.shape[0] for Xs, _, _ in ensemble_batches))
        if total_views == 0:
            return TTTUpdateResult(
                applied=False,
                loss=None,
                steps=0,
                update_seconds=time.time() - update_start,
                reason="No ensemble views available for TTT",
                grad_aug=GradAugDiagnostics(reason="No ensemble views available for TTT"),
            )

        aug_ensemble_batches = []
        total_aug_views = 0
        try:
            support_aug_X, support_aug_y, grad_aug_diagnostics = _filter_gradient_matched_augments(
                classifier,
                X_b,
                y_b,
                X_c,
                y_c,
                config,
                forward_context,
                trainable_params,
            )
            if support_aug_X is not None and support_aug_y is not None and len(support_aug_y):
                X_aug_encoded = classifier.X_encoder_.transform(support_aug_X)
                y_aug_encoded = classifier.y_encoder_.transform(support_aug_y)
                aug_ensemble_batches = list(_iter_classification_ensemble_batches(classifier, X_aug_encoded, y_aug_encoded))
                total_aug_views = int(sum(Xs.shape[0] for Xs, _, _ in aug_ensemble_batches))
                if total_aug_views == 0:
                    support_aug_X = None
                    support_aug_y = None
                    grad_aug_diagnostics.applied = False
                    grad_aug_diagnostics.kept_count = 0
                    grad_aug_diagnostics.support_added = 0
                    grad_aug_diagnostics.reason = "No ensemble views available for kept augmented samples"
                else:
                    print(
                        f"[grad-aug] model={model_name} dataset={dataset_name} "
                        f"kept={grad_aug_diagnostics.kept_count} "
                        f"mean_cosine={format_optional_float(grad_aug_diagnostics.mean_cosine)}",
                        flush=True,
                    )
        except Exception as exc:
            grad_aug_diagnostics = GradAugDiagnostics(
                eligible=bool(config.grad_aug and (len(y_b) + len(y_c)) <= config.grad_aug_max_train_size),
                reason=f"gradient augmentation failed: {type(exc).__name__}: {exc}",
            )
            support_aug_X = None
            support_aug_y = None
            aug_ensemble_batches = []
            total_aug_views = 0
            _clear_param_grads(trainable_params)

        last_saved_step = 0
        for step_idx in range(1, config.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for Xs, ys, y_targets in ensemble_batches:
                for start_idx in range(0, Xs.shape[0], effective_micro_batch_size):
                    end_idx = min(start_idx + effective_micro_batch_size, Xs.shape[0])
                    X_batch_np = Xs[start_idx:end_idx]
                    y_train_np = ys[start_idx:end_idx]
                    y_target_np = y_targets[start_idx:end_idx]

                    X_batch = torch.from_numpy(X_batch_np).float().to(device)
                    y_train_batch = torch.from_numpy(y_train_np).float().to(device)
                    y_target_batch = torch.from_numpy(y_target_np).long().to(device)
                    d_batch = torch.full(
                        (X_batch.shape[0],),
                        X_batch.shape[2],
                        device=device,
                        dtype=torch.long,
                    )

                    with forward_context():
                        logits = forward_model(X_batch, y_train_batch, d_batch)
                        loss = F.cross_entropy(logits.flatten(end_dim=-2), y_target_batch.flatten())
                        scaled_loss = loss * (X_batch.shape[0] / total_views)

                    scaled_loss.backward()
                    step_loss += float(scaled_loss.detach().cpu())

            if aug_ensemble_batches and total_aug_views > 0 and config.grad_aug_loss_weight > 0:
                for Xs, ys, y_targets in aug_ensemble_batches:
                    for start_idx in range(0, Xs.shape[0], effective_micro_batch_size):
                        end_idx = min(start_idx + effective_micro_batch_size, Xs.shape[0])
                        X_batch_np = Xs[start_idx:end_idx]
                        y_train_np = ys[start_idx:end_idx]
                        y_target_np = y_targets[start_idx:end_idx]

                        X_batch = torch.from_numpy(X_batch_np).float().to(device)
                        y_train_batch = torch.from_numpy(y_train_np).float().to(device)
                        y_target_batch = torch.from_numpy(y_target_np).long().to(device)
                        d_batch = torch.full(
                            (X_batch.shape[0],),
                            X_batch.shape[2],
                            device=device,
                            dtype=torch.long,
                        )

                        with forward_context():
                            logits = forward_model(X_batch, y_train_batch, d_batch)
                            loss = F.cross_entropy(logits.flatten(end_dim=-2), y_target_batch.flatten())
                            scaled_loss = (
                                loss
                                * (X_batch.shape[0] / total_aug_views)
                                * float(config.grad_aug_loss_weight)
                            )

                        scaled_loss.backward()
                        step_loss += float(scaled_loss.detach().cpu())

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
            optimizer.step()
            last_loss = step_loss
            if step_idx % 3 == 0:
                print(
                    f"[ttt-loss] model={model_name} dataset={dataset_name} "
                    f"step={step_idx} loss={step_loss:.6f}",
                    flush=True,
                )
            if _should_save_ttt_ckpt_step(config, step_idx):
                ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, step_idx)
                last_saved_step = step_idx
                print(
                    f"[ttt-ckpt] saved model={model_name} dataset={dataset_name} "
                    f"step={step_idx} path={ckpt_path}",
                    flush=True,
                )

        if _should_save_ttt_final_ckpt(config) and last_loss is not None and last_saved_step != config.steps:
            ckpt_path = _save_ttt_model_ckpt(classifier, config, model_name, dataset_name, config.steps)
            print(
                f"[ttt-ckpt] saved model={model_name} dataset={dataset_name} "
                f"step={config.steps} path={ckpt_path}",
                flush=True,
            )
    finally:
        if hasattr(base_model, "clear_cache"):
            base_model.clear_cache()
        classifier.model_kv_cache_ = None
        base_model.eval()

    return TTTUpdateResult(
        applied=True,
        loss=last_loss,
        steps=config.steps,
        update_seconds=time.time() - update_start,
        reason=None,
        support_aug_X=support_aug_X,
        support_aug_y=support_aug_y,
        grad_aug=grad_aug_diagnostics,
    )


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


def evaluate_one_dataset(
    classifier,
    dataset_dir: Path,
    ttt_config: TTTConfig | None = None,
    *,
    model_name: str,
) -> ResultRow:
    ensure_runtime_deps()
    if ttt_config is None:
        ttt_config = TTTConfig()

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

        ttt_loss = None
        ttt_steps = 0
        ttt_lr = ttt_config.lr if ttt_config.enabled else None
        ttt_applied = False
        ttt_update_seconds = 0.0
        ttt_split_strategy = None
        ttt_split_reason = None
        n_train_b = 0
        n_holdout_c = 0
        grad_aug_diagnostics = GradAugDiagnostics(reason="TTT disabled")

        t0 = time.time()
        if ttt_config.enabled:
            if should_skip_ttt_for_dataset(dataset_dir, info):
                ttt_split_reason = "TTT skipped for dataset=volkert to avoid OOM"
                grad_aug_diagnostics = GradAugDiagnostics(reason=ttt_split_reason)
                classifier.fit(X_train, y_train)
            else:
                split = split_ttt_holdout(y_train, ttt_config)
                ttt_split_strategy = split.strategy
                ttt_split_reason = split.reason
                n_train_b = int(len(split.b_indices))
                n_holdout_c = int(len(split.c_indices))

                if n_train_b > 0 and n_holdout_c > 0:
                    X_b = take_rows(X_train, split.b_indices)
                    y_b = np.asarray(y_train)[split.b_indices]
                    X_c = take_rows(X_train, split.c_indices)
                    y_c = np.asarray(y_train)[split.c_indices]
                    ttt_result = run_ttt_holdout_update(
                        classifier,
                        X_b,
                        y_b,
                        X_c,
                        y_c,
                        ttt_config,
                        model_name=model_name,
                        dataset_name=dataset_dir.name,
                    )
                else:
                    ttt_result = TTTUpdateResult(
                        applied=False,
                        loss=None,
                        steps=0,
                        update_seconds=0.0,
                        reason="B/C split produced an empty side; skipped TTT",
                        grad_aug=GradAugDiagnostics(reason="B/C split produced an empty side; skipped TTT"),
                    )

                ttt_loss = ttt_result.loss
                ttt_steps = ttt_result.steps
                ttt_applied = ttt_result.applied
                ttt_update_seconds = float(ttt_result.update_seconds)
                if ttt_result.grad_aug is not None:
                    grad_aug_diagnostics = ttt_result.grad_aug
                if ttt_result.reason:
                    ttt_split_reason = f"{ttt_split_reason} | {ttt_result.reason}"

                if ttt_applied:
                    support_X = X_train
                    support_y = np.asarray(y_train)
                    if ttt_result.support_aug_X is not None and ttt_result.support_aug_y is not None:
                        support_aug_y = np.asarray(ttt_result.support_aug_y, dtype=np.asarray(y_train).dtype)
                        if len(support_aug_y):
                            support_X = _append_feature_rows(X_train, ttt_result.support_aug_X)
                            support_y = np.concatenate([support_y, support_aug_y], axis=0)
                    _fit_preserving_model_weights(classifier, support_X, support_y)
                else:
                    classifier.fit(X_train, y_train)
        else:
            classifier.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_pred = classifier.predict(X_test)
        predict_seconds = time.time() - t1

        accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))

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
            n_train_a=int(len(y_train)),
            n_train_b=n_train_b,
            n_holdout_c=n_holdout_c,
            n_test_d=int(len(y_test)),
            ttt_loss=ttt_loss,
            ttt_steps=ttt_steps,
            ttt_lr=ttt_lr,
            ttt_applied=ttt_applied,
            ttt_update_seconds=ttt_update_seconds,
            ttt_split_strategy=ttt_split_strategy,
            ttt_split_reason=ttt_split_reason,
            grad_aug_eligible=grad_aug_diagnostics.eligible,
            grad_aug_applied=grad_aug_diagnostics.applied,
            grad_aug_anchor_count=grad_aug_diagnostics.anchor_count,
            grad_aug_candidate_count=grad_aug_diagnostics.candidate_count,
            grad_aug_kept_count=grad_aug_diagnostics.kept_count,
            grad_aug_support_added=grad_aug_diagnostics.support_added,
            grad_aug_mean_cosine=grad_aug_diagnostics.mean_cosine,
            grad_aug_min_cosine=grad_aug_diagnostics.min_cosine,
            grad_aug_reason=grad_aug_diagnostics.reason,
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
    gpu_group: str,
    assigned_dataset_dirs: List[str],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    ttt_config: TTTConfig,
    verbose: bool,
) -> None:
    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_group)
        worker_label = f"worker {worker_id} | gpu {gpu_group}"
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

        torch_diag = collect_torch_diagnostics()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU backend is not available in this worker. "
                f"Diagnostics: {json.dumps(torch_diag, ensure_ascii=False)}"
            )

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = device_str
        classifier = TabICLClassifier(**worker_kwargs)
        model_name = _derive_model_name(
            str(worker_kwargs.get("model_path")) if worker_kwargs.get("model_path") is not None else None,
            str(worker_kwargs.get("checkpoint_version")) if worker_kwargs.get("checkpoint_version") is not None else None,
        )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "gpu_group": gpu_group,
                "status": "ready",
                "assigned_count": len(assigned_dataset_dirs),
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        for dataset_dir in assigned_dataset_dirs:
            row = evaluate_one_dataset(
                classifier,
                Path(dataset_dir),
                worker_ttt_config,
                model_name=model_name,
            )
            rows.append(row)
            print(
                format_dataset_result_log(
                    worker_label,
                    row,
                ),
                flush=True,
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
                asdict(
                    ResultRow(
                        dataset_name=f"__WORKER_CRASH__{worker_id}",
                        dataset_dir="__worker__",
                        task_type=None,
                        n_train=0,
                        n_val=0,
                        n_test=0,
                        n_features=0,
                        n_classes=0,
                        accuracy=None,
                        fit_seconds=0.0,
                        predict_seconds=0.0,
                        status="fail",
                        error=traceback.format_exc(),
                    )
                )
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def model_pool_worker_main(
    worker_id: int,
    gpu_id: int,
    gpu_group: str,
    dataset_dirs: List[str],
    ready_queue,
    task_queue,
    result_queue,
    base_model_kwargs: Dict,
    ttt_config: TTTConfig,
    verbose: bool,
) -> None:
    device_str = "cuda:0"
    worker_label = f"worker {worker_id} | gpu {gpu_group}"

    try:
        ensure_runtime_deps()
        device_str = apply_worker_environment_updates(gpu_group)
        worker_ttt_config = replace(ttt_config, gpu_group=gpu_group)

        import torch
        from tabicl import TabICLClassifier

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
                "gpu_group": gpu_group,
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
            started_at = time.time()
            classifier = None

            try:
                worker_kwargs = dict(base_model_kwargs)
                worker_kwargs["device"] = device_str
                worker_kwargs["model_path"] = normalize_model_path(str(model_path))
                classifier = TabICLClassifier(**worker_kwargs)
                model_name = _derive_model_name(str(model_path), worker_kwargs.get("checkpoint_version"))
                if not worker_ttt_config.enabled:
                    preload_model_once(classifier, worker_label, verbose)

                rows: List[ResultRow] = []
                for dataset_dir in resolved_dataset_dirs:
                    row = evaluate_one_dataset(
                        classifier,
                        dataset_dir,
                        worker_ttt_config,
                        model_name=model_name,
                    )
                    rows.append(row)
                    print(
                        format_dataset_result_log(
                            worker_label,
                            row,
                            model_name=model_path.stem,
                        ),
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

    if len(ok_df) and "grad_aug_eligible" in ok_df.columns:
        eligible_count = int(ok_df["grad_aug_eligible"].fillna(False).astype(bool).sum())
        applied_count = int(ok_df["grad_aug_applied"].fillna(False).astype(bool).sum())
        kept_total = int(pd.to_numeric(ok_df["grad_aug_kept_count"], errors="coerce").fillna(0).sum())
        support_added_total = int(pd.to_numeric(ok_df["grad_aug_support_added"], errors="coerce").fillna(0).sum())
        cosine_series = pd.to_numeric(ok_df["grad_aug_mean_cosine"], errors="coerce").dropna()
        lines.extend(
            [
                f"grad_aug_eligible_ok: {eligible_count}",
                f"grad_aug_applied_ok: {applied_count}",
                f"grad_aug_kept_total_ok: {kept_total}",
                f"grad_aug_support_added_total_ok: {support_added_total}",
                (
                    f"grad_aug_mean_cosine_ok: {cosine_series.mean():.6f}"
                    if len(cosine_series)
                    else "grad_aug_mean_cosine_ok: (none)"
                ),
            ]
        )

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
            "Run independent TabICLv2 gradient-matched TTT classification "
            "benchmarks on data178 with AMD/ROCm multi-GPU workers."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--checkpoint-version", default=DEFAULT_CHECKPOINT_VERSION)
    parser.add_argument("--out-dir", default="1b_result/gradient_ttt_step9_lr5e-6")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-groups", default="1,2,3")
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--batch-size", type=parse_optional_int, default=8)
    parser.add_argument("--kv-cache", type=parse_kv_cache, default=False)
    parser.add_argument("--use-amp", type=parse_auto_bool, default="auto")
    parser.add_argument("--use-fa3", type=parse_auto_bool, default="auto")
    parser.add_argument("--offload-mode", choices=["auto", "gpu", "cpu", "disk"], default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--prefetch-models", type=int, default=4)
    parser.add_argument("--ttt-holdout", default=True, action="store_true")
    parser.add_argument("--ttt-lr", type=float, default=5e-6)
    parser.add_argument("--ttt-scheduler", choices=["constant"], default="constant")
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--ttt-micro-batch-size",
        type=int,
        default=1,
        help=(
            "Per-step TTT micro-batch size. When TTT data parallel is active, "
            "this value is interpreted per GPU and the effective batch becomes "
            "micro_batch_size x number_of_gpus_in_gpu_group."
        ),
    )
    parser.add_argument("--ttt-weight-decay", type=float, default=0.0)
    parser.add_argument("--ttt-steps", type=int, default=9)
    parser.add_argument("--ttt-freeze-col", type=parse_bool, default=True)
    parser.add_argument("--ttt-freeze-row", type=parse_bool, default=True)
    parser.add_argument(
        "--ttt-save-ckpt",
        type=parse_bool,
        default=True,
        help="Whether to save intermediate TabICL checkpoints during the TTT update path.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-every",
        type=int,
        default=9,
        help="Save a TTT checkpoint every N optimizer steps and always save the final step.",
    )
    parser.add_argument(
        "--ttt-save-ckpt-start-step",
        type=parse_optional_int,
        default=None,
        help=(
            "First optimizer step to save a TTT checkpoint. Use None to keep the legacy "
            "multiple-of --ttt-save-ckpt-every schedule."
        ),
    )
    parser.add_argument(
        "--ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_true",
        help="Enable intra-worker multi-GPU data parallelism for the TTT update path.",
    )
    parser.add_argument(
        "--no-ttt-data-parallel",
        dest="ttt_data_parallel",
        action="store_false",
        help="Disable intra-worker multi-GPU data parallelism for the TTT update path.",
    )
    parser.set_defaults(ttt_data_parallel=True)
    parser.add_argument(
        "--grad-aug",
        type=parse_bool,
        default=True,
        help="Enable gradient-matched augmentation for small training sets.",
    )
    parser.add_argument(
        "--grad-aug-max-train-size",
        type=int,
        default=2048,
        help="Apply gradient augmentation only when len(train+val) is at most this value.",
    )
    parser.add_argument(
        "--grad-aug-max-anchor-rows",
        type=int,
        default=64,
        help="Maximum hold-out C rows used as augmentation anchors per dataset.",
    )
    parser.add_argument(
        "--grad-aug-candidates-per-anchor",
        type=int,
        default=4,
        help="Number of same-label semantic candidates generated per anchor.",
    )
    parser.add_argument(
        "--grad-aug-keep-per-anchor",
        type=int,
        default=1,
        help="Maximum gradient-matched candidates retained per anchor.",
    )
    parser.add_argument(
        "--grad-aug-max-kept",
        type=int,
        default=64,
        help="Maximum retained augmented support rows per dataset.",
    )
    parser.add_argument(
        "--grad-aug-min-cosine",
        type=float,
        default=0.80,
        help="Minimum cosine similarity between anchor and candidate gradient signatures.",
    )
    parser.add_argument(
        "--grad-aug-loss-weight",
        type=float,
        default=0.5,
        help="Weight applied to retained augmented rows in the TTT hold-out loss.",
    )
    parser.add_argument(
        "--grad-aug-signature-views",
        type=int,
        default=4,
        help="Maximum ensemble views used to estimate each gradient signature.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
    gpu_ids = parse_gpu_id_list(args.gpus) if args.gpus else detect_default_gpu_ids()
    if args.workers is None:
        args.workers = len(gpu_ids)
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")
    return gpu_ids


def resolve_gpu_assignments(args: argparse.Namespace) -> tuple[List[int], List[str]]:
    if args.gpu_groups:
        if args.gpus:
            raise ValueError("Use either --gpu-groups or --gpus, not both")
        gpu_groups = parse_gpu_group_list(args.gpu_groups)
        if not gpu_groups:
            raise ValueError("--gpu-groups must contain at least one group")
        if args.workers is None:
            args.workers = len(gpu_groups)
        if len(gpu_groups) != args.workers:
            raise ValueError(f"--gpu-groups must contain exactly {args.workers} groups")
        gpu_ids = [first_gpu_id_from_group(gpu_group) for gpu_group in gpu_groups]
        return gpu_ids, gpu_groups

    gpu_ids = resolve_gpu_ids(args)
    return gpu_ids, [str(gpu_id) for gpu_id in gpu_ids]


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
    }


def run_single_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    model_kwargs = build_common_model_kwargs(args)
    model_kwargs["model_path"] = normalize_model_path(args.model_path or DEFAULT_MODEL_PATH)
    ttt_config = build_ttt_config(args)

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
                gpu_groups[worker_id],
                assigned_dirs,
                ready_queue,
                start_event,
                str(worker_csv),
                dict(model_kwargs),
                ttt_config,
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
                    f"[worker {message['worker_id']} | gpu {message.get('gpu_group', message['gpu_id'])}] "
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
    if ttt_config.enabled:
        print("ttt_config:")
        print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))


def run_multi_model_mode(
    args: argparse.Namespace,
    dataset_dirs: List[Path],
    gpu_ids: List[int],
    gpu_groups: List[str],
    out_dir: Path,
) -> None:
    model_paths = discover_model_paths(Path(args.models_dir), max_models=args.max_models)
    if not model_paths:
        raise FileNotFoundError(f"No checkpoint files found under {args.models_dir}")

    worker_count = min(args.workers, len(model_paths))
    base_model_kwargs = build_common_model_kwargs(args)
    ttt_config = build_ttt_config(args)
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
                    gpu_groups[worker_id],
                    [str(path.resolve()) for path in dataset_dirs],
                    ready_queue,
                    task_queue,
                    result_queue,
                    dict(base_model_kwargs),
                    ttt_config,
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
                        f"[worker {message['worker_id']} | gpu {message.get('gpu_group', message['gpu_id'])}] "
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
        if ttt_config.enabled:
            print("ttt_config:")
            print(json.dumps(asdict(ttt_config), indent=2, ensure_ascii=False))
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

    gpu_ids, gpu_groups = resolve_gpu_assignments(args)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.models_dir is not None:
        run_multi_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)
    else:
        run_single_model_mode(args, dataset_dirs, gpu_ids, gpu_groups, out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

import benchmark as tabicl_benchmark


DEFAULT_DATA_ROOT = Path("data178")
DEFAULT_OUT_DIR = Path("result/method_compare")
CLASSIFICATION_TASKS = {"binclass", "multiclass"}
SUPPORTED_METHODS = ("tabpfn25", "limix", "tabpfnv2", "tabr")

BASE_RESULT_COLUMNS = [
    "dataset_name",
    "dataset_dir",
    "task_type",
    "n_train",
    "n_val",
    "n_test",
    "n_features",
    "n_classes",
    "accuracy",
    "fit_seconds",
    "predict_seconds",
    "status",
    "error",
]
EXTRA_RESULT_COLUMNS = [
    "method_name",
    "backend",
    "model_version",
    "model_path",
    "categorical_feature_indices",
    "worker_id",
    "gpu_id",
]
RESULT_COLUMNS = BASE_RESULT_COLUMNS + EXTRA_RESULT_COLUMNS


class SkipDataset(Exception):
    pass


@dataclass
class LoadedDataset:
    dataset_name: str
    dataset_dir: Path
    task_type: Optional[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    n_val: int
    categorical_feature_indices: list[int]

    @property
    def n_classes(self) -> int:
        labels = np.concatenate([np.asarray(self.y_train), np.asarray(self.y_test)], axis=0)
        return int(len(pd.unique(pd.Series(labels))))


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
    method_name: str
    backend: str
    model_version: str
    model_path: Optional[str]
    categorical_feature_indices: str
    worker_id: int
    gpu_id: int


@dataclass
class PredictionResult:
    y_pred: Optional[np.ndarray]
    fit_seconds: float
    predict_seconds: float


class MethodAdapter:
    method_name = "unknown"
    backend = "unknown"
    model_version = "unknown"

    def __init__(self, args: argparse.Namespace, device: str) -> None:
        self.args = args
        self.device = device

    @property
    def model_path(self) -> Optional[str]:
        return None

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        categorical_feature_indices: list[int],
        dataset_name: str,
    ) -> PredictionResult:
        raise NotImplementedError


class TabPFNAdapter(MethodAdapter):
    backend = "tabpfn_oss"

    def __init__(self, args: argparse.Namespace, device: str, *, version_name: str) -> None:
        super().__init__(args, device)
        self.version_name = version_name
        if version_name == "v2_5":
            self.method_name = "tabpfn25"
            self.model_version = "TabPFN-2.5"
            self._model_path = args.tabpfn25_model_path
        elif version_name == "v2":
            self.method_name = "tabpfnv2"
            self.model_version = "TabPFNv2"
            self._model_path = args.tabpfnv2_model_path
        else:
            raise ValueError(f"Unsupported TabPFN version: {version_name}")

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    def _make_classifier(self):
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except Exception as exc:
            raise RuntimeError(f"Failed to import tabpfn: {exc}") from exc

        if self._model_path:
            return TabPFNClassifier(model_path=self._model_path, device=self.device)

        version = ModelVersion.V2_5 if self.version_name == "v2_5" else ModelVersion.V2
        try:
            return TabPFNClassifier.create_default_for_version(version, device=self.device)
        except TypeError:
            classifier = TabPFNClassifier.create_default_for_version(version)
            if hasattr(classifier, "set_params"):
                try:
                    classifier.set_params(device=self.device)
                except Exception:
                    pass
            elif hasattr(classifier, "device"):
                classifier.device = self.device
            return classifier

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        categorical_feature_indices: list[int],
        dataset_name: str,
    ) -> PredictionResult:
        classifier = self._make_classifier()
        X_train_np = X_train.to_numpy()
        X_test_np = X_test.to_numpy()

        fit_started = time.time()
        try:
            classifier.fit(
                X_train_np,
                y_train,
                categorical_feature_indices=categorical_feature_indices or None,
            )
        except TypeError:
            classifier.fit(X_train_np, y_train)
        fit_seconds = time.time() - fit_started

        predict_started = time.time()
        y_pred = np.asarray(classifier.predict(X_test_np))
        predict_seconds = time.time() - predict_started
        return PredictionResult(y_pred=y_pred, fit_seconds=fit_seconds, predict_seconds=predict_seconds)


class LimiXAdapter(MethodAdapter):
    method_name = "limix"
    backend = "limix_oss"
    model_version = "LimiX"

    def __init__(self, args: argparse.Namespace, device: str) -> None:
        super().__init__(args, device)
        self._predictor = None
        self._inference_config = None

    @property
    def model_path(self) -> Optional[str]:
        return self.args.limix_model_path

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor

        if not self.args.limix_root:
            raise RuntimeError("--limix-root is required for method=limix")
        limix_root = Path(self.args.limix_root).expanduser().resolve()
        if not limix_root.exists():
            raise FileNotFoundError(f"LimiX root does not exist: {limix_root}")
        if str(limix_root) not in sys.path:
            sys.path.insert(0, str(limix_root))

        if not self.args.limix_model_path:
            raise RuntimeError("--limix-model-path is required for method=limix")
        if not self.args.limix_config_path:
            raise RuntimeError("--limix-config-path is required for method=limix")

        config_path = Path(self.args.limix_config_path).expanduser()
        with config_path.open("r", encoding="utf-8") as handle:
            self._inference_config = json.load(handle)

        try:
            import torch
            from inference.predictor import LimiXPredictor
        except Exception as exc:
            raise RuntimeError(f"Failed to import LimiX predictor: {exc}") from exc

        self._predictor = LimiXPredictor(
            device=torch.device(self.device),
            model_path=self.args.limix_model_path,
            inference_config=self._inference_config,
            task_type="Classification",
        )
        return self._predictor

    @staticmethod
    def _encode_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        categorical_feature_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

        train = X_train.copy()
        test = X_test.copy()
        cat_cols = [train.columns[idx] for idx in categorical_feature_indices if idx < len(train.columns)]
        num_cols = [col for col in train.columns if col not in set(cat_cols)]

        blocks_train: list[np.ndarray] = []
        blocks_test: list[np.ndarray] = []

        if num_cols:
            num_train = train[num_cols].apply(pd.to_numeric, errors="coerce")
            num_test = test[num_cols].apply(pd.to_numeric, errors="coerce")
            imputer = SimpleImputer(strategy="median")
            scaler = MinMaxScaler()
            arr_train = imputer.fit_transform(num_train)
            arr_test = imputer.transform(num_test)
            blocks_train.append(scaler.fit_transform(arr_train))
            blocks_test.append(scaler.transform(arr_test))

        if cat_cols:
            cat_train = train[cat_cols].astype("string").fillna("__missing__").astype(str)
            cat_test = test[cat_cols].astype("string").fillna("__missing__").astype(str)
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            blocks_train.append(encoder.fit_transform(cat_train))
            blocks_test.append(encoder.transform(cat_test))

        if not blocks_train:
            raise ValueError("No features available after preprocessing")

        return (
            np.concatenate(blocks_train, axis=1).astype(np.float32),
            np.concatenate(blocks_test, axis=1).astype(np.float32),
        )

    @staticmethod
    def _encode_labels(y_train: np.ndarray):
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(np.asarray(y_train))
        return encoder, encoded.astype(np.int64)

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        categorical_feature_indices: list[int],
        dataset_name: str,
    ) -> PredictionResult:
        predictor = self._load_predictor()
        label_encoder, y_train_encoded = self._encode_labels(y_train)
        if len(label_encoder.classes_) > 10 or len(label_encoder.classes_) < 2:
            raise RuntimeError(
                f"LimiX supports 2..10 classes in the reference inference script; got {len(label_encoder.classes_)}"
            )
        if len(X_train) >= 50000:
            raise RuntimeError(f"LimiX reference inference script skips train size >= 50000; got {len(X_train)}")

        X_train_np, X_test_np = self._encode_features(X_train, X_test, categorical_feature_indices)
        fit_seconds = 0.0
        predict_started = time.time()
        prediction = np.asarray(
            predictor.predict(X_train_np, y_train_encoded, X_test_np, task_type="Classification")
        )
        predict_seconds = time.time() - predict_started
        if prediction.ndim == 1:
            pred_encoded = prediction.astype(np.int64)
        else:
            pred_encoded = np.argmax(prediction, axis=1).astype(np.int64)
        y_pred = label_encoder.inverse_transform(pred_encoded)
        return PredictionResult(y_pred=y_pred, fit_seconds=fit_seconds, predict_seconds=predict_seconds)


class TabRAdapter(MethodAdapter):
    method_name = "tabr"
    backend = "tabr_external_repo"
    model_version = "TabR"

    @property
    def model_path(self) -> Optional[str]:
        return self.args.tabr_root

    @staticmethod
    def _write_csv(path: Path, X: pd.DataFrame, y: np.ndarray) -> None:
        frame = X.copy()
        frame["target"] = y
        frame.to_csv(path, index=False)

    @staticmethod
    def _render_template(template: str, values: dict[str, str]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered

    @staticmethod
    def _read_predictions(output_dir: Path) -> np.ndarray:
        candidates = list(output_dir.rglob("predictions.npz")) + list(output_dir.rglob("predictions.npy"))
        if not candidates:
            raise FileNotFoundError(f"No predictions.npz or predictions.npy found under {output_dir}")
        path = candidates[0]
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
        else:
            with np.load(path, allow_pickle=True) as data:
                if "test" in data.files:
                    arr = data["test"]
                elif "test_pred" in data.files:
                    arr = data["test_pred"]
                else:
                    arr = data[data.files[0]]
        arr = np.asarray(arr)
        return np.argmax(arr, axis=1) if arr.ndim == 2 else arr

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        categorical_feature_indices: list[int],
        dataset_name: str,
    ) -> PredictionResult:
        if not self.args.tabr_root:
            raise RuntimeError("--tabr-root is required for method=tabr")
        if not self.args.tabr_template_config:
            raise RuntimeError("--tabr-template-config is required for method=tabr")

        tabr_root = Path(self.args.tabr_root).expanduser().resolve()
        template_path = Path(self.args.tabr_template_config).expanduser().resolve()
        if not tabr_root.exists():
            raise FileNotFoundError(f"TabR root does not exist: {tabr_root}")
        if not template_path.exists():
            raise FileNotFoundError(f"TabR template config does not exist: {template_path}")

        work_root = Path(self.args.out_dir).resolve() / "tabr" / "_tabr_work" / dataset_name
        work_root.mkdir(parents=True, exist_ok=True)
        train_csv = work_root / "train.csv"
        test_csv = work_root / "test.csv"
        config_path = work_root / "config.toml"
        output_dir = work_root / "run"

        label_values = list(dict.fromkeys(np.asarray(y_train).tolist()))
        label_to_idx = {value: idx for idx, value in enumerate(label_values)}
        y_train_encoded = np.asarray([label_to_idx[value] for value in np.asarray(y_train).tolist()], dtype=np.int64)
        self._write_csv(train_csv, X_train, y_train_encoded)
        self._write_csv(test_csv, X_test, np.zeros(len(X_test), dtype=np.int64))

        template = template_path.read_text(encoding="utf-8")
        config_path.write_text(
            self._render_template(
                template,
                {
                    "dataset_name": dataset_name,
                    "train_csv": train_csv.as_posix(),
                    "test_csv": test_csv.as_posix(),
                    "output_dir": output_dir.as_posix(),
                    "n_features": str(X_train.shape[1]),
                    "n_classes": str(len(label_values)),
                    "tabr_max_epochs": str(self.args.tabr_max_epochs),
                },
            ),
            encoding="utf-8",
        )

        command = [sys.executable, "bin/tabr.py", str(config_path), "--force"]
        started = time.time()
        completed = subprocess.run(
            command,
            cwd=tabr_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        fit_seconds = time.time() - started
        if completed.returncode != 0:
            raise RuntimeError(
                "TabR command failed with exit code "
                f"{completed.returncode}: {completed.stderr[-2000:] or completed.stdout[-2000:]}"
            )

        predict_started = time.time()
        pred_encoded = self._read_predictions(output_dir)
        predict_seconds = time.time() - predict_started
        idx_to_label = {idx: value for value, idx in label_to_idx.items()}
        y_pred = np.asarray([idx_to_label[int(idx)] for idx in np.asarray(pred_encoded).tolist()])
        return PredictionResult(y_pred=y_pred, fit_seconds=fit_seconds, predict_seconds=predict_seconds)


def parse_methods(value: str) -> list[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in methods if item not in SUPPORTED_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported methods: {', '.join(unknown)}. Supported: {', '.join(SUPPORTED_METHODS)}"
        )
    return methods


def detect_gpu_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass

    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            return sum(1 for line in completed.stdout.splitlines() if line.strip().startswith("GPU "))
    except Exception:
        pass

    return 0


def resolve_workers_and_gpu_ids(args: argparse.Namespace) -> tuple[int, list[int]]:
    if args.gpus is None or str(args.gpus).strip().lower() == "auto":
        detected_count = detect_gpu_count()
        if detected_count <= 0:
            raise RuntimeError(
                "No GPUs detected. Pass --gpus with explicit ids if GPUs are hidden by the environment."
            )
        gpu_ids = list(range(detected_count))
    else:
        gpu_ids = [int(item.strip()) for item in str(args.gpus).split(",") if item.strip()]
        if not gpu_ids:
            raise ValueError("--gpus must contain at least one GPU id or use 'auto'")

    workers = len(gpu_ids) if args.workers is None else int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if len(gpu_ids) != workers:
        raise ValueError(
            f"--gpus must contain exactly --workers ids; got {len(gpu_ids)} gpu ids for {workers} workers"
        )
    return workers, gpu_ids


def categorical_indices_from_split(train_split) -> list[int]:
    num_path, cat_path, _ = train_split
    n_num = 0
    n_cat = 0
    if num_path is not None:
        arr = tabicl_benchmark.load_array(Path(num_path))
        n_num = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    if cat_path is not None:
        arr = tabicl_benchmark.load_array(Path(cat_path))
        n_cat = int(arr.reshape(arr.shape[0], -1).shape[1]) if arr.ndim > 1 else 1
    return list(range(n_num, n_num + n_cat))


def load_classification_dataset(dataset_dir: Path) -> LoadedDataset:
    info = tabicl_benchmark.load_dataset_info(dataset_dir)
    task_type = str(info.get("task_type", "")).lower() if info else None
    if task_type not in CLASSIFICATION_TASKS:
        raise SkipDataset(f"Skipped due to task_type={task_type!r}")

    train_split, val_split, test_split = tabicl_benchmark.find_split_files(dataset_dir)
    categorical_feature_indices = categorical_indices_from_split(train_split)

    X_train, y_train = tabicl_benchmark.load_split(
        train_split[0],
        train_split[1],
        train_split[2],
        context=f"{dataset_dir.name}-train",
    )
    train_count = int(len(y_train))

    n_val = 0
    if val_split is not None:
        X_val, y_val = tabicl_benchmark.load_split(
            val_split[0],
            val_split[1],
            val_split[2],
            context=f"{dataset_dir.name}-val",
        )
        n_val = int(len(y_val))
        X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_train = np.concatenate([np.asarray(y_train), np.asarray(y_val)], axis=0)

    X_test, y_test = tabicl_benchmark.load_split(
        test_split[0],
        test_split[1],
        test_split[2],
        context=f"{dataset_dir.name}-test",
    )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch after loading: train has {X_train.shape[1]}, test has {X_test.shape[1]}"
        )
    if len(y_train) != train_count + n_val:
        raise ValueError("Internal train/val merge count mismatch")

    return LoadedDataset(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        task_type=task_type,
        X_train=X_train,
        y_train=np.asarray(y_train),
        X_test=X_test,
        y_test=np.asarray(y_test),
        n_val=n_val,
        categorical_feature_indices=categorical_feature_indices,
    )


def make_adapter(method: str, args: argparse.Namespace, device: str) -> MethodAdapter:
    if method == "tabpfn25":
        return TabPFNAdapter(args, device, version_name="v2_5")
    if method == "tabpfnv2":
        return TabPFNAdapter(args, device, version_name="v2")
    if method == "limix":
        return LimiXAdapter(args, device)
    if method == "tabr":
        return TabRAdapter(args, device)
    raise ValueError(f"Unsupported method: {method}")


def empty_row_for_dataset(
    dataset_dir: Path,
    adapter: MethodAdapter,
    worker_id: int,
    gpu_id: int,
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
        fit_seconds=0.0,
        predict_seconds=0.0,
        status=status,
        error=error,
        method_name=adapter.method_name,
        backend=adapter.backend,
        model_version=adapter.model_version,
        model_path=adapter.model_path,
        categorical_feature_indices="[]",
        worker_id=worker_id,
        gpu_id=gpu_id,
    )


def evaluate_one_dataset(
    adapter: MethodAdapter,
    dataset_dir: Path,
    worker_id: int,
    gpu_id: int,
) -> ResultRow:
    task_type: Optional[str] = None
    try:
        loaded = load_classification_dataset(dataset_dir)
        task_type = loaded.task_type
        result = adapter.fit_predict(
            loaded.X_train,
            loaded.y_train,
            loaded.X_test,
            loaded.categorical_feature_indices,
            loaded.dataset_name,
        )
        if result.y_pred is None:
            raise RuntimeError("Adapter returned no predictions")
        y_pred = np.asarray(result.y_pred)
        if len(y_pred) != len(loaded.y_test):
            raise ValueError(f"Prediction length mismatch: got {len(y_pred)}, expected {len(loaded.y_test)}")
        accuracy = float(np.mean(y_pred == np.asarray(loaded.y_test)))
        return ResultRow(
            dataset_name=loaded.dataset_name,
            dataset_dir=loaded.dataset_dir.as_posix(),
            task_type=loaded.task_type,
            n_train=int(len(loaded.y_train)),
            n_val=loaded.n_val,
            n_test=int(len(loaded.y_test)),
            n_features=int(loaded.X_train.shape[1]),
            n_classes=loaded.n_classes,
            accuracy=accuracy,
            fit_seconds=float(result.fit_seconds),
            predict_seconds=float(result.predict_seconds),
            status="ok",
            error=None,
            method_name=adapter.method_name,
            backend=adapter.backend,
            model_version=adapter.model_version,
            model_path=adapter.model_path,
            categorical_feature_indices=json.dumps(loaded.categorical_feature_indices),
            worker_id=worker_id,
            gpu_id=gpu_id,
        )
    except SkipDataset as exc:
        return empty_row_for_dataset(dataset_dir, adapter, worker_id, gpu_id, "skip", str(exc), task_type=task_type)
    except Exception as exc:
        return empty_row_for_dataset(
            dataset_dir,
            adapter,
            worker_id,
            gpu_id,
            "fail",
            f"{type(exc).__name__}: {exc}",
            task_type=task_type,
        )


def rows_to_frame(rows: Iterable[ResultRow]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[RESULT_COLUMNS]


def write_method_summary(summary_path: Path, result_df: pd.DataFrame, dataset_dirs: list[Path], wall_seconds: float) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {len(dataset_dirs)}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"skipped_count: {len(skipped_df)}",
        f"avg_accuracy_ok: {ok_df['accuracy'].mean():.6f}" if len(ok_df) else "avg_accuracy_ok: (none)",
        (
            f"avg_fit_seconds_ok: {ok_df['fit_seconds'].mean():.6f}"
            if len(ok_df)
            else "avg_fit_seconds_ok: (none)"
        ),
        (
            f"avg_predict_seconds_ok: {ok_df['predict_seconds'].mean():.6f}"
            if len(ok_df)
            else "avg_predict_seconds_ok: (none)"
        ),
        f"wall_seconds: {wall_seconds:.3f}",
    ]
    lines.append(
        "failed_datasets: "
        + (", ".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "(none)")
    )
    lines.append(
        "skipped_datasets: "
        + (", ".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "(none)")
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_method_frame(method_name: str, result_df: pd.DataFrame, wall_seconds: float) -> dict[str, Any]:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()
    skipped_df = result_df[result_df["status"] == "skip"].copy() if len(result_df) else pd.DataFrame()
    return {
        "method_name": method_name,
        "processed_datasets": int(len(result_df)),
        "ok_count": int(len(ok_df)),
        "failed_count": int(len(failed_df)),
        "skipped_count": int(len(skipped_df)),
        "avg_accuracy_ok": float(ok_df["accuracy"].mean()) if len(ok_df) else None,
        "avg_fit_seconds_ok": float(ok_df["fit_seconds"].mean()) if len(ok_df) else None,
        "avg_predict_seconds_ok": float(ok_df["predict_seconds"].mean()) if len(ok_df) else None,
        "avg_dataset_seconds_ok": (
            float((ok_df["fit_seconds"] + ok_df["predict_seconds"]).mean()) if len(ok_df) else None
        ),
        "wall_seconds": float(wall_seconds),
        "failed_datasets": ",".join(failed_df["dataset_name"].astype(str).tolist()) if len(failed_df) else "",
        "skipped_datasets": ",".join(skipped_df["dataset_name"].astype(str).tolist()) if len(skipped_df) else "",
    }


def write_comparison_markdown(path: Path, summary_df: pd.DataFrame) -> None:
    lines = ["# Method Comparison", ""]
    if summary_df.empty:
        lines.append("(no methods processed)")
    else:
        sortable = summary_df.copy()
        sortable["avg_accuracy_ok_sort"] = pd.to_numeric(sortable["avg_accuracy_ok"], errors="coerce")
        sortable = sortable.sort_values("avg_accuracy_ok_sort", ascending=False, na_position="last")
        lines.extend(
            [
                "| method | ok | fail | skip | avg_accuracy_ok | avg_fit_s | avg_predict_s | wall_s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in sortable.iterrows():
            lines.append(
                "| {method} | {ok} | {fail} | {skip} | {acc} | {fit} | {pred} | {wall} |".format(
                    method=row["method_name"],
                    ok=int(row["ok_count"]),
                    fail=int(row["failed_count"]),
                    skip=int(row["skipped_count"]),
                    acc=(
                        f"{float(row['avg_accuracy_ok']):.6f}"
                        if pd.notna(row["avg_accuracy_ok"])
                        else "(none)"
                    ),
                    fit=(
                        f"{float(row['avg_fit_seconds_ok']):.3f}"
                        if pd.notna(row["avg_fit_seconds_ok"])
                        else "(none)"
                    ),
                    pred=(
                        f"{float(row['avg_predict_seconds_ok']):.3f}"
                        if pd.notna(row["avg_predict_seconds_ok"])
                        else "(none)"
                    ),
                    wall=f"{float(row['wall_seconds']):.3f}",
                )
            )

        lines.extend(["", "## Failed / Skipped Datasets", ""])
        for _, row in sortable.iterrows():
            lines.append(f"### {row['method_name']}")
            lines.append(f"- failed: {row['failed_datasets'] or '(none)'}")
            lines.append(f"- skipped: {row['skipped_datasets'] or '(none)'}")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def worker_main(
    method: str,
    worker_id: int,
    gpu_id: int,
    dataset_dirs: list[str],
    worker_csv: str,
    args_dict: dict[str, Any],
) -> None:
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = argparse.Namespace(**args_dict)
    adapter = make_adapter(method, args, device="cuda:0")
    rows: list[ResultRow] = []
    try:
        for dataset_dir in dataset_dirs:
            row = evaluate_one_dataset(adapter, Path(dataset_dir), worker_id, gpu_id)
            rows.append(row)
            if args.verbose:
                if row.status == "ok":
                    print(
                        f"[{method} worker {worker_id} | gpu {gpu_id}] "
                        f"[ok] {row.dataset_name} accuracy={row.accuracy:.6f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{method} worker {worker_id} | gpu {gpu_id}] "
                        f"[{row.status}] {row.dataset_name} error={row.error}",
                        flush=True,
                    )
    except Exception:
        rows.append(
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
                method_name=adapter.method_name,
                backend=adapter.backend,
                model_version=adapter.model_version,
                model_path=adapter.model_path,
                categorical_feature_indices="[]",
                worker_id=worker_id,
                gpu_id=gpu_id,
            )
        )
    rows_to_frame(rows).to_csv(worker_csv, index=False)


def run_method(
    method: str,
    args: argparse.Namespace,
    dataset_dirs: list[Path],
    gpu_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method_out_dir = Path(args.out_dir) / method
    method_out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    worker_csvs: list[Path] = []
    processes: list[mp.Process] = []
    args_dict = vars(args).copy()

    for worker_id in range(args.workers):
        assigned = [str(path.resolve()) for path in dataset_dirs[worker_id:: args.workers]]
        worker_csv = method_out_dir / f"worker_{worker_id}.csv"
        worker_csvs.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(method, worker_id, gpu_ids[worker_id], assigned, str(worker_csv), args_dict),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    frames = [pd.read_csv(path) for path in worker_csvs if path.exists()]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESULT_COLUMNS)
    result_df = result_df[RESULT_COLUMNS]
    all_csv = method_out_dir / "all_classification_results.csv"
    summary_txt = method_out_dir / "summary.txt"
    result_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - started
    write_method_summary(summary_txt, result_df, dataset_dirs, wall_seconds)
    summary = summarize_method_frame(method, result_df, wall_seconds)

    print(f"[{method}] saved_all_csv: {all_csv}")
    print(f"[{method}] saved_summary: {summary_txt}")
    return result_df, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare TabPFN-2.5, LimiX, TabPFNv2, and TabR on benchmark.py-style datasets."
    )
    parser.add_argument("--methods", type=parse_methods, default=list(SUPPORTED_METHODS))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to the number of resolved GPUs.",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="Comma-separated physical GPU ids, or 'auto' to use all detected GPUs.",
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--limix-root", default=None)
    parser.add_argument("--limix-model-path", default=None)
    parser.add_argument("--limix-config-path", default=None)
    parser.add_argument("--tabr-root", default=None)
    parser.add_argument("--tabr-template-config", default=None)
    parser.add_argument("--tabr-max-epochs", type=int, default=100)
    parser.add_argument("--tabpfn25-model-path", default=None)
    parser.add_argument("--tabpfnv2-model-path", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = tabicl_benchmark.find_dataset_dirs(data_root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset directories found under {data_root}")

    args.workers, gpu_ids = resolve_workers_and_gpu_ids(args)
    if args.verbose:
        print(f"resolved_workers: {args.workers}", flush=True)
        print(f"resolved_gpus: {','.join(str(gpu_id) for gpu_id in gpu_ids)}", flush=True)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    all_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for method in args.methods:
        result_df, summary = run_method(method, args, dataset_dirs, gpu_ids)
        all_frames.append(result_df)
        summaries.append(summary)

    all_results = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=RESULT_COLUMNS)
    all_summary = pd.DataFrame(summaries)
    all_results.to_csv(out_dir / "all_methods_results.csv", index=False)
    all_summary.to_csv(out_dir / "all_methods_summary.csv", index=False)
    write_comparison_markdown(out_dir / "method_comparison.md", all_summary)
    print(f"saved_all_methods_results: {out_dir / 'all_methods_results.csv'}")
    print(f"saved_all_methods_summary: {out_dir / 'all_methods_summary.csv'}")
    print(f"saved_method_comparison: {out_dir / 'method_comparison.md'}")


if __name__ == "__main__":
    main()

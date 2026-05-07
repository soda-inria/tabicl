#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
TABR_DIR = PIPELINE_DIR / "tabular-dl-tabr"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(TABR_DIR) not in sys.path:
    sys.path.insert(0, str(TABR_DIR))

import infer_shared as shared


def _as_float_frame(frame, columns: list[int], medians=None):
    shared.ensure_runtime_deps()
    values = frame.iloc[:, columns].apply(shared.pd.to_numeric, errors="coerce")
    if medians is None:
        medians = values.median(axis=0, skipna=True).fillna(0.0)
    values = values.fillna(medians).fillna(0.0)
    return values.astype("float32"), medians


def _as_cat_array(frame, columns: list[int]):
    shared.ensure_runtime_deps()
    values = frame.iloc[:, columns].astype("string").fillna(shared.CATEGORICAL_MISSING_TOKEN)
    return values.astype(str).to_numpy()


def _split_train_for_val(bundle: shared.DatasetBundle):
    from sklearn.model_selection import train_test_split

    try:
        return train_test_split(
            bundle.X_train,
            bundle.y_train,
            test_size=0.2,
            random_state=0,
            stratify=bundle.y_train if len(set(bundle.y_train)) > 1 else None,
        )
    except Exception:
        return train_test_split(
            bundle.X_train,
            bundle.y_train,
            test_size=0.2,
            random_state=0,
            stratify=None,
        )


class TaBRPredictor:
    def __init__(self, args: dict[str, Any], worker_id: int) -> None:
        self.args = args
        self.worker_id = worker_id
        self.out_dir = Path(args["out_dir"]).resolve()
        self.data_root = self.out_dir / "tabr_data" / f"worker_{worker_id}"
        self.run_root = self.out_dir / "tabr_runs" / f"worker_{worker_id}"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._tabr_main = None

    def _load_tabr(self):
        if self._tabr_main is not None:
            return self._tabr_main

        os.environ.setdefault("PROJECT_DIR", str(TABR_DIR.resolve()))
        if str(TABR_DIR) not in sys.path:
            sys.path.insert(0, str(TABR_DIR))

        import lib

        lib.configure_libraries()
        tabr_module = importlib.import_module("bin.tabr")
        self._tabr_main = tabr_module.main
        return self._tabr_main

    def _write_tabr_dataset(self, bundle: shared.DatasetBundle, data_dir: Path):
        shared.ensure_runtime_deps()
        from sklearn.preprocessing import LabelEncoder

        if bundle.X_val is None or bundle.y_val is None or len(bundle.y_val) == 0:
            if len(bundle.y_train) < 5:
                raise ValueError("TaBR needs a validation split or at least 5 training rows")
            X_train, X_val, y_train, y_val = _split_train_for_val(bundle)
        else:
            X_train, y_train = bundle.X_train, bundle.y_train
            X_val, y_val = bundle.X_val, bundle.y_val

        X_test, y_test = bundle.X_test, bundle.y_test
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        if len(le.classes_) < 2:
            raise ValueError("TaBR needs at least two classes in the training split")

        cat_cols = list(bundle.categorical_feature_indices)
        num_cols = [idx for idx in range(X_train.shape[1]) if idx not in set(cat_cols)]

        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if num_cols:
            X_num_train, medians = _as_float_frame(X_train, num_cols)
            X_num_val, _ = _as_float_frame(X_val, num_cols, medians=medians)
            X_num_test, _ = _as_float_frame(X_test, num_cols, medians=medians)
            shared.np.save(data_dir / "X_num_train.npy", X_num_train.to_numpy())
            shared.np.save(data_dir / "X_num_val.npy", X_num_val.to_numpy())
            shared.np.save(data_dir / "X_num_test.npy", X_num_test.to_numpy())

        if cat_cols:
            shared.np.save(data_dir / "X_cat_train.npy", _as_cat_array(X_train, cat_cols))
            shared.np.save(data_dir / "X_cat_val.npy", _as_cat_array(X_val, cat_cols))
            shared.np.save(data_dir / "X_cat_test.npy", _as_cat_array(X_test, cat_cols))

        shared.np.save(data_dir / "Y_train.npy", y_train_enc.astype("int64"))
        shared.np.save(data_dir / "Y_val.npy", y_val_enc.astype("int64"))
        shared.np.save(data_dir / "Y_test.npy", y_test_enc.astype("int64"))
        (data_dir / "info.json").write_text(
            json.dumps({"task_type": bundle.task_type}, indent=2) + "\n",
            encoding="utf-8",
        )
        return le, len(y_train_enc), bool(num_cols), bool(cat_cols)

    def _make_config(self, data_dir: Path, train_size: int, has_num: bool, has_cat: bool) -> dict[str, Any]:
        context_size = max(1, min(int(self.args["context_size"]), max(1, train_size - 1)))
        config: dict[str, Any] = {
            "seed": int(self.args["random_state"]),
            "batch_size": int(self.args["batch_size"]),
            "patience": None if self.args["patience"] is None else int(self.args["patience"]),
            "n_epochs": int(self.args["n_epochs"]),
            "context_size": context_size,
            "data": {
                "path": str(data_dir),
                "num_policy": self.args["num_policy"] if has_num else None,
                "cat_policy": self.args["cat_policy"] if has_cat else None,
                "y_policy": None,
                "score": "accuracy",
                "seed": int(self.args["random_state"]),
                "cache": False,
            },
            "model": {
                "num_embeddings": None,
                "d_main": int(self.args["d_main"]),
                "d_multiplier": float(self.args["d_multiplier"]),
                "encoder_n_blocks": int(self.args["encoder_n_blocks"]),
                "predictor_n_blocks": int(self.args["predictor_n_blocks"]),
                "mixer_normalization": "auto",
                "context_dropout": float(self.args["context_dropout"]),
                "dropout0": float(self.args["dropout0"]),
                "dropout1": float(self.args["dropout1"]),
                "normalization": "LayerNorm",
                "activation": "ReLU",
                "memory_efficient": bool(self.args["memory_efficient"]),
                "candidate_encoding_batch_size": self.args["candidate_encoding_batch_size"],
            },
            "optimizer": {
                "type": "AdamW",
                "lr": float(self.args["lr"]),
                "weight_decay": float(self.args["weight_decay"]),
            },
        }
        return config

    def fit_predict(self, bundle: shared.DatasetBundle):
        tabr_main = self._load_tabr()
        dataset_key = shared.sanitize_name(bundle.dataset_name)
        data_dir = self.data_root / dataset_key
        run_dir = self.run_root / dataset_key
        label_encoder, train_size, has_num, has_cat = self._write_tabr_dataset(bundle, data_dir)
        config = self._make_config(data_dir, train_size, has_num, has_cat)
        fit_start = time.time()
        report = tabr_main(config, run_dir, force=True)
        fit_seconds = time.time() - fit_start
        if report is None:
            raise RuntimeError(f"TaBR did not run for {bundle.dataset_name}")

        predict_start = time.time()
        predictions_npz = shared.np.load(run_dir / "predictions.npz")
        logits = shared.np.asarray(predictions_npz["test"])
        if bundle.task_type == "binclass":
            proba = 1.0 / (1.0 + shared.np.exp(-logits))
            encoded_pred = shared.np.round(proba).astype("int64").reshape(-1)
        else:
            encoded_pred = logits.argmax(axis=1).astype("int64")
        y_pred = label_encoder.inverse_transform(encoded_pred)
        predict_seconds = time.time() - predict_start
        return y_pred, fit_seconds, predict_seconds


def build_predictor(args: dict[str, Any], worker_id: int, gpu_id: int) -> TaBRPredictor:
    del gpu_id
    return TaBRPredictor(args, worker_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TaBR classification benchmark training/inference.")
    shared.add_common_args(parser, default_out_dir="pipeline_compare_result/tabr")
    parser.add_argument("--n-epochs", type=int, default=128)
    parser.add_argument("--patience", type=shared.parse_optional_int, default=16)
    parser.add_argument("--context-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-policy", choices=["standard", "quantile"], default="quantile")
    parser.add_argument("--cat-policy", choices=["ordinal", "one-hot"], default="ordinal")
    parser.add_argument("--d-main", type=int, default=265)
    parser.add_argument("--d-multiplier", type=float, default=2.0)
    parser.add_argument("--encoder-n-blocks", type=int, default=0)
    parser.add_argument("--predictor-n-blocks", type=int, default=1)
    parser.add_argument("--context-dropout", type=float, default=0.38920071545944357)
    parser.add_argument("--dropout0", type=float, default=0.38852797479169876)
    parser.add_argument("--dropout1", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3.1212736415169e-4)
    parser.add_argument("--weight-decay", type=float, default=1.2260352006404615e-6)
    parser.add_argument("--memory-efficient", action="store_true")
    parser.add_argument("--candidate-encoding-batch-size", type=shared.parse_optional_int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    shared.run_benchmark(args, model_factory=build_predictor, merge_val_into_train=False)


if __name__ == "__main__":
    main()

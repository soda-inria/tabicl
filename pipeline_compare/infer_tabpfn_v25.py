#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
TABPFN_SRC = PIPELINE_DIR / "TabPFN-main" / "src"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(TABPFN_SRC) not in sys.path:
    sys.path.insert(0, str(TABPFN_SRC))

import infer_shared as shared


class TabPFNVersionPredictor:
    def __init__(self, args: dict[str, Any], version_name: str) -> None:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        version = ModelVersion(version_name)
        overrides: dict[str, Any] = {
            "n_estimators": args["n_estimators"],
            "device": args["device"],
            "ignore_pretraining_limits": args["ignore_pretraining_limits"],
            "fit_mode": args["fit_mode"],
            "memory_saving_mode": args["memory_saving_mode"],
            "inference_precision": args["inference_precision"],
            "random_state": args["random_state"],
        }
        if args.get("model_path"):
            overrides["model_path"] = args["model_path"]
        self.classifier = TabPFNClassifier.create_default_for_version(version, **overrides)

    def fit_predict(self, bundle: shared.DatasetBundle):
        self.classifier.categorical_features_indices = list(bundle.categorical_feature_indices)
        fit_start = time.time()
        self.classifier.fit(bundle.X_train, bundle.y_train)
        fit_seconds = time.time() - fit_start
        predict_start = time.time()
        y_pred = self.classifier.predict(bundle.X_test)
        predict_seconds = time.time() - predict_start
        return y_pred, fit_seconds, predict_seconds


def build_predictor(args: dict[str, Any], worker_id: int, gpu_id: int) -> TabPFNVersionPredictor:
    del worker_id, gpu_id
    return TabPFNVersionPredictor(args, "v2.5")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TabPFN v2.5 classification benchmark inference.")
    shared.add_common_args(parser, default_out_dir="pipeline_compare_result/tabpfn_v25")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-mode", default="fit_preprocessors")
    parser.add_argument("--memory-saving-mode", default="auto")
    parser.add_argument("--inference-precision", default="auto")
    parser.add_argument("--ignore-pretraining-limits", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    shared.run_benchmark(args, model_factory=build_predictor, merge_val_into_train=True)


if __name__ == "__main__":
    main()

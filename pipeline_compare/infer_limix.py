#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
LIMIX_DIR = PIPELINE_DIR / "LimiX"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(LIMIX_DIR) not in sys.path:
    sys.path.insert(0, str(LIMIX_DIR))

import infer_shared as shared


def resolve_limix_config(config_path: str) -> str:
    path = Path(config_path)
    candidates = [path, LIMIX_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"LimiX inference config not found: {config_path}")


def resolve_limix_model(args: dict[str, Any]) -> str:
    model_path = args.get("model_path")
    if model_path:
        path = Path(str(model_path)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LimiX model checkpoint not found: {path}")
        return str(path.resolve())

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "LimiX --model-path was not provided and huggingface_hub is unavailable "
            "for automatic download."
        ) from exc

    cache_dir = Path(args["cache_dir"]).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=args["hf_repo_id"],
        filename=args["hf_filename"],
        local_dir=str(cache_dir),
    )


class LimiXPipelinePredictor:
    def __init__(self, args: dict[str, Any]) -> None:
        import torch
        from inference.predictor import LimiXPredictor

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args["master_port"]))

        config_path = resolve_limix_config(args["inference_config_path"])
        model_path = resolve_limix_model(args)
        self.predictor = LimiXPredictor(
            device=torch.device(args["device"]),
            model_path=model_path,
            inference_config=config_path,
            mix_precision=not args["disable_amp"],
            inference_with_DDP=False,
            seed=args["random_state"],
        )

    def fit_predict(self, bundle: shared.DatasetBundle):
        shared.ensure_runtime_deps()
        predict_start = time.time()
        proba = self.predictor.predict(
            shared.np.asarray(bundle.X_train),
            shared.np.asarray(bundle.y_train),
            shared.np.asarray(bundle.X_test),
            task_type="Classification",
        )
        predict_seconds = time.time() - predict_start
        encoded = shared.np.asarray(proba).argmax(axis=1)
        classes = getattr(self.predictor, "classes", None)
        if classes is None:
            y_pred = encoded
        else:
            y_pred = shared.np.asarray(classes)[encoded]
        return y_pred, 0.0, predict_seconds


def build_predictor(args: dict[str, Any], worker_id: int, gpu_id: int) -> LimiXPipelinePredictor:
    del worker_id, gpu_id
    return LimiXPipelinePredictor(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LimiX classification benchmark inference.")
    shared.add_common_args(parser, default_out_dir="pipeline_compare_result/limix")
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--inference-config-path",
        default=str(LIMIX_DIR / "config" / "cls_default_noretrieval.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--cache-dir", default=str(LIMIX_DIR / "cache"))
    parser.add_argument("--hf-repo-id", default="stableai-org/LimiX-16M")
    parser.add_argument("--hf-filename", default="LimiX-16M.ckpt")
    parser.add_argument("--master-port", type=int, default=29500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    shared.run_benchmark(args, model_factory=build_predictor, merge_val_into_train=True)


if __name__ == "__main__":
    main()

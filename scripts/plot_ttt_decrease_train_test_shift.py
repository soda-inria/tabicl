#!/usr/bin/env python3
"""Draw class-faceted train/test t-SNE plots for regressed TTT datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.manifold import TSNE

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_data178_tsne import (  # noqa: E402
    clean_numeric,
    encode_categorical,
    label_codes,
    load_optional_array,
    reduce_for_tsne,
    stratified_sample_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For datasets listed in ttt_decrease/decrease_manifest.csv, draw one PNG per "
            "dataset where every class is shown in a separate panel. Train and test samples "
            "are colored differently to inspect distribution shift."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data178"))
    parser.add_argument("--manifest", type=Path, default=Path("ttt_decrease/decrease_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("ttt_decrease/train_test_shift"))
    parser.add_argument("--max-samples-per-split", type=int, default=2000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--max-categories", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


def compute_embedding(dataset_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    stable_offset = int(hashlib.md5(dataset_dir.name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.random_state + stable_offset % 1_000_000)

    y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
    y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=True)
    train_idx = stratified_sample_indices(y_train, args.max_samples_per_split, rng)
    test_idx = stratified_sample_indices(y_test, args.max_samples_per_split, rng)

    n_train = load_optional_array(dataset_dir, "N_train.npy")
    n_test = load_optional_array(dataset_dir, "N_test.npy")
    c_train = load_optional_array(dataset_dir, "C_train.npy")
    c_test = load_optional_array(dataset_dir, "C_test.npy")

    numeric = None
    categorical = None
    feature_count = 0
    if n_train is not None and n_test is not None:
        numeric = np.vstack([n_train[train_idx], n_test[test_idx]])
        feature_count += numeric.shape[1] if numeric.ndim > 1 else 1
    if c_train is not None and c_test is not None:
        categorical = np.vstack([c_train[train_idx], c_test[test_idx]])
        feature_count += categorical.shape[1] if categorical.ndim > 1 else 1

    blocks = []
    numeric_block = clean_numeric(numeric)
    if numeric_block is not None:
        blocks.append(numeric_block)
    categorical_block = encode_categorical(categorical, args.max_categories)
    if categorical_block is not None:
        blocks.append(categorical_block)
    if not blocks:
        raise ValueError("missing both numeric and categorical features")

    x = sparse.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    y = np.concatenate([y_train[train_idx], y_test[test_idx]])
    split = np.asarray(["train"] * len(train_idx) + ["test"] * len(test_idx))

    if x.shape[0] < 3:
        raise ValueError("not enough samples for t-SNE")

    reduced = reduce_for_tsne(x, args.pca_dims, args.random_state)
    perplexity = min(args.perplexity, max(1.0, math.floor((x.shape[0] - 1) / 3)))
    init = "pca" if reduced.shape[1] >= 2 and reduced.shape[0] > 2 else "random"
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init=init,
        learning_rate="auto",
        random_state=args.random_state,
        n_iter_without_progress=300,
    ).fit_transform(reduced)

    return {
        "embedding": embedding,
        "y": y,
        "split": split,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "sampled_train": len(train_idx),
        "sampled_test": len(test_idx),
        "n_features_raw": feature_count,
        "n_features_encoded": x.shape[1],
        "perplexity": perplexity,
    }


def plot_class_facets(
    dataset_name: str,
    result_row: pd.Series | None,
    payload: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    embedding = payload["embedding"]
    split = payload["split"]
    y = payload["y"]
    codes, classes = label_codes(y)
    n_classes = len(classes)

    cols = min(4, n_classes)
    rows = math.ceil(n_classes / cols)
    fig_w = max(4.4 * cols, 7.5)
    fig_h = 3.8 * rows + 1.2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()

    x_min, x_max = np.percentile(embedding[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(embedding[:, 1], [0.5, 99.5])
    x_pad = max((x_max - x_min) * 0.06, 1e-6)
    y_pad = max((y_max - y_min) * 0.06, 1e-6)

    train_color = "#2563eb"
    test_color = "#f97316"
    background_color = "#d1d5db"

    for class_idx, class_label in enumerate(classes):
        ax = axes_flat[class_idx]
        class_mask = codes == class_idx
        other_mask = ~class_mask
        train_mask = class_mask & (split == "train")
        test_mask = class_mask & (split == "test")

        ax.scatter(
            embedding[other_mask, 0],
            embedding[other_mask, 1],
            s=8,
            c=background_color,
            alpha=0.11,
            marker="o",
            linewidths=0,
        )
        ax.scatter(
            embedding[train_mask, 0],
            embedding[train_mask, 1],
            s=24,
            facecolors="none",
            edgecolors=train_color,
            alpha=0.72,
            linewidths=0.8,
            marker="o",
            label=f"train ({int(train_mask.sum())})",
        )
        ax.scatter(
            embedding[test_mask, 0],
            embedding[test_mask, 1],
            s=24,
            facecolors="none",
            edgecolors=test_color,
            alpha=0.82,
            linewidths=0.9,
            marker="o",
            label=f"test ({int(test_mask.sum())})",
        )
        ax.set_title(f"class {class_label}", fontsize=10)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, linewidth=0.3, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(loc="best", fontsize=7, frameon=True)

    for ax in axes_flat[n_classes:]:
        ax.axis("off")

    sample_note = (
        f"sample train={payload['sampled_train']}/{payload['n_train']}, "
        f"test={payload['sampled_test']}/{payload['n_test']}"
    )
    delta_note = ""
    if result_row is not None:
        delta_note = (
            f"; 1C={float(result_row['accuracy_1c']):.6f}, "
            f"ICLv2={float(result_row['accuracy_iclv2']):.6f}, "
            f"delta={float(result_row['delta_accuracy']) * 100:.3f} pp"
        )
    fig.suptitle(
        f"{dataset_name} train/test distribution by class\n"
        f"{sample_note}, features={payload['n_features_raw']}{delta_note}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_index(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        if row["status"] != "ok":
            continue
        img_name = Path(row["png"]).name
        cards.append(
            f'<a class="card" href="{html.escape(img_name)}">'
            f'<img src="{html.escape(img_name)}" loading="lazy">'
            f'<span>{html.escape(row["dataset_name"])}</span>'
            f'<small>delta={float(row["delta_accuracy"]) * 100:.3f} pp; '
            f'classes={row["n_classes"]}</small></a>'
        )
    failures = "".join(
        f"<li>{html.escape(row['dataset_name'])}: {html.escape(row['error'])}</li>"
        for row in rows
        if row["status"] != "ok"
    )
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TTT Decrease Train/Test Shift t-SNE</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; color: #111; text-decoration: none; }}
    .card img {{ width: 100%; display: block; }}
    .card span {{ display: block; margin-top: 6px; font-size: 13px; overflow-wrap: anywhere; }}
    .card small {{ display: block; color: #555; margin-top: 3px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>TTT Decrease Train/Test Shift t-SNE</h1>
  <p>Each class is plotted in a separate panel. Blue circles are train samples, orange circles are test samples, and gray points are other classes.</p>
  <div class="grid">{''.join(cards)}</div>
  {"<h2>Failed</h2><ul>" + failures + "</ul>" if failures else ""}
</body>
</html>
"""
    (out_dir / "index.html").write_text(content)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if args.only:
        manifest = manifest[manifest["dataset_name"].isin(args.only)].copy()
    manifest = manifest.sort_values("delta_accuracy")

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(manifest.to_dict("records"), start=1):
        dataset_name = row["dataset_name"]
        print(f"[{i}/{len(manifest)}] {dataset_name}", flush=True)
        out_path = args.out_dir / f"{dataset_name}_train_test_shift.png"
        try:
            payload = compute_embedding(args.data_root / dataset_name, args)
            plot_class_facets(
                dataset_name=dataset_name,
                result_row=pd.Series(row),
                payload=payload,
                out_path=out_path,
                dpi=args.dpi,
            )
            _, classes = label_codes(payload["y"])
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "status": "ok",
                    "png": str(out_path),
                    "n_classes": len(classes),
                    "accuracy_1c": row["accuracy_1c"],
                    "accuracy_iclv2": row["accuracy_iclv2"],
                    "delta_accuracy": row["delta_accuracy"],
                    "sampled_train": payload["sampled_train"],
                    "sampled_test": payload["sampled_test"],
                    "n_train": payload["n_train"],
                    "n_test": payload["n_test"],
                    "error": "",
                }
            )
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "status": "failed",
                    "png": "",
                    "n_classes": "",
                    "accuracy_1c": row.get("accuracy_1c", ""),
                    "accuracy_iclv2": row.get("accuracy_iclv2", ""),
                    "delta_accuracy": row.get("delta_accuracy", ""),
                    "sampled_train": "",
                    "sampled_test": "",
                    "n_train": "",
                    "n_test": "",
                    "error": repr(exc),
                }
            )

    with (args.out_dir / "train_test_shift_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset_name"])
        writer.writeheader()
        writer.writerows(rows)
    write_index(args.out_dir, rows)
    ok = sum(row["status"] == "ok" for row in rows)
    print(f"done: ok={ok}, failed={len(rows) - ok}, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()

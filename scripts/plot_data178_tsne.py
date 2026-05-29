#!/usr/bin/env python3
"""Generate train/test t-SNE plots for every dataset under data178."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw one t-SNE PNG per data178 dataset, comparing train and test splits."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data178"))
    parser.add_argument("--out-dir", type=Path, default=Path("data178_tsne_plots"))
    parser.add_argument("--max-samples-per-split", type=int, default=2000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--max-categories", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--only", nargs="*", default=None, help="Optional dataset names to plot.")
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args()


def load_optional_array(dataset_dir: Path, name: str) -> np.ndarray | None:
    path = dataset_dir / name
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def stratified_sample_indices(y: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n)

    labels = np.asarray(y)
    selected: list[np.ndarray] = []
    unique, counts = np.unique(labels, return_counts=True)
    remaining = max_samples
    allocations: dict[Any, int] = {}

    for label, count in zip(unique, counts):
        alloc = max(1, int(round(max_samples * count / n)))
        alloc = min(alloc, int(count))
        allocations[label] = alloc

    diff = max_samples - sum(allocations.values())
    if diff != 0:
        order = sorted(unique, key=lambda label: counts[np.where(unique == label)[0][0]], reverse=diff > 0)
        for label in order:
            if diff == 0:
                break
            count = int(counts[np.where(unique == label)[0][0]])
            if diff > 0 and allocations[label] < count:
                allocations[label] += 1
                diff -= 1
            elif diff < 0 and allocations[label] > 1:
                allocations[label] -= 1
                diff += 1

    for label in unique:
        label_idx = np.flatnonzero(labels == label)
        take = allocations[label]
        selected.append(rng.choice(label_idx, size=take, replace=False))

    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return np.sort(indices)


def clean_numeric(arr: np.ndarray | None) -> sparse.csr_matrix | None:
    if arr is None or arr.size == 0:
        return None
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x[~np.isfinite(x)] = np.nan
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    missing = np.isnan(x)
    if missing.any():
        x[missing] = np.take(med, np.where(missing)[1])
    x = StandardScaler().fit_transform(x)
    return sparse.csr_matrix(x)


def encode_categorical(arr: np.ndarray | None, max_categories: int) -> sparse.csr_matrix | None:
    if arr is None or arr.size == 0:
        return None
    c = np.asarray(arr, dtype=object)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    c = np.where(c == None, "__MISSING__", c)  # noqa: E711
    c = np.where(c == "", "__MISSING__", c).astype(str)
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        max_categories=max_categories,
        sparse_output=True,
    )
    return encoder.fit_transform(c)


def reduce_for_tsne(x: sparse.csr_matrix, pca_dims: int, random_state: int) -> np.ndarray:
    n_samples, n_features = x.shape
    if n_features == 0:
        raise ValueError("dataset has zero usable features")
    target_dims = min(pca_dims, n_samples - 1, n_features)
    if target_dims >= 2 and n_features > target_dims:
        reducer = TruncatedSVD(n_components=target_dims, random_state=random_state)
        return reducer.fit_transform(x)
    dense = x.toarray()
    if dense.shape[1] > 2 and min(dense.shape) > 2:
        n_components = min(target_dims, dense.shape[0] - 1, dense.shape[1])
        return PCA(n_components=n_components, random_state=random_state).fit_transform(dense)
    return dense


def label_codes(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    label_strings = np.asarray([str(x) for x in labels])
    unique = sorted(np.unique(label_strings), key=lambda x: (len(x), x))
    mapping = {label: i for i, label in enumerate(unique)}
    codes = np.asarray([mapping[x] for x in label_strings], dtype=int)
    return codes, unique


def plot_embedding(
    embedding: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    dataset_name: str,
    original_train: int,
    original_test: int,
    feature_count: int,
    sampled: bool,
    out_path: Path,
    dpi: int,
) -> None:
    codes, classes = label_codes(y)
    n_classes = len(classes)
    cmap_name = "tab20" if n_classes <= 20 else "gist_ncar"
    cmap = plt.get_cmap(cmap_name, max(n_classes, 1))

    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    for split_name, marker, size, alpha, linewidth in [
        ("train", "o", 16, 0.62, 0.0),
        ("test", "x", 22, 0.85, 0.8),
    ]:
        mask = split == split_name
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=codes[mask],
            cmap=cmap,
            vmin=-0.5,
            vmax=max(n_classes - 0.5, 0.5),
            s=size,
            marker=marker,
            alpha=alpha,
            linewidths=linewidth,
            label=split_name,
        )

    sample_note = "sampled" if sampled else "full"
    title = (
        f"{dataset_name}\n"
        f"t-SNE train/test ({sample_note}); train={original_train}, test={original_test}, features={feature_count}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linewidth=0.35, alpha=0.25)
    split_legend = ax.legend(title="split", loc="upper right", frameon=True, fontsize=8)
    ax.add_artist(split_legend)

    if n_classes <= 12:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=cmap(i),
                markeredgecolor="none",
                markersize=6,
                label=classes[i],
            )
            for i in range(n_classes)
        ]
        ax.legend(handles=handles, title="class", loc="lower right", frameon=True, fontsize=7)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_classes - 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"class id (n={n_classes})")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_dataset(dataset_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    stable_offset = int(hashlib.md5(dataset_dir.name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(args.random_state + stable_offset % 1_000_000)
    y_train = np.load(dataset_dir / "y_train.npy", allow_pickle=True)
    y_test = np.load(dataset_dir / "y_test.npy", allow_pickle=True)

    train_idx = stratified_sample_indices(y_train, args.max_samples_per_split, rng)
    test_idx = stratified_sample_indices(y_test, args.max_samples_per_split, rng)
    sampled = len(train_idx) < len(y_train) or len(test_idx) < len(y_test)

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

    out_path = args.out_dir / f"{dataset_dir.name}.png"
    plot_embedding(
        embedding=embedding,
        y=y,
        split=split,
        dataset_name=dataset_dir.name,
        original_train=len(y_train),
        original_test=len(y_test),
        feature_count=feature_count,
        sampled=sampled,
        out_path=out_path,
        dpi=args.dpi,
    )
    return {
        "dataset_name": dataset_dir.name,
        "status": "ok",
        "png": str(out_path),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "sampled_train": len(train_idx),
        "sampled_test": len(test_idx),
        "n_features_raw": feature_count,
        "n_features_encoded": x.shape[1],
        "perplexity": perplexity,
        "error": "",
    }


def write_index(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    cards = []
    for row in ok_rows:
        path = Path(row["png"]).name
        title = html.escape(row["dataset_name"])
        cards.append(
            f'<a class="card" href="{path}"><img src="{path}" loading="lazy"><span>{title}</span></a>'
        )
    failed = "".join(
        f"<li>{html.escape(row['dataset_name'])}: {html.escape(row['error'])}</li>"
        for row in rows
        if row["status"] != "ok"
    )
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>data178 t-SNE plots</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ color: #111; text-decoration: none; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }}
    .card img {{ width: 100%; display: block; }}
    .card span {{ display: block; margin-top: 6px; font-size: 13px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>data178 t-SNE plots</h1>
  <p>Generated PNGs compare train vs test splits for each dataset.</p>
  <div class="grid">{''.join(cards)}</div>
  {"<h2>Failed</h2><ul>" + failed + "</ul>" if failed else ""}
</body>
</html>
"""
    (out_dir / "index.html").write_text(content)


def write_contact_sheet(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return
    images = []
    for row in ok_rows:
        img = plt.imread(row["png"])
        images.append((row["dataset_name"], img))
    cols = 6
    rows_n = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.0, rows_n * 2.55), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (name, img) in zip(axes_arr, images):
        ax.imshow(img)
        ax.set_title(name, fontsize=7)
        ax.axis("off")
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.savefig(out_dir / "contact_sheet.png", dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only)
        datasets = [p for p in datasets if p.name in wanted]

    rows: list[dict[str, Any]] = []
    for i, dataset_dir in enumerate(datasets, start=1):
        print(f"[{i}/{len(datasets)}] {dataset_dir.name}", flush=True)
        try:
            row = plot_dataset(dataset_dir, args)
        except Exception as exc:  # Keep the all-dataset run moving.
            row = {
                "dataset_name": dataset_dir.name,
                "status": "failed",
                "png": "",
                "n_train": "",
                "n_test": "",
                "sampled_train": "",
                "sampled_test": "",
                "n_features_raw": "",
                "n_features_encoded": "",
                "perplexity": "",
                "error": repr(exc),
            }
            print(f"  failed: {exc}", flush=True)
        rows.append(row)

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset_name"])
        writer.writeheader()
        writer.writerows(rows)
    write_index(args.out_dir, rows)
    if not args.no_contact_sheet:
        write_contact_sheet(args.out_dir, rows)

    ok = sum(row["status"] == "ok" for row in rows)
    failed = len(rows) - ok
    print(f"done: ok={ok}, failed={failed}, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()

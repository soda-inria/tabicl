#!/usr/bin/env python
"""Visualize ICL parameter deltas between per-dataset TTT ckpts and a baseline."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


EPS = 1e-12


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format in {path}")


def parse_layer(key: str) -> str:
    match = re.search(r"\.blocks\.(\d+)\.", key)
    if match:
        return f"block_{int(match.group(1)):02d}"
    if ".decoder." in key:
        return "decoder"
    if ".y_encoder." in key:
        return "y_encoder"
    if ".ln." in key:
        return "ln"
    return "other"


def parse_group(key: str) -> str:
    if ".attn.in_proj_" in key:
        return "attn.in_proj"
    if ".attn.out_proj." in key:
        return "attn.out_proj"
    if ".linear1." in key:
        return "ffn.linear1"
    if ".linear2." in key:
        return "ffn.linear2"
    if ".norm1." in key:
        return "norm1"
    if ".norm2." in key:
        return "norm2"
    if ".decoder." in key:
        return "decoder"
    if ".y_encoder." in key:
        return "y_encoder"
    if ".ln." in key:
        return "ln"
    return "other"


def dataset_name(path: Path) -> str:
    name = path.stem
    return re.sub(r"_?step_?\d+$", "", name)


def tensor_metrics(dataset: str, key: str, base: torch.Tensor, current: torch.Tensor) -> dict[str, object]:
    if base.shape != current.shape:
        raise ValueError(f"Shape mismatch for {dataset}:{key}: {tuple(base.shape)} vs {tuple(current.shape)}")

    base_f = base.detach().cpu().float()
    current_f = current.detach().cpu().float()
    delta = current_f - base_f
    abs_delta = delta.abs()
    base_norm = torch.linalg.vector_norm(base_f).item()
    delta_norm = torch.linalg.vector_norm(delta).item()
    current_norm = torch.linalg.vector_norm(current_f).item()
    flat_base = base_f.reshape(-1)
    flat_current = current_f.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_base, flat_current, dim=0).item()

    return {
        "dataset": dataset,
        "key": key,
        "layer": parse_layer(key),
        "group": parse_group(key),
        "shape": "x".join(str(v) for v in base.shape),
        "numel": base.numel(),
        "base_norm": base_norm,
        "current_norm": current_norm,
        "delta_l2": delta_norm,
        "rel_l2": delta_norm / (base_norm + EPS),
        "mean_abs_delta": abs_delta.mean().item(),
        "median_abs_delta": abs_delta.median().item(),
        "max_abs_delta": abs_delta.max().item(),
        "rms_delta": math.sqrt(torch.mean(delta * delta).item()),
        "changed_fraction_exact": (delta != 0).float().mean().item(),
        "changed_fraction_gt_1e_12": (abs_delta > 1e-12).float().mean().item(),
        "cosine_similarity": cosine,
    }


def make_tensor_summary(
    baseline: dict[str, torch.Tensor],
    ckpt_paths: list[Path],
    prefix: str,
) -> pd.DataFrame:
    keys = [key for key, value in baseline.items() if key.startswith(prefix) and torch.is_tensor(value)]
    if not keys:
        raise ValueError(f"No tensor keys found with prefix {prefix!r}")

    rows: list[dict[str, object]] = []
    base_key_set = set(keys)
    for path in ckpt_paths:
        name = dataset_name(path)
        state_dict = load_state_dict(path)
        missing = sorted(base_key_set - set(state_dict))
        if missing:
            raise KeyError(f"{path} is missing {len(missing)} baseline ICL keys, first: {missing[:3]}")
        for key in keys:
            rows.append(tensor_metrics(name, key, baseline[key], state_dict[key]))
    return pd.DataFrame(rows)


def save_layer_summary(tensor_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    weighted = tensor_df.copy()
    for col in ["mean_abs_delta", "rms_delta", "rel_l2", "changed_fraction_gt_1e_12"]:
        weighted[f"{col}_weighted"] = weighted[col] * weighted["numel"]

    grouped = (
        weighted.groupby(["dataset", "layer"], as_index=False)
        .agg(
            num_tensors=("key", "count"),
            numel=("numel", "sum"),
            delta_l2=("delta_l2", lambda values: float(np.sqrt(np.sum(np.square(values))))),
            mean_abs_delta_weighted=("mean_abs_delta_weighted", "sum"),
            rms_delta_weighted=("rms_delta_weighted", "sum"),
            rel_l2_weighted=("rel_l2_weighted", "sum"),
            changed_fraction_gt_1e_12_weighted=("changed_fraction_gt_1e_12_weighted", "sum"),
            max_abs_delta=("max_abs_delta", "max"),
        )
        .sort_values(["dataset", "layer"])
    )
    for col in ["mean_abs_delta", "rms_delta", "rel_l2", "changed_fraction_gt_1e_12"]:
        grouped[col] = grouped[f"{col}_weighted"] / grouped["numel"]
        grouped = grouped.drop(columns=[f"{col}_weighted"])
    grouped.to_csv(output_dir / "icl_layer_diff_summary.csv", index=False)
    return grouped


def save_group_summary(tensor_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    grouped = (
        tensor_df.groupby(["dataset", "layer", "group"], as_index=False)
        .agg(
            num_tensors=("key", "count"),
            numel=("numel", "sum"),
            mean_abs_delta=("mean_abs_delta", "mean"),
            rel_l2=("rel_l2", "mean"),
            max_abs_delta=("max_abs_delta", "max"),
            changed_fraction_gt_1e_12=("changed_fraction_gt_1e_12", "mean"),
        )
        .sort_values(["dataset", "layer", "group"])
    )
    grouped.to_csv(output_dir / "icl_layer_group_diff_summary.csv", index=False)
    return grouped


def save_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, title: str, path: Path, figsize: tuple[float, float]):
    pivot = df.pivot(index=index, columns=columns, values=values)
    order = sorted(pivot.index, key=lambda x: (not str(x).startswith("block_"), str(x)))
    pivot = pivot.loc[order]

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, cmap="magma", linewidths=0.25, linecolor="white", cbar_kws={"label": values})
    plt.title(title)
    plt.xlabel(columns)
    plt.ylabel(index)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_top_barplots(tensor_df: pd.DataFrame, output_dir: Path, top_n: int):
    datasets = list(tensor_df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, max(4, 4.8 * len(datasets))), squeeze=False)
    for axis, dataset in zip(axes[:, 0], datasets):
        top = tensor_df[tensor_df["dataset"] == dataset].nlargest(top_n, "rel_l2").iloc[::-1]
        labels = [key.replace("icl_predictor.", "") for key in top["key"]]
        axis.barh(labels, top["rel_l2"], color="#2f6f73")
        axis.set_title(f"{dataset}: top {top_n} ICL tensors by relative L2 delta")
        axis.set_xlabel("relative L2 delta")
        axis.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "icl_top_tensor_rel_l2_barplots.png", dpi=220)
    plt.close()


def downsample_abs_delta(base: torch.Tensor, current: torch.Tensor, max_side: int = 96) -> np.ndarray:
    delta = (current.detach().cpu().float() - base.detach().cpu().float()).abs()
    if delta.ndim == 1:
        return delta.numpy()[None, :]
    if delta.ndim > 2:
        delta = delta.reshape(delta.shape[0], -1)
    array = delta.numpy()
    row_step = max(1, math.ceil(array.shape[0] / max_side))
    col_step = max(1, math.ceil(array.shape[1] / max_side))
    return array[::row_step, ::col_step]


def save_top_tensor_delta_heatmaps(
    baseline: dict[str, torch.Tensor],
    ckpt_paths: list[Path],
    tensor_df: pd.DataFrame,
    output_dir: Path,
    top_n: int,
):
    for path in ckpt_paths:
        name = dataset_name(path)
        state_dict = load_state_dict(path)
        top_keys = (
            tensor_df[(tensor_df["dataset"] == name) & tensor_df["shape"].str.contains("x", regex=False)]
            .nlargest(top_n, "rel_l2")["key"]
            .tolist()
        )
        if not top_keys:
            continue
        fig, axes = plt.subplots(1, len(top_keys), figsize=(5 * len(top_keys), 4), squeeze=False)
        for axis, key in zip(axes[0], top_keys):
            image = downsample_abs_delta(baseline[key], state_dict[key])
            sns.heatmap(image, ax=axis, cmap="viridis", cbar=True)
            axis.set_title(key.replace("icl_predictor.", ""), fontsize=8)
            axis.set_xlabel("downsampled columns")
            axis.set_ylabel("downsampled rows")
        fig.suptitle(f"{name}: top {top_n} absolute delta maps", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f"icl_top_tensor_delta_maps_{name}.png", dpi=220, bbox_inches="tight")
        plt.close()


def save_overview(tensor_df: pd.DataFrame, layer_df: pd.DataFrame, output_dir: Path):
    summary = (
        tensor_df.groupby("dataset", as_index=False)
        .agg(
            num_icl_tensors=("key", "count"),
            numel=("numel", "sum"),
            mean_rel_l2=("rel_l2", "mean"),
            max_rel_l2=("rel_l2", "max"),
            mean_abs_delta=("mean_abs_delta", "mean"),
            max_abs_delta=("max_abs_delta", "max"),
            mean_changed_fraction_gt_1e_12=("changed_fraction_gt_1e_12", "mean"),
            max_changed_fraction_gt_1e_12=("changed_fraction_gt_1e_12", "max"),
            min_cosine_similarity=("cosine_similarity", "min"),
        )
        .sort_values("dataset")
    )
    top_rows = tensor_df.sort_values(["dataset", "rel_l2"], ascending=[True, False]).groupby("dataset").head(10)

    lines = [
        "# ICL checkpoint difference summary",
        "",
        "Baseline: `tabicl-classifier-v1.1-20250506.ckpt`",
        "Compared tensors: `icl_predictor.*`",
        "",
        "## Dataset overview",
        "",
        summary.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Top changed tensors by dataset",
        "",
        top_rows[
            [
                "dataset",
                "key",
                "layer",
                "group",
                "shape",
                "rel_l2",
                "mean_abs_delta",
                "max_abs_delta",
                "changed_fraction_gt_1e_12",
                "cosine_similarity",
            ]
        ].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Largest layer deltas",
        "",
        layer_df.sort_values(["dataset", "rel_l2"], ascending=[True, False])
        .groupby("dataset")
        .head(5)[["dataset", "layer", "num_tensors", "numel", "rel_l2", "mean_abs_delta", "max_abs_delta"]]
        .to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    (output_dir / "icl_diff_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("tabicl-classifier-v1.1-20250506.ckpt"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("1b_result_CKPT/ensemble_ttt_step10_ckpt"))
    parser.add_argument("--prefix", default="icl_predictor.")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    output_dir = args.ckpt_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_paths = sorted(path for path in args.ckpt_dir.glob("*.ckpt") if path.name != args.baseline.name)
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found in {args.ckpt_dir}")

    baseline = load_state_dict(args.baseline)
    tensor_df = make_tensor_summary(baseline, ckpt_paths, args.prefix)
    tensor_df.to_csv(output_dir / "icl_tensor_diff_summary.csv", index=False)
    changed_df = tensor_df[tensor_df["changed_fraction_gt_1e_12"] > 0].copy()
    changed_df.to_csv(output_dir / "icl_changed_tensor_diff_summary.csv", index=False)

    layer_df = save_layer_summary(tensor_df, output_dir)
    save_group_summary(tensor_df, output_dir)

    save_heatmap(
        layer_df,
        index="layer",
        columns="dataset",
        values="rel_l2",
        title="ICL layer relative L2 delta vs baseline",
        path=output_dir / "icl_layer_rel_l2_heatmap.png",
        figsize=(7, 7),
    )
    save_heatmap(
        layer_df,
        index="layer",
        columns="dataset",
        values="mean_abs_delta",
        title="ICL layer mean absolute delta vs baseline",
        path=output_dir / "icl_layer_mean_abs_delta_heatmap.png",
        figsize=(7, 7),
    )

    tensor_heatmap = tensor_df.copy()
    tensor_heatmap["short_key"] = tensor_heatmap["key"].str.replace("icl_predictor.", "", regex=False)
    save_heatmap(
        tensor_heatmap,
        index="short_key",
        columns="dataset",
        values="rel_l2",
        title="ICL tensor relative L2 delta vs baseline",
        path=output_dir / "icl_tensor_rel_l2_heatmap.png",
        figsize=(8, 28),
    )

    save_top_barplots(tensor_df, output_dir, args.top_n)
    save_top_tensor_delta_heatmaps(baseline, ckpt_paths, tensor_df, output_dir, top_n=min(4, args.top_n))
    save_overview(tensor_df, layer_df, output_dir)

    print(f"Wrote ICL diff visualizations and summaries to {output_dir}")
    print(f"Datasets: {', '.join(dataset_name(path) for path in ckpt_paths)}")
    print(f"ICL tensors compared: {tensor_df['key'].nunique()}")
    print(f"Changed tensor rows: {len(changed_df)} / {len(tensor_df)}")


if __name__ == "__main__":
    main()

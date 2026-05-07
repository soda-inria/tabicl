from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full training with ICL training on shared successful datasets."
    )
    parser.add_argument("--full-root", type=Path, default=Path("result/1a_result_all"))
    parser.add_argument("--step-root", type=Path, default=Path("result/ttt_step_result"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("result/1a_result_all/compare_ttt_step_result"),
    )
    return parser.parse_args()


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return "\n".join(lines)


def fmt_pp(value: float) -> str:
    return f"{value:+.4f} pp"


def discover_steps(full_root: Path, step_root: Path) -> list[int]:
    full_steps: set[int] = set()
    for path in full_root.rglob("all_classification_results.csv"):
        match = re.search(r"ttt_step(\d+)", str(path.parent))
        if match:
            full_steps.add(int(match.group(1)))
    step_steps: set[int] = set()
    for path in step_root.rglob("all_classification_results.csv"):
        match = re.search(r"ttt_step(\d+)", str(path.parent))
        if match:
            step_steps.add(int(match.group(1)))
    return sorted(full_steps & step_steps)


def build_dataframes(
    full_root: Path,
    step_root: Path,
    steps: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], set[str], dict[int, set[str]]]:
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    extra_failure_intersection: set[str] | None = None
    shared_sets: dict[int, set[str]] = {}
    raw_frames: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for step in steps:
        full_csv = full_root / f"iclv1.1_ttt_step{step}" / "all_classification_results.csv"
        step_csv = step_root / f"iclv1.1_ttt_step{step}" / "all_classification_results.csv"

        full_df = pd.read_csv(full_csv)
        step_df = pd.read_csv(step_csv)
        raw_frames[step] = (full_df, step_df)
        shared_sets[step] = set(full_df.loc[full_df["status"] == "ok", "dataset_name"]) & set(
            step_df.loc[step_df["status"] == "ok", "dataset_name"]
        )

    reference_steps = [step for step in steps if step >= 10]
    if not reference_steps:
        reference_steps = steps
    reference_shared_set = set.intersection(*(shared_sets[step] for step in reference_steps))

    for step in steps:
        full_df, step_df = raw_frames[step]

        full_ok = full_df.loc[full_df["status"] == "ok", ["dataset_name", "accuracy", "task_type"]].copy()
        full_ok = full_ok.rename(columns={"accuracy": "acc_full"})
        step_ok = step_df.loc[step_df["status"] == "ok", ["dataset_name", "accuracy"]].copy()
        step_ok = step_ok.rename(columns={"accuracy": "acc_step"})

        merged = full_ok.merge(step_ok, on="dataset_name", how="inner")
        merged = merged.loc[merged["dataset_name"].isin(reference_shared_set)].copy()
        merged["step"] = step
        merged["delta"] = merged["acc_full"] - merged["acc_step"]
        merged["delta_pp"] = merged["delta"] * 100.0
        merged["winner"] = np.where(
            merged["delta"] > 1e-12,
            "full_training",
            np.where(merged["delta"] < -1e-12, "icl_training", "tie"),
        )
        detail_frames.append(merged)

        full_fail = set(full_df.loc[full_df["status"] != "ok", "dataset_name"].tolist())
        step_fail = set(step_df.loc[step_df["status"] != "ok", "dataset_name"].tolist())
        extra_full_fail = sorted(full_fail - step_fail)
        if extra_failure_intersection is None:
            extra_failure_intersection = set(extra_full_fail)
        else:
            extra_failure_intersection &= set(extra_full_fail)

        summary_rows.append(
            {
                "step": step,
                "full_ok": int((full_df["status"] == "ok").sum()),
                "step_ok": int((step_df["status"] == "ok").sum()),
                "common_ok": int(len(merged)),
                "raw_common_ok": int(len(shared_sets[step])),
                "full_avg_common": float(merged["acc_full"].mean()),
                "step_avg_common": float(merged["acc_step"].mean()),
                "mean_gap_pp": float(merged["delta_pp"].mean()),
                "median_gap_pp": float(merged["delta_pp"].median()),
                "full_better": int((merged["delta"] > 1e-12).sum()),
                "step_better": int((merged["delta"] < -1e-12).sum()),
                "tie": int((merged["delta"].abs() <= 1e-12).sum()),
                "abs_gap_ge_0_1pp": int((merged["delta_pp"].abs() >= 0.1).sum()),
                "abs_gap_ge_0_5pp": int((merged["delta_pp"].abs() >= 0.5).sum()),
                "extra_full_failures": int(len(extra_full_fail)),
                "shared_failures": int(len(full_fail & step_fail)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("step").reset_index(drop=True)
    detail_df = pd.concat(detail_frames, ignore_index=True)
    stable_extra_failures = sorted(extra_failure_intersection) if extra_failure_intersection else []
    return summary_df, detail_df, stable_extra_failures, reference_shared_set, shared_sets


def build_plots(summary_df: pd.DataFrame, detail_df: pd.DataFrame, out_dir: Path) -> None:
    x_positions = np.arange(len(summary_df))
    steps = [str(int(step)) for step in summary_df["step"]]

    color_full = "#c44e52"
    color_step = "#4c72b0"
    color_tie = "#9aa0a6"

    full_values = summary_df["full_avg_common"].to_numpy()
    icl_values = summary_df["step_avg_common"].to_numpy()
    gap_values = summary_df["mean_gap_pp"].to_numpy()

    fig, ax_acc = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    ax_acc.set_facecolor("#fcfcfd")

    ax_acc.plot(
        x_positions,
        full_values,
        color=color_full,
        marker="o",
        markersize=8,
        linewidth=3.0,
        zorder=3,
    )
    ax_acc.plot(
        x_positions,
        icl_values,
        color=color_step,
        marker="s",
        markersize=8,
        linewidth=3.0,
        zorder=3,
    )

    full_better_mask = full_values >= icl_values
    ax_acc.fill_between(
        x_positions,
        full_values,
        icl_values,
        where=full_better_mask,
        interpolate=True,
        color=color_full,
        alpha=0.10,
        zorder=1,
    )
    ax_acc.fill_between(
        x_positions,
        full_values,
        icl_values,
        where=~full_better_mask,
        interpolate=True,
        color=color_step,
        alpha=0.12,
        zorder=1,
    )

    for idx, (full_v, icl_v, gap_v) in enumerate(zip(full_values, icl_values, gap_values)):
        mid_y = (full_v + icl_v) / 2.0
        ax_acc.text(
            idx,
            mid_y + 0.00018,
            fmt_pp(gap_v),
            ha="center",
            va="center",
            fontsize=10,
            color="#2f2f2f",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#d9dce3",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=4,
        )

    best_full_idx = int(np.argmax(full_values))
    best_icl_idx = int(np.argmax(icl_values))
    ax_acc.scatter(
        [best_full_idx],
        [full_values[best_full_idx]],
        s=170,
        facecolor="white",
        edgecolor=color_full,
        linewidth=2.0,
        zorder=5,
    )
    ax_acc.scatter(
        [best_icl_idx],
        [icl_values[best_icl_idx]],
        s=170,
        facecolor="white",
        edgecolor=color_step,
        linewidth=2.0,
        zorder=5,
    )

    ax_acc.annotate(
        "Full training",
        xy=(x_positions[-1], full_values[-1]),
        xytext=(10, -2),
        textcoords="offset points",
        color=color_full,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax_acc.annotate(
        "ICL training",
        xy=(x_positions[-1], icl_values[-1]),
        xytext=(10, 2),
        textcoords="offset points",
        color=color_step,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
    )

    ax_acc.text(
        0.015,
        0.97,
        "Reference set: 150 shared successful datasets",
        transform=ax_acc.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#5b6270",
    )

    y_min = min(full_values.min(), icl_values.min())
    y_max = max(full_values.max(), icl_values.max())
    margin = max((y_max - y_min) * 0.24, 0.00018)
    ax_acc.set_ylim(y_min - margin, y_max + margin)
    ax_acc.set_title("Shared-Set Mean Accuracy", fontsize=20, pad=14)
    ax_acc.set_xticks(x_positions)
    ax_acc.set_xticklabels(steps, fontsize=11)
    ax_acc.set_xlabel("TTT Steps", fontsize=13)
    ax_acc.set_ylabel("Accuracy", fontsize=13)
    ax_acc.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax_acc.grid(True, linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.28)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    fig.savefig(out_dir / "shared_mean_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax_win = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    ax_win.bar(
        x_positions,
        summary_df["full_better"],
        color=color_full,
        alpha=0.88,
        label="Full training wins",
    )
    ax_win.bar(
        x_positions,
        summary_df["tie"],
        bottom=summary_df["full_better"],
        color=color_tie,
        alpha=0.95,
        label="Tie",
    )
    ax_win.bar(
        x_positions,
        summary_df["step_better"],
        bottom=summary_df["full_better"] + summary_df["tie"],
        color=color_step,
        alpha=0.88,
        label="ICL training wins",
    )
    ax_win.set_title("Win / Tie / Loss on Shared Datasets")
    ax_win.set_xticks(x_positions)
    ax_win.set_xticklabels(steps)
    ax_win.set_xlabel("TTT Steps")
    ax_win.set_ylabel("Dataset Count")
    ax_win.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_win.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(out_dir / "win_tie_loss.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    width = 0.34
    fig, ax_cov = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    ax_cov.bar(
        x_positions - width / 2,
        summary_df["common_ok"],
        width=width,
        color="#55a868",
        alpha=0.85,
        label="Shared successful datasets",
    )
    ax_cov.bar(
        x_positions + width / 2,
        summary_df["extra_full_failures"],
        width=width,
        color="#dd8452",
        alpha=0.85,
        label="Extra full-training failures",
    )
    for idx, row in summary_df.iterrows():
        ax_cov.text(idx - width / 2, row["common_ok"] + 1.0, str(int(row["common_ok"])), ha="center", fontsize=9)
        ax_cov.text(
            idx + width / 2,
            row["extra_full_failures"] + 0.8,
            str(int(row["extra_full_failures"])),
            ha="center",
            fontsize=9,
        )
    ax_cov.set_title("Coverage Gap")
    ax_cov.set_xticks(x_positions)
    ax_cov.set_xticklabels(steps)
    ax_cov.set_xlabel("TTT Steps")
    ax_cov.set_ylabel("Dataset Count")
    ax_cov.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_cov.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(out_dir / "coverage_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    box_data = [
        detail_df.loc[detail_df["step"] == step, "delta_pp"].tolist()
        for step in summary_df["step"].tolist()
    ]
    fig, ax_box = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    ax_box.boxplot(
        box_data,
        positions=x_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "#f6d7d9", "edgecolor": color_full, "linewidth": 1.4},
        medianprops={"color": "#222222", "linewidth": 1.8},
        whiskerprops={"color": color_full},
        capprops={"color": color_full},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "#f2a2a7",
            "markeredgecolor": color_full,
            "alpha": 0.65,
        },
    )
    ax_box.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2, alpha=0.85)
    ax_box.plot(
        x_positions,
        summary_df["mean_gap_pp"],
        color="#1f1f1f",
        marker="D",
        linewidth=2.0,
        label="Mean gap (full - step)",
    )
    ax_box.set_title("Per-Dataset Accuracy Gap Distribution")
    ax_box.set_xticks(x_positions)
    ax_box.set_xticklabels(steps)
    ax_box.set_xlabel("TTT Steps")
    ax_box.set_ylabel("Accuracy Gap (pp)")
    ax_box.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_box.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(out_dir / "accuracy_gap_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_summary(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    stable_extra_failures: list[str],
    out_dir: Path,
    reference_shared_set: set[str],
    shared_sets: dict[int, set[str]],
) -> str:
    best_full = summary_df.loc[summary_df["mean_gap_pp"].idxmax()]
    best_step = summary_df.loc[summary_df["mean_gap_pp"].idxmin()]
    largest_shared = summary_df.loc[summary_df["common_ok"].idxmax()]
    step1_excluded = sorted(shared_sets.get(1, set()) - reference_shared_set)

    step_rows = []
    for _, row in summary_df.iterrows():
        step_rows.append(
            [
                int(row["step"]),
                int(row["common_ok"]),
                f"{row['full_avg_common']:.6f}",
                f"{row['step_avg_common']:.6f}",
                fmt_pp(row["mean_gap_pp"]),
                fmt_pp(row["median_gap_pp"]),
                int(row["full_better"]),
                int(row["step_better"]),
                int(row["tie"]),
                int(row["extra_full_failures"]),
            ]
        )

    top_full_rows = []
    for step in summary_df["step"].tolist():
        step_detail = detail_df.loc[detail_df["step"] == step].sort_values("delta_pp", ascending=False).head(3)
        for _, row in step_detail.iterrows():
            top_full_rows.append(
                [
                    int(step),
                    row["dataset_name"],
                    row["task_type"],
                    f"{row['acc_full']:.6f}",
                    f"{row['acc_step']:.6f}",
                    fmt_pp(row["delta_pp"]),
                ]
            )

    top_step_rows = []
    for step in summary_df["step"].tolist():
        step_detail = detail_df.loc[detail_df["step"] == step].sort_values("delta_pp", ascending=True).head(3)
        for _, row in step_detail.iterrows():
            top_step_rows.append(
                [
                    int(step),
                    row["dataset_name"],
                    row["task_type"],
                    f"{row['acc_full']:.6f}",
                    f"{row['acc_step']:.6f}",
                    fmt_pp(row["delta_pp"]),
                ]
            )

    stable_fail_text = ", ".join(stable_extra_failures) if stable_extra_failures else "(none)"

    lines = [
        "# 1a_result_all vs ICL training",
        "",
        "## 输入",
        "- full training root: `result/1a_result_all`",
        "- ICL training root: `result/ttt_step_result`",
        f"- 使用固定 reference set 比较: `step10/20/30/40/50` 共同成功交集，共 `{len(reference_shared_set)}` 个数据集。",
        "- step1 也强制使用这组 reference set，以避免因样本集合不同带来偏差。",
        f"- 产物目录: `{out_dir}`",
        "",
        "## 主要结论",
        f"- 在公平的共同成功数据集上，两者平均 accuracy 非常接近。`1a_result_all` 最有利的 step 是 `{int(best_full['step'])}`，均值领先 `{fmt_pp(best_full['mean_gap_pp'])}`；`ICL training` 最有利的 step 是 `{int(best_step['step'])}`，均值领先 `{fmt_pp(best_step['mean_gap_pp'])}`。",
        f"- 真正更稳定的是 `ICL training`：它每个 step 都有 `{int(largest_shared['common_ok'])}` 或 `{int(summary_df['common_ok'].min())}` 个共同成功数据集，而 `1a_result_all` 持续多出 `20-22` 个额外失败数据集。",
        f"- 高 step 时胜负天平转向 `ICL training`。在 step40/50，共同数据集上 `ICL training` 分别赢 `{int(summary_df.loc[summary_df['step'] == 40, 'step_better'].iloc[0])}` / `{int(summary_df.loc[summary_df['step'] == 50, 'step_better'].iloc[0])}` 个数据集，而 `1a_result_all` 只赢 `{int(summary_df.loc[summary_df['step'] == 40, 'full_better'].iloc[0])}` / `{int(summary_df.loc[summary_df['step'] == 50, 'full_better'].iloc[0])}` 个。",
        f"- step1 原始共同成功集有 `{len(shared_sets.get(1, set()))}` 个，但为与后续 step 保持同一基准，额外排除了 `{len(step1_excluded)}` 个 step1-only 数据集: `{', '.join(step1_excluded) if step1_excluded else '(none)'}`。",
        f"- `1a_result_all` 的额外失败集高度稳定，几乎固定在同一批大表/高负载数据集上: `{stable_fail_text}`。",
        "",
        "## Step 汇总",
        markdown_table(
            [
                "step",
                "common_ok",
                "full_avg",
                "step_avg",
                "mean_gap",
                "median_gap",
                "full_win",
                "step_win",
                "tie",
                "extra_full_fail",
            ],
            step_rows,
        ),
        "",
        "## 每个 Step 中 full training 优势最大的样本",
        markdown_table(
            ["step", "dataset", "task", "full_acc", "step_acc", "gap"],
            top_full_rows,
        ),
        "",
        "## 每个 Step 中 ICL training 优势最大的样本",
        markdown_table(
            ["step", "dataset", "task", "full_acc", "step_acc", "gap"],
            top_step_rows,
        ),
        "",
        "## 图像说明",
        "- `shared_mean_accuracy.png`: 共同成功数据集上的均值 accuracy 曲线，标注的是 `full - step` 的均值差。",
        "- `win_tie_loss.png`: 每个 step 上 full training / ICL training / tie 的数据集计数。",
        "- `coverage_gap.png`: 共同成功数据集数量，以及 full training 相比 ICL training 的额外失败数量。",
        "- `accuracy_gap_distribution.png`: 逐 dataset 的 accuracy gap 分布；正值代表 full training 更好，负值代表 ICL training 更好。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    steps = discover_steps(args.full_root, args.step_root)
    if not steps:
        raise FileNotFoundError("No overlapping ttt_step runs were found.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, detail_df, stable_extra_failures, reference_shared_set, shared_sets = build_dataframes(
        args.full_root,
        args.step_root,
        steps,
    )
    summary_df.to_csv(out_dir / "step_comparison_summary.csv", index=False)
    detail_df.to_csv(out_dir / "per_dataset_delta.csv", index=False)

    build_plots(summary_df, detail_df, out_dir)
    summary_text = generate_summary(
        summary_df,
        detail_df,
        stable_extra_failures,
        out_dir,
        reference_shared_set,
        shared_sets,
    )
    (out_dir / "summary.md").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()

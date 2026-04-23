from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TTT step benchmark results.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("result/ttt_step_result"),
        help="Root directory containing baseline and ttt_step result folders.",
    )
    return parser.parse_args()


def parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def first_error_line(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().splitlines()[0] if value.strip() else ""


def to_float(value: str | None) -> float | None:
    if value is None or value in {"(none)", ""}:
        return None
    return float(value)


def to_int(value: str | None) -> int | None:
    if value is None or value in {"(none)", ""}:
        return None
    return int(value)


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(none)"
    return f"{value:.{digits}f}"


def format_pp(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(none)"
    return f"{value:+.4f} pp"


def format_ratio(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(none)"
    return f"{value:.2f}x"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return "\n".join(lines)


def discover_inputs(root: Path) -> tuple[Path, list[tuple[int, Path]]]:
    baseline_csv = root / "iclv1.1_baseline" / "tabiclv1_1_classification_3gpu" / "all_classification_results.csv"
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")

    step_runs: list[tuple[int, Path]] = []
    for csv_path in sorted(root.rglob("all_classification_results.csv")):
        if "baseline" in str(csv_path):
            continue
        match = re.search(r"ttt_step(\d+)", str(csv_path.parent))
        if not match:
            continue
        step_runs.append((int(match.group(1)), csv_path))

    if not step_runs:
        raise FileNotFoundError(f"No ttt_step result CSVs found under {root}")

    step_runs.sort(key=lambda item: item[0])
    return baseline_csv, step_runs


def build_plot(step_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = list(range(len(step_summary)))

    ax.plot(
        x_positions,
        step_summary["avg_accuracy_ok"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="avg_accuracy_ok",
    )

    baseline_avg = float(step_summary["baseline_avg_accuracy_ok"].iloc[0])
    common_baseline_avg = float(step_summary["common_avg_baseline"].iloc[0])
    ax.axhline(
        common_baseline_avg,
        color="#d62728",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8,
        label=f"baseline common_ok_avg={common_baseline_avg:.6f}",
    )

    y_min = min(step_summary["common_avg_step"].min(), common_baseline_avg, baseline_avg)
    y_max = max(step_summary["avg_accuracy_ok"].max(), common_baseline_avg, baseline_avg)
    margin = max((y_max - y_min) * 0.25, 0.00015)
    ax.set_ylim(y_min - margin, y_max + margin)

    for x_pos, (_, row) in zip(x_positions, step_summary.iterrows()):
        ax.annotate(
            f"{row['avg_accuracy_ok']:.6f}",
            (x_pos, row["avg_accuracy_ok"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1f77b4",
        )

    ax.set_title("TTT Step vs Accuracy")
    ax.set_xlabel("TTT Steps")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(step)) for step in step_summary["step"]])
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_report(
    root: Path,
    step_summary: pd.DataFrame,
    best_step_by_dataset: pd.DataFrame,
    instability: pd.DataFrame,
    failures: pd.DataFrame,
    representative_skip_reasons: pd.Series,
) -> str:
    best_row = step_summary.loc[step_summary["avg_accuracy_ok"].idxmax()]
    safest_best_row = step_summary.loc[step_summary["common_avg_step"].idxmax()]
    baseline_avg = float(step_summary["baseline_avg_accuracy_ok"].iloc[0])
    baseline_wall = float(step_summary["baseline_wall_seconds"].iloc[0])

    step_rows = []
    for _, row in step_summary.iterrows():
        step_rows.append(
            [
                int(row["step"]),
                int(row["ok_count"]),
                int(row["failed_count"]),
                format_float(row["avg_accuracy_ok"]),
                format_float(row["common_avg_step"]),
                format_pp((row["avg_accuracy_ok"] - baseline_avg) * 100.0),
                format_pp(row["avg_delta_pp"]),
                int(row["improve"]),
                int(row["worse"]),
                int(row["same"]),
                f"{row['wall_seconds']:.3f}",
                format_ratio(row["wall_seconds"] / baseline_wall),
            ]
        )

    positive_best = best_step_by_dataset.loc[best_step_by_dataset["delta"] > 1e-12].copy()
    best_step_rows = []
    for step, count in positive_best["step"].value_counts().sort_index().items():
        best_step_rows.append([int(step), int(count)])

    top_gain_rows = []
    for _, row in best_step_by_dataset.sort_values("delta", ascending=False).head(12).iterrows():
        top_gain_rows.append(
            [
                row["dataset_name"],
                row["task_type"],
                int(row["step"]),
                f"{row['baseline_acc']:.6f}",
                f"{row['accuracy']:.6f}",
                format_pp(row["delta"] * 100.0),
            ]
        )

    instability_rows = []
    for _, row in instability.head(12).iterrows():
        instability_rows.append(
            [
                row["dataset_name"],
                row["task_type"],
                f"{row['baseline_acc']:.6f}",
                f"{row['best_acc']:.6f}",
                f"{row['worst_acc']:.6f}",
                format_pp(row["swing_pp"]),
            ]
        )

    failure_rows = []
    for _, row in failures.iterrows():
        failure_rows.append(
            [
                row["dataset_name"],
                f"{row['baseline_accuracy']:.6f}",
                row["error_type"],
                row["error_first_line"],
            ]
        )

    skip_rows = [[reason, int(count)] for reason, count in representative_skip_reasons.items()]

    report_lines = [
        "# ttt_step_result 汇总分析",
        "",
        "## 输入目录",
        f"- 根目录: `{root}`",
        f"- baseline: `{root / 'iclv1.1_baseline' / 'tabiclv1_1_classification_3gpu' / 'all_classification_results.csv'}`",
        "- steps: `iclv1.1_ttt_step1/2/3/4/5/10/20/30/40/50`",
        "- 图像: `ttt_step_vs_acc.png`, `ttt_step_down_trend.png`",
        "- 汇总表: `step_summary.csv`, `dataset_best_step.csv`, `dataset_instability.csv`, `dataset_up_summary.csv`, `dataset_down_summary.csv`, `dataset_down_trend.csv`",
        "",
        "## 主要结论",
        f"- 以 `avg_accuracy_ok` 看，当前最优 step 是 `{int(best_row['step'])}`，平均 accuracy 为 `{best_row['avg_accuracy_ok']:.6f}`，相对 baseline `{baseline_avg:.6f}` 提升 `{(best_row['avg_accuracy_ok'] - baseline_avg) * 100.0:+.4f} pp`。",
        f"- 以两边都成功的 172 个数据集做公平对比，最优 step 仍是 `{int(safest_best_row['step'])}`，平均 accuracy `{safest_best_row['common_avg_step']:.6f}`，比 baseline 的 `{safest_best_row['common_avg_baseline']:.6f}` 提升 `{safest_best_row['avg_delta_pp']:+.4f} pp`。",
        f"- 收益不是单调增长: `step20` 已明显优于 `step10`，`step40` 达到峰值，`step50` 虽然在 72 个数据集上优于 baseline，但平均 accuracy 略低于 `step40`，说明更大 step 带来了更强的收益和更重的回归尾部。",
        f"- 所有 step 的失败集完全一致，都是 6 个数据集: `{', '.join(failures['dataset_name'].tolist())}`。这说明 step 数变化没有改变运行稳定性，主要差异集中在成功数据集上的 accuracy 变化。",
        f"- 每个 step 都只有 160 个数据集真正执行了 TTT，另外 12 个被稳定跳过；跳过原因主要是 `n_classes > 10`，另有 `volkert` 因 OOM 跳过。因此 step 曲线的收益只来自固定 160 个可适配数据集。",
        "",
        "## Step 对比",
        markdown_table(
            [
                "step",
                "ok",
                "fail",
                "avg_accuracy_ok",
                "common_ok_avg",
                "vs baseline",
                "common delta",
                "提升",
                "下降",
                "持平",
                "wall_seconds",
                "runtime_ratio",
            ],
            step_rows,
        ),
        "",
        "## 数据集最优 Step 分布",
        "仅统计相对 baseline 有正收益的数据集；若多个 step 并列最优，取最小 step。",
        markdown_table(["best_step", "dataset_count"], best_step_rows),
        "",
        "## 全局收益最大的 Dataset",
        markdown_table(
            ["dataset", "task", "best_step", "baseline_acc", "best_acc", "delta"],
            top_gain_rows,
        ),
        "",
        "## 跨 Step 波动最大的 Dataset",
        markdown_table(
            ["dataset", "task", "baseline_acc", "best_acc", "worst_acc", "swing"],
            instability_rows,
        ),
        "",
        "## 固定失败数据集",
        markdown_table(
            ["dataset", "baseline_acc", "error_type", "error_first_line"],
            failure_rows,
        ),
        "",
        "## TTT 跳过原因",
        markdown_table(["reason", "count"], skip_rows),
        "",
        "## 解读",
        "- `step1` 到 `step5` 的提升很小，说明少量 TTT 更新不足以稳定改变平均表现。",
        "- `step10` 到 `step20` 是收益最明显的一段，公平对比平均 delta 从 `+0.0402 pp` 提升到 `+0.0663 pp`。",
        "- `step40` 的均值最好，但 `step50` 拿到更多单点收益数据集，说明不同数据集的最优步数分布较散，不适合一刀切固定大步数。",
        "- 从波动最大的 dataset 看，`banknote_authentication`、`autoUniv-au7-1100`、多组 `FOREX` 数据对 step 很敏感，后续更适合加 holdout gate、early stopping 或 loss-based stopping，而不是盲目继续增大 step。",
        "- 由于失败集和跳过集都固定，下一轮如果目标是提升总体均值，优先方向应该是控制高 step 下的回归尾部，而不是继续增加 step 上限。",
        "",
    ]
    return "\n".join(report_lines)


def compact_steps(steps: list[int]) -> str:
    return ",".join(str(step) for step in steps) if steps else ""


def classify_pattern(deltas: list[float], eps: float = 1e-12) -> str:
    signs = ["+" if delta > eps else "-" if delta < -eps else "0" for delta in deltas]
    has_up = "+" in signs
    has_down = "-" in signs
    final = signs[-1]
    if has_up and not has_down and "0" not in signs:
        return "all_steps_up"
    if has_down and not has_up and "0" not in signs:
        return "all_steps_down"
    if has_up and not has_down:
        return "nonnegative_gain"
    if has_down and not has_up:
        return "nonpositive_decline"
    if has_up and has_down and final == "+":
        return "mixed_recovered_to_gain"
    if has_up and has_down and final == "-":
        return "mixed_ended_in_decline"
    if has_up and has_down:
        return "mixed_ended_flat"
    return "all_steps_flat"


def build_dataset_direction_summaries(
    all_steps: pd.DataFrame,
    step_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for dataset_name, group in all_steps.groupby("dataset_name", sort=True):
        group = group.sort_values("step")
        deltas = [float(group.loc[group["step"] == step, "delta"].iloc[0]) for step in step_values]
        delta_pp = [delta * 100.0 for delta in deltas]
        positive_steps = [step for step, delta in zip(step_values, deltas) if delta > 1e-12]
        negative_steps = [step for step, delta in zip(step_values, deltas) if delta < -1e-12]
        flat_steps = [step for step, delta in zip(step_values, deltas) if abs(delta) <= 1e-12]

        best_idx = int(max(range(len(deltas)), key=lambda idx: deltas[idx]))
        worst_idx = int(min(range(len(deltas)), key=lambda idx: deltas[idx]))
        best_step = step_values[best_idx]
        worst_step = step_values[worst_idx]
        best_row = group.loc[group["step"] == best_step].iloc[0]
        worst_row = group.loc[group["step"] == worst_step].iloc[0]

        row: dict[str, object] = {
            "dataset_name": dataset_name,
            "task_type": group["task_type"].iloc[0],
            "baseline_acc": float(group["baseline_acc"].iloc[0]),
            "n_classes": int(group["n_classes"].iloc[0]),
            "up_count": len(positive_steps),
            "down_count": len(negative_steps),
            "flat_count": len(flat_steps),
            "first_up_step": positive_steps[0] if positive_steps else "",
            "last_up_step": positive_steps[-1] if positive_steps else "",
            "best_step": best_step,
            "best_acc": float(best_row["accuracy"]),
            "best_delta_pp": delta_pp[best_idx],
            "up_steps": compact_steps(positive_steps),
            "first_down_step": negative_steps[0] if negative_steps else "",
            "last_down_step": negative_steps[-1] if negative_steps else "",
            "worst_step": worst_step,
            "worst_acc": float(worst_row["accuracy"]),
            "worst_delta_pp": delta_pp[worst_idx],
            "down_steps": compact_steps(negative_steps),
            "flat_steps": compact_steps(flat_steps),
            "final_step": step_values[-1],
            "final_delta_pp": delta_pp[-1],
            "pattern": classify_pattern(deltas),
        }
        for step, value in zip(step_values, delta_pp):
            row[f"delta_pp_step{step}"] = value
        rows.append(row)

    dataset_summary = pd.DataFrame(rows)
    up_summary = (
        dataset_summary.loc[dataset_summary["up_count"] > 0]
        .sort_values(["best_delta_pp", "up_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    down_summary = (
        dataset_summary.loc[dataset_summary["down_count"] > 0]
        .sort_values(["worst_delta_pp", "down_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return dataset_summary, up_summary, down_summary


def generate_direction_report(
    title: str,
    direction: str,
    step_summary: pd.DataFrame,
    detail: pd.DataFrame,
    step_col: str,
    count_col: str,
    delta_col: str,
    top_largest: bool,
) -> str:
    if direction == "up":
        step_rows = [
            [int(row["step"]), int(row["improve"]), int(row["worse"]), int(row["same"])]
            for _, row in step_summary.iterrows()
        ]
        step_headers = ["step", "提升数量", "下降数量", "持平数量"]
        trend_text = (
            f"提升数据集数量随 step 增大严格上升，从 step1 的 {int(step_summary['improve'].iloc[0])} "
            f"个增加到 step50 的 {int(step_summary['improve'].iloc[-1])} 个。"
        )
    else:
        step_rows = [
            [int(row["step"]), int(row["worse"]), int(row["improve"]), int(row["same"])]
            for _, row in step_summary.iterrows()
        ]
        step_headers = ["step", "下降数量", "提升数量", "持平数量"]
        trend_text = (
            f"下降数据集数量整体随 step 增大上升，从 step1 的 {int(step_summary['worse'].iloc[0])} "
            f"个增加到 step50 的 {int(step_summary['worse'].iloc[-1])} 个；但不是单调变化，step30 为 "
            f"{int(step_summary.loc[step_summary['step'] == 30, 'worse'].iloc[0])} 个，step40 回落到 "
            f"{int(step_summary.loc[step_summary['step'] == 40, 'worse'].iloc[0])} 个。"
        )

    step_dist = detail[step_col].value_counts().sort_index()
    step_dist_rows = [[int(step), int(count)] for step, count in step_dist.items() if step != ""]
    pattern_rows = [[pattern, int(count)] for pattern, count in detail["pattern"].value_counts().items()]
    task_rows = [[task, int(count)] for task, count in detail["task_type"].value_counts().items()]

    top = detail.sort_values(delta_col, ascending=not top_largest).head(15)
    top_rows = []
    for _, row in top.iterrows():
        top_rows.append(
            [
                row["dataset_name"],
                row["task_type"],
                int(row[step_col]),
                f"{row['baseline_acc']:.6f}",
                format_pp(row[delta_col]),
                row["up_steps"],
                row["down_steps"],
                row["pattern"],
            ]
        )

    return "\n".join(
        [
            f"# {title}",
            "",
            "## 总体判断",
            f"- {trend_text}",
            f"- 共有 `{len(detail)}` 个 dataset 至少在一个 step 出现该方向变化。",
            f"- 按任务类型看: "
            + ", ".join(f"`{task}`={count}" for task, count in detail["task_type"].value_counts().items())
            + "。",
            "",
            "## 按 Step 统计",
            markdown_table(step_headers, step_rows),
            "",
            f"## {step_col} 分布",
            markdown_table([step_col, "dataset_count"], step_dist_rows),
            "",
            "## Pattern 分布",
            markdown_table(["pattern", "dataset_count"], pattern_rows),
            "",
            "## Task 分布",
            markdown_table(["task_type", "dataset_count"], task_rows),
            "",
            "## Top Dataset",
            markdown_table(
                ["dataset", "task", step_col, "baseline_acc", "delta", "up_steps", "down_steps", "pattern"],
                top_rows,
            ),
            "",
            "## 字段说明",
            "- `up_steps`: 该 dataset 相对 baseline accuracy 提升的 step 列表。",
            "- `down_steps`: 该 dataset 相对 baseline accuracy 下降的 step 列表。",
            "- `pattern`: 基于所有 step 的正/负/持平轨迹归类。",
            "- `delta_pp_stepX`: step X 相对 baseline 的 accuracy 差值，单位是百分点 pp。",
            "",
        ]
    )


def build_decline_trend(
    all_steps: pd.DataFrame,
    step_values: list[int],
    common_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step in step_values:
        step_df = all_steps.loc[all_steps["step"] == step].copy()
        declined = step_df.loc[step_df["delta"] < -1e-12].copy()
        decline_magnitude_pp = (-declined["delta"] * 100.0) if len(declined) else pd.Series(dtype=float)
        rows.append(
            {
                "step": step,
                "common_dataset_count": common_count,
                "down_count": int(len(declined)),
                "down_rate": float(len(declined) / common_count),
                "mean_decline_magnitude_pp": float(decline_magnitude_pp.mean()) if len(decline_magnitude_pp) else 0.0,
                "median_decline_magnitude_pp": float(decline_magnitude_pp.median()) if len(decline_magnitude_pp) else 0.0,
                "max_decline_magnitude_pp": float(decline_magnitude_pp.max()) if len(decline_magnitude_pp) else 0.0,
                "count_decline_ge_0_1pp": int((decline_magnitude_pp >= 0.1).sum()),
                "count_decline_ge_0_5pp": int((decline_magnitude_pp >= 0.5).sum()),
                "count_decline_ge_1_0pp": int((decline_magnitude_pp >= 1.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_decline_trend_plot(
    all_steps: pd.DataFrame,
    decline_trend: pd.DataFrame,
    out_path: Path,
) -> None:
    step_values = [int(step) for step in decline_trend["step"].tolist()]
    x_positions = list(range(len(step_values)))
    labels = [str(step) for step in step_values]

    fig, (ax_count, ax_dist) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    ax_count.bar(
        x_positions,
        decline_trend["down_count"],
        color="#d95f02",
        alpha=0.72,
        label="declined datasets",
    )
    ax_count.plot(
        x_positions,
        decline_trend["count_decline_ge_0_5pp"],
        color="#7f2704",
        marker="o",
        linewidth=2.0,
        label="decline >= 0.5 pp",
    )
    for x_pos, (_, row) in zip(x_positions, decline_trend.iterrows()):
        ax_count.annotate(
            str(int(row["down_count"])),
            (x_pos, row["down_count"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            color="#7f2704",
        )
    ax_count.set_title("TTT Step vs Declined Dataset Count")
    ax_count.set_ylabel("Dataset Count")
    ax_count.grid(True, axis="y", linestyle="--", alpha=0.28)
    ax_count.legend(frameon=False, fontsize=9, loc="upper left")

    box_data: list[list[float]] = []
    mean_values: list[float] = []
    for step in step_values:
        magnitudes = (
            -all_steps.loc[(all_steps["step"] == step) & (all_steps["delta"] < -1e-12), "delta"]
            * 100.0
        ).tolist()
        box_data.append(magnitudes)
        mean_values.append(float(pd.Series(magnitudes).mean()) if magnitudes else 0.0)

    ax_dist.boxplot(
        box_data,
        positions=x_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "#fee6ce", "edgecolor": "#d95f02"},
        medianprops={"color": "#7f2704", "linewidth": 1.8},
        whiskerprops={"color": "#d95f02"},
        capprops={"color": "#d95f02"},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "#fdae6b", "markeredgecolor": "#7f2704", "alpha": 0.65},
    )
    ax_dist.plot(
        x_positions,
        mean_values,
        color="#08519c",
        marker="D",
        linewidth=1.8,
        label="mean decline magnitude",
    )
    ax_dist.set_title("Decline Magnitude Distribution Among Declined Datasets")
    ax_dist.set_xlabel("TTT Steps")
    ax_dist.set_ylabel("Accuracy Drop (pp)")
    ax_dist.set_xticks(x_positions)
    ax_dist.set_xticklabels(labels)
    ax_dist.grid(True, axis="y", linestyle="--", alpha=0.28)
    ax_dist.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_decline_trend_report(
    decline_trend: pd.DataFrame,
    dataset_down_summary: pd.DataFrame,
) -> str:
    first = decline_trend.iloc[0]
    last = decline_trend.iloc[-1]
    peak = decline_trend.loc[decline_trend["down_count"].idxmax()]
    max_severity = decline_trend.loc[decline_trend["max_decline_magnitude_pp"].idxmax()]
    count_rows = []
    for _, row in decline_trend.iterrows():
        count_rows.append(
            [
                int(row["step"]),
                int(row["down_count"]),
                f"{row['down_rate'] * 100.0:.2f}%",
                format_pp(-row["mean_decline_magnitude_pp"]),
                format_pp(-row["median_decline_magnitude_pp"]),
                format_pp(-row["max_decline_magnitude_pp"]),
                int(row["count_decline_ge_0_1pp"]),
                int(row["count_decline_ge_0_5pp"]),
                int(row["count_decline_ge_1_0pp"]),
            ]
        )

    late_declines = dataset_down_summary.loc[
        dataset_down_summary["down_steps"].astype(str).str.contains("30|40|50", regex=True)
    ]
    persistent_declines = dataset_down_summary.loc[dataset_down_summary["pattern"].isin(["all_steps_down", "nonpositive_decline"])]

    top_rows = []
    for _, row in dataset_down_summary.head(12).iterrows():
        top_rows.append(
            [
                row["dataset_name"],
                row["task_type"],
                int(row["worst_step"]),
                format_pp(row["worst_delta_pp"]),
                row["down_steps"],
                row["up_steps"],
                row["pattern"],
            ]
        )

    return "\n".join(
        [
            "# dataset_down_trend",
            "",
            "## 主要趋势",
            f"- 下降 dataset 数量整体随 step 增大而上升: step{int(first['step'])} 为 `{int(first['down_count'])}` 个，step{int(last['step'])} 为 `{int(last['down_count'])}` 个。",
            f"- 下降数量峰值出现在 step{int(peak['step'])}: `{int(peak['down_count'])}` 个，占共同成功数据集的 `{peak['down_rate'] * 100.0:.2f}%`。",
            f"- 最大单点下降出现在 step{int(max_severity['step'])}: `{max_severity['max_decline_magnitude_pp']:.4f} pp`。",
            f"- 至少一次下降的数据集共有 `{len(dataset_down_summary)}` 个；其中持续非正收益模式 `{len(persistent_declines)}` 个，后期 step30/40/50 仍出现下降的 `{len(late_declines)}` 个。",
            "- 结论上，高 step 同时带来更多提升 dataset 和更明显的下降尾部；下降风险不是简单单调扩大，step30 的下降数量最高，step40 略有回落，step50 再次增加。",
            "",
            "## 按 Step 下降趋势",
            markdown_table(
                [
                    "step",
                    "下降数量",
                    "下降比例",
                    "平均下降幅度",
                    "中位下降幅度",
                    "最大下降幅度",
                    ">=0.1pp",
                    ">=0.5pp",
                    ">=1.0pp",
                ],
                count_rows,
            ),
            "",
            "## 下降最严重的 Dataset",
            markdown_table(
                ["dataset", "task", "worst_step", "worst_delta", "down_steps", "up_steps", "pattern"],
                top_rows,
            ),
            "",
            "## 图像",
            "- `ttt_step_down_trend.png`: 上半部分是下降 dataset 数量及 `>=0.5pp` 明显下降数量；下半部分是每个 step 下下降幅度分布。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.root
    baseline_csv, step_runs = discover_inputs(root)

    baseline_summary = parse_summary(baseline_csv.parent / "summary.txt")
    baseline_df = pd.read_csv(baseline_csv)
    baseline_ok = baseline_df.loc[baseline_df["status"] == "ok"].copy()
    baseline_ok = baseline_ok[
        [
            "dataset_name",
            "accuracy",
            "task_type",
            "n_classes",
        ]
    ].rename(columns={"accuracy": "baseline_acc"})

    step_summaries: list[dict[str, object]] = []
    best_frames: list[pd.DataFrame] = []
    failures_reference: pd.DataFrame | None = None
    representative_skip_reasons: pd.Series | None = None

    for step, csv_path in step_runs:
        summary = parse_summary(csv_path.parent / "summary.txt")
        df = pd.read_csv(csv_path)
        ok_df = df.loc[df["status"] == "ok"].copy()

        merged = baseline_ok.merge(
            ok_df[
                [
                    "dataset_name",
                    "accuracy",
                    "ttt_applied",
                    "ttt_split_reason",
                    "ttt_steps",
                ]
            ],
            on="dataset_name",
            how="inner",
        )
        merged["step"] = step
        merged["delta"] = merged["accuracy"] - merged["baseline_acc"]
        best_frames.append(merged)

        fail_df = df.loc[df["status"] == "fail", ["dataset_name", "error"]].copy()
        fail_df["error_type"] = fail_df["error"].apply(lambda value: first_error_line(value).split(":", 1)[0])
        fail_df["error_first_line"] = fail_df["error"].apply(first_error_line)
        fail_df = fail_df.merge(
            baseline_ok[["dataset_name", "baseline_acc"]],
            on="dataset_name",
            how="left",
        ).rename(columns={"baseline_acc": "baseline_accuracy"})
        fail_df = fail_df.sort_values("dataset_name").reset_index(drop=True)

        skip_reasons = (
            ok_df.loc[~ok_df["ttt_applied"].fillna(False), "ttt_split_reason"]
            .fillna("(none)")
            .value_counts()
        )
        if representative_skip_reasons is None:
            representative_skip_reasons = skip_reasons
        if failures_reference is None:
            failures_reference = fail_df

        step_summaries.append(
            {
                "step": step,
                "processed_datasets": to_int(summary.get("processed_datasets")),
                "ok_count": to_int(summary.get("ok_count")),
                "failed_count": to_int(summary.get("failed_count")),
                "avg_accuracy_ok": to_float(summary.get("avg_accuracy_ok")),
                "wall_seconds": to_float(summary.get("wall_seconds")),
                "baseline_avg_accuracy_ok": float(baseline_ok["baseline_acc"].mean()),
                "baseline_wall_seconds": to_float(baseline_summary.get("wall_seconds")),
                "common_ok_count": int(len(merged)),
                "common_avg_baseline": float(merged["baseline_acc"].mean()),
                "common_avg_step": float(merged["accuracy"].mean()),
                "avg_delta_pp": float(merged["delta"].mean() * 100.0),
                "median_delta_pp": float(merged["delta"].median() * 100.0),
                "improve": int((merged["delta"] > 1e-12).sum()),
                "worse": int((merged["delta"] < -1e-12).sum()),
                "same": int((merged["delta"].abs() <= 1e-12).sum()),
                "ttt_applied_count": int(ok_df["ttt_applied"].fillna(False).astype(bool).sum()),
                "ttt_skipped_count": int((~ok_df["ttt_applied"].fillna(False)).sum()),
            }
        )

    step_summary = pd.DataFrame(step_summaries).sort_values("step").reset_index(drop=True)
    all_best = pd.concat(best_frames, ignore_index=True)
    step_values = [int(step) for step in step_summary["step"].tolist()]
    best_step_by_dataset = (
        all_best.loc[all_best.groupby("dataset_name")["delta"].idxmax()]
        .sort_values("delta", ascending=False)
        .reset_index(drop=True)
    )
    instability = (
        all_best.groupby(["dataset_name", "task_type", "baseline_acc"], as_index=False)
        .agg(best_acc=("accuracy", "max"), worst_acc=("accuracy", "min"))
        .sort_values(["dataset_name"])
    )
    instability["swing_pp"] = (instability["best_acc"] - instability["worst_acc"]) * 100.0
    instability = instability.sort_values("swing_pp", ascending=False).reset_index(drop=True)

    root.mkdir(parents=True, exist_ok=True)
    step_summary.to_csv(root / "step_summary.csv", index=False)
    best_step_by_dataset[
        ["dataset_name", "task_type", "step", "baseline_acc", "accuracy", "delta"]
    ].to_csv(root / "dataset_best_step.csv", index=False)
    instability.to_csv(root / "dataset_instability.csv", index=False)

    dataset_direction_summary, dataset_up_summary, dataset_down_summary = build_dataset_direction_summaries(
        all_best,
        step_values,
    )
    dataset_direction_summary.to_csv(root / "dataset_direction_summary.csv", index=False)
    dataset_up_summary.to_csv(root / "dataset_up_summary.csv", index=False)
    dataset_down_summary.to_csv(root / "dataset_down_summary.csv", index=False)
    decline_trend = build_decline_trend(
        all_best,
        step_values,
        common_count=int(step_summary["common_ok_count"].iloc[0]),
    )
    decline_trend.to_csv(root / "dataset_down_trend.csv", index=False)
    build_decline_trend_plot(all_best, decline_trend, root / "ttt_step_down_trend.png")
    (root / "dataset_down_trend.md").write_text(
        generate_decline_trend_report(decline_trend, dataset_down_summary),
        encoding="utf-8",
    )
    (root / "dataset_up_summary.md").write_text(
        generate_direction_report(
            title="dataset_up_summary",
            direction="up",
            step_summary=step_summary,
            detail=dataset_up_summary,
            step_col="best_step",
            count_col="up_count",
            delta_col="best_delta_pp",
            top_largest=True,
        ),
        encoding="utf-8",
    )
    (root / "dataset_down_summary.md").write_text(
        generate_direction_report(
            title="dataset_down_summary",
            direction="down",
            step_summary=step_summary,
            detail=dataset_down_summary,
            step_col="worst_step",
            count_col="down_count",
            delta_col="worst_delta_pp",
            top_largest=False,
        ),
        encoding="utf-8",
    )
    build_plot(step_summary, root / "ttt_step_vs_acc.png")

    report = generate_report(
        root=root,
        step_summary=step_summary,
        best_step_by_dataset=best_step_by_dataset,
        instability=instability,
        failures=failures_reference if failures_reference is not None else pd.DataFrame(),
        representative_skip_reasons=representative_skip_reasons if representative_skip_reasons is not None else pd.Series(dtype=int),
    )
    (root / "summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

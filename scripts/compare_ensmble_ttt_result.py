from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("result/ensmble_TTT_result")
ENS_DIR = ROOT / "ensmble_ttt_step10_lr5e-6"
ICL_DIR = ROOT / "iclv1.1_ttt_step10"


def parse_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def fmt_pp(value: float) -> str:
    return f"{value:+.4f} pp"


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


def main() -> None:
    ens_summary = parse_summary(ENS_DIR / "summary.txt")
    icl_summary = parse_summary(ICL_DIR / "summary.txt")

    ens_df = pd.read_csv(ENS_DIR / "all_classification_results.csv")
    icl_df = pd.read_csv(ICL_DIR / "all_classification_results.csv")

    ens_ok = ens_df.loc[
        ens_df["status"] == "ok",
        ["dataset_name", "task_type", "accuracy", "fit_seconds", "predict_seconds", "ttt_update_seconds"],
    ].copy()
    ens_ok = ens_ok.rename(columns={"accuracy": "accuracy_ensemble"})
    icl_ok = icl_df.loc[
        icl_df["status"] == "ok",
        ["dataset_name", "accuracy", "fit_seconds", "predict_seconds", "ttt_update_seconds"],
    ].copy()
    icl_ok = icl_ok.rename(columns={"accuracy": "accuracy_icl"})

    detail = ens_ok.merge(icl_ok, on="dataset_name", how="inner")
    detail["delta_ensemble_minus_icl"] = detail["accuracy_ensemble"] - detail["accuracy_icl"]
    detail["delta_pp"] = detail["delta_ensemble_minus_icl"] * 100.0
    detail["total_seconds_ensemble"] = (
        detail["fit_seconds_x"].fillna(0.0) + detail["predict_seconds_x"].fillna(0.0)
    )
    detail["total_seconds_icl"] = (
        detail["fit_seconds_y"].fillna(0.0) + detail["predict_seconds_y"].fillna(0.0)
    )
    detail = detail.rename(
        columns={
            "fit_seconds_x": "fit_seconds_ensemble",
            "predict_seconds_x": "predict_seconds_ensemble",
            "ttt_update_seconds_x": "ttt_update_seconds_ensemble",
            "fit_seconds_y": "fit_seconds_icl",
            "predict_seconds_y": "predict_seconds_icl",
            "ttt_update_seconds_y": "ttt_update_seconds_icl",
        }
    )
    detail["winner"] = detail["delta_ensemble_minus_icl"].map(
        lambda value: "ensemble"
        if value > 1e-12
        else "icl_training"
        if value < -1e-12
        else "tie"
    )
    detail = detail.sort_values(["delta_ensemble_minus_icl", "dataset_name"], ascending=[False, True]).reset_index(drop=True)
    detail.to_csv(ROOT / "detail.csv", index=False)

    ens_fail = set(ens_df.loc[ens_df["status"] != "ok", "dataset_name"].tolist())
    icl_fail = set(icl_df.loc[icl_df["status"] != "ok", "dataset_name"].tolist())
    extra_ens_fail = sorted(ens_fail - icl_fail)
    shared_fail = sorted(ens_fail & icl_fail)

    summary_rows = [
        [
            "ensemble_ttt_step10_lr5e-6",
            int(ens_summary["ok_count"]),
            int(ens_summary["failed_count"]),
            ens_summary["avg_accuracy_ok"],
            ens_summary["wall_seconds"],
        ],
        [
            "iclv1.1_ttt_step10",
            int(icl_summary["ok_count"]),
            int(icl_summary["failed_count"]),
            icl_summary["avg_accuracy_ok"],
            icl_summary["wall_seconds"],
        ],
    ]

    top_ens = detail.sort_values("delta_ensemble_minus_icl", ascending=False).head(10)
    top_icl = detail.sort_values("delta_ensemble_minus_icl", ascending=True).head(10)
    common_total_ens = float(detail["total_seconds_ensemble"].sum())
    common_total_icl = float(detail["total_seconds_icl"].sum())
    common_fit_ens = float(detail["fit_seconds_ensemble"].sum())
    common_fit_icl = float(detail["fit_seconds_icl"].sum())
    common_pred_ens = float(detail["predict_seconds_ensemble"].sum())
    common_pred_icl = float(detail["predict_seconds_icl"].sum())

    report = "\n".join(
        [
            "# ensmble_TTT_result summary",
            "",
            "## 输入目录",
            f"- ensemble: `{ENS_DIR}`",
            f"- ICL training: `{ICL_DIR}`",
            f"- 明细文件: `{ROOT / 'detail.csv'}`",
            "",
            "## 主要结论",
            f"- 原始全量 `avg_accuracy_ok` 上，ensemble 更高：`{float(ens_summary['avg_accuracy_ok']):.6f}` vs `{float(icl_summary['avg_accuracy_ok']):.6f}`，差值 `{fmt_pp((float(ens_summary['avg_accuracy_ok']) - float(icl_summary['avg_accuracy_ok'])) * 100.0)}`。",
            f"- 但这不能直接当作公平对比，因为 ensemble 失败了 `{int(ens_summary['failed_count'])}` 个数据集，而 ICL training 只失败了 `{int(icl_summary['failed_count'])}` 个。",
            f"- 在两边都成功的 `{len(detail)}` 个共同数据集上，ensemble 平均 acc 为 `{detail['accuracy_ensemble'].mean():.6f}`，ICL training 为 `{detail['accuracy_icl'].mean():.6f}`，ensemble 平均领先 `{fmt_pp(detail['delta_pp'].mean())}`。",
            f"- 逐 dataset 计数上，ensemble / ICL training / 持平 为 `{int((detail['winner'] == 'ensemble').sum())}` / `{int((detail['winner'] == 'icl_training').sum())}` / `{int((detail['winner'] == 'tie').sum())}`，说明 ensemble 在交集上总体占优，但并非全面压制。",
            f"- 只看这 `{len(detail)}` 个交集数据集的总耗时，ensemble 为 `{common_total_ens:.3f}s`，ICL training 为 `{common_total_icl:.3f}s`，ensemble 是 ICL training 的 `{(common_total_ens / common_total_icl):.2f}x`。",
            f"- ensemble 的额外失败集共有 `{len(extra_ens_fail)}` 个，主要是大表和高负载任务；共享失败集有 `{len(shared_fail)}` 个，都是两边都没跑通的数据集。",
            "",
            "## 原始运行概况",
            markdown_table(
                ["method", "ok_count", "failed_count", "avg_accuracy_ok", "wall_seconds"],
                summary_rows,
            ),
            "",
            "## 共同成功交集对比",
            markdown_table(
                [
                    "common_ok",
                    "ensemble_avg_acc",
                    "icl_avg_acc",
                    "mean_delta",
                    "median_delta",
                    "ensemble_win",
                    "icl_win",
                    "tie",
                    "|delta|>=0.1pp",
                    "|delta|>=0.5pp",
                    "ensemble_total_seconds",
                    "icl_total_seconds",
                    "time_ratio",
                ],
                [[
                    len(detail),
                    f"{detail['accuracy_ensemble'].mean():.6f}",
                    f"{detail['accuracy_icl'].mean():.6f}",
                    fmt_pp(detail['delta_pp'].mean()),
                    fmt_pp(detail['delta_pp'].median()),
                    int((detail['winner'] == 'ensemble').sum()),
                    int((detail['winner'] == 'icl_training').sum()),
                    int((detail['winner'] == 'tie').sum()),
                    int((detail['delta_pp'].abs() >= 0.1).sum()),
                    int((detail['delta_pp'].abs() >= 0.5).sum()),
                    f"{common_total_ens:.3f}",
                    f"{common_total_icl:.3f}",
                    f"{(common_total_ens / common_total_icl):.2f}x",
                ]],
            ),
            "",
            "## 交集时间拆解",
            markdown_table(
                ["metric", "ensemble", "ICL training"],
                [
                    ["fit_seconds_sum", f"{common_fit_ens:.3f}", f"{common_fit_icl:.3f}"],
                    ["predict_seconds_sum", f"{common_pred_ens:.3f}", f"{common_pred_icl:.3f}"],
                    ["total_seconds_sum", f"{common_total_ens:.3f}", f"{common_total_icl:.3f}"],
                ],
            ),
            "",
            "## Ensemble 优势最大的 Dataset",
            markdown_table(
                ["dataset", "task", "ensemble_acc", "icl_acc", "delta"],
                [
                    [
                        row["dataset_name"],
                        row["task_type"],
                        f"{row['accuracy_ensemble']:.6f}",
                        f"{row['accuracy_icl']:.6f}",
                        fmt_pp(row["delta_pp"]),
                    ]
                    for _, row in top_ens.iterrows()
                ],
            ),
            "",
            "## ICL training 优势最大的 Dataset",
            markdown_table(
                ["dataset", "task", "ensemble_acc", "icl_acc", "delta"],
                [
                    [
                        row["dataset_name"],
                        row["task_type"],
                        f"{row['accuracy_ensemble']:.6f}",
                        f"{row['accuracy_icl']:.6f}",
                        fmt_pp(row["delta_pp"]),
                    ]
                    for _, row in top_icl.iterrows()
                ],
            ),
            "",
            "## 失败集说明",
            f"- ensemble 额外失败集 `{len(extra_ens_fail)}` 个: `{', '.join(extra_ens_fail) if extra_ens_fail else '(none)'}`。",
            f"- 共享失败集 `{len(shared_fail)}` 个: `{', '.join(shared_fail) if shared_fail else '(none)'}`。",
            "",
        ]
    )

    (ROOT / "summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

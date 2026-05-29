#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data178}"
MODEL_PATH="${MODEL_PATH:-tabicl-classifier-v2-20260212.ckpt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-1b_result/ablation_1c_vs_1d_${RUN_ID}}"
ONE_C_GPU_GROUP="${ONE_C_GPU_GROUP:-2}"
ONE_D_GPU_GROUP="${ONE_D_GPU_GROUP:-3}"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Data root does not exist: ${DATA_ROOT}" >&2
    exit 1
fi

if [[ -z "${MAX_DATASETS:-}" ]]; then
    MAX_DATASETS="$(find "${DATA_ROOT}" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d '[:space:]')"
fi

ONE_C_OUT_DIR="${OUT_ROOT}/1c_chunk_ttt"
ONE_D_OUT_DIR="${OUT_ROOT}/1d_taar_sample_ttt"
LOG_DIR="${OUT_ROOT}/logs"
ONE_C_LOG="${LOG_DIR}/1c.log"
ONE_D_LOG="${LOG_DIR}/1d.log"

mkdir -p "${ONE_C_OUT_DIR}" "${ONE_D_OUT_DIR}" "${LOG_DIR}"

COMMON_ARGS=(
    --data-root "${DATA_ROOT}"
    --model-path "${MODEL_PATH}"
    --max-datasets "${MAX_DATASETS}"
    --n-estimators 32
    --batch-size 8
    --kv-cache False
    --use-amp auto
    --use-fa3 auto
    --offload-mode auto
    --random-state 42
    --ttt-lr 1e-5
    --ttt-scheduler cosine_warmup
    --ttt-warmup-proportion 0.1
    --ttt-grad-clip 1.0
    --ttt-dtype float32
    --ttt-micro-batch-size 1
    --ttt-weight-decay 0.01
    --ttt-epochs 30
    --ttt-max-chunk-size 8000
    --ttt-min-chunk-size 50
    --ttt-query-ratio 0.2
    --ttt-n-estimators-finetune 2
    --ttt-early-stopping True
    --ttt-patience 8
    --ttt-min-delta 1e-4
    --ttt-eval-metric roc_auc
    --ttt-validation-fraction 0.1
    --ttt-validation-n-estimators 2
    --ttt-freeze-col False
    --ttt-freeze-row False
    --ttt-freeze-icl False
    --ttt-save-ckpt False
)

echo "run_id: ${RUN_ID}"
echo "out_root: ${OUT_ROOT}"
echo "data_root: ${DATA_ROOT}"
echo "model_path: ${MODEL_PATH}"
echo "max_datasets: ${MAX_DATASETS}"
echo "1c_gpu_group: ${ONE_C_GPU_GROUP}"
echo "1d_gpu_group: ${ONE_D_GPU_GROUP}"

(
    set -x
    echo "[1C] GPU group: ${ONE_C_GPU_GROUP}"
    "${PYTHON_BIN}" 1C_Chunk_TTT.py \
        --out-dir "${ONE_C_OUT_DIR}" \
        --workers 1 \
        --gpu-groups "${ONE_C_GPU_GROUP}" \
        "${COMMON_ARGS[@]}"
) >"${ONE_C_LOG}" 2>&1 &
pid_1c=$!

(
    set -x
    echo "[1d] GPU group: ${ONE_D_GPU_GROUP}"
    "${PYTHON_BIN}" 1d_taar_sample_TTT.py \
        --out-dir "${ONE_D_OUT_DIR}" \
        --workers 1 \
        --gpu-groups "${ONE_D_GPU_GROUP}" \
        "${COMMON_ARGS[@]}" \
        --ttt-context-selector taar_sample \
        --ttt-taar-score-source attention \
        --ttt-taar-sample-selection True \
        --ttt-taar-feature-selection False \
        --ttt-taar-context-ratio 0.3 \
        --ttt-taar-min-context 300 \
        --ttt-taar-sample-attn 0.8 \
        --ttt-taar-class-coverage-repair True
) >"${ONE_D_LOG}" 2>&1 &
pid_1d=$!

status=0
if ! wait "${pid_1c}"; then
    echo "1C experiment failed. See ${ONE_C_LOG}" >&2
    status=1
fi
if ! wait "${pid_1d}"; then
    echo "1d experiment failed. See ${ONE_D_LOG}" >&2
    status=1
fi

if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
fi

COMPARE_CSV="${OUT_ROOT}/compare_1d_vs_1c.csv"
COMPARE_MD="${OUT_ROOT}/compare_1d_vs_1c.md"

"${PYTHON_BIN}" - "${ONE_C_OUT_DIR}" "${ONE_D_OUT_DIR}" "${COMPARE_CSV}" "${COMPARE_MD}" <<'PY'
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


one_c_dir = Path(sys.argv[1])
one_d_dir = Path(sys.argv[2])
compare_csv = Path(sys.argv[3])
compare_md = Path(sys.argv[4])

one_c_csv = one_c_dir / "all_classification_results.csv"
one_d_csv = one_d_dir / "all_classification_results.csv"

missing = [str(path) for path in (one_c_csv, one_d_csv) if not path.exists()]
if missing:
    raise FileNotFoundError("Missing result CSV(s): " + ", ".join(missing))

df_1c = pd.read_csv(one_c_csv)
df_1d = pd.read_csv(one_d_csv)

required = {"dataset_name", "status", "accuracy"}
for label, df in (("1C", df_1c), ("1d", df_1d)):
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"{label} result is missing columns: {missing_cols}")

ok_1c = df_1c[df_1c["status"].astype(str) == "ok"].copy()
ok_1d = df_1d[df_1d["status"].astype(str) == "ok"].copy()

ok_1c = ok_1c.drop_duplicates(subset=["dataset_name"], keep="last")
ok_1d = ok_1d.drop_duplicates(subset=["dataset_name"], keep="last")

merged = ok_1c.merge(
    ok_1d,
    on="dataset_name",
    how="inner",
    suffixes=("_1c", "_1d"),
)

out = pd.DataFrame({"dataset_name": merged["dataset_name"]})

metric_columns = ["accuracy", "f1", "balanced_accuracy", "roc_auc", "log_loss"]
for metric in metric_columns:
    left = f"{metric}_1c"
    right = f"{metric}_1d"
    if left in merged.columns and right in merged.columns:
        out[left] = pd.to_numeric(merged[left], errors="coerce")
        out[right] = pd.to_numeric(merged[right], errors="coerce")
        delta_name = "delta_accuracy" if metric == "accuracy" else f"delta_{metric}"
        out[delta_name] = out[right] - out[left]

taar_columns = [
    "ttt_context_selector",
    "ttt_taar_score_source",
    "ttt_taar_selected_context_mean",
    "ttt_taar_selected_context_min",
    "ttt_taar_selected_context_max",
    "ttt_taar_fallback_count",
    "ttt_taar_skip_count",
    "ttt_taar_feature_enabled",
    "ttt_taar_feature_score_source",
    "ttt_taar_selected_feature_mean",
    "ttt_taar_feature_fallback_count",
]
for column in taar_columns:
    source = f"{column}_1d"
    if source in merged.columns:
        out[column] = merged[source]

if "delta_accuracy" in out.columns:
    out = out.sort_values("delta_accuracy", ascending=False, na_position="last")

compare_csv.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(compare_csv, index=False)

total_1c = len(df_1c)
total_1d = len(df_1d)
ok_count_1c = len(ok_1c)
ok_count_1d = len(ok_1d)
intersection = len(out)

if intersection and "delta_accuracy" in out.columns:
    delta = pd.to_numeric(out["delta_accuracy"], errors="coerce")
    acc_1c = pd.to_numeric(out["accuracy_1c"], errors="coerce")
    acc_1d = pd.to_numeric(out["accuracy_1d"], errors="coerce")
    improved = int((delta > 0).sum())
    regressed = int((delta < 0).sum())
    tied = int((delta == 0).sum())
    mean_1c = float(acc_1c.mean())
    mean_1d = float(acc_1d.mean())
    mean_delta = float(delta.mean())
    median_delta = float(delta.median())
    direction = "improved" if mean_delta > 0 else "regressed" if mean_delta < 0 else "tied"
    headline = (
        f"1d {direction} vs 1C on the shared successful intersection: "
        f"mean accuracy {mean_1d:.6f} vs {mean_1c:.6f}, "
        f"delta {mean_delta:+.6f} across {intersection} datasets."
    )
else:
    improved = regressed = tied = 0
    mean_1c = mean_1d = mean_delta = median_delta = math.nan
    headline = "No shared successful datasets were available for comparison."

top_gain = out.head(20).copy() if "delta_accuracy" in out.columns else pd.DataFrame()
top_loss = (
    out.sort_values("delta_accuracy", ascending=True, na_position="last").head(20).copy()
    if "delta_accuracy" in out.columns
    else pd.DataFrame()
)

def fmt_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.6f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(none)"
    visible = [column for column in columns if column in df.columns]
    rows = df[visible].copy()
    for column in visible:
        if pd.api.types.is_numeric_dtype(rows[column]):
            rows[column] = rows[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
        else:
            rows[column] = rows[column].astype(str)
    header = "| " + " | ".join(visible) + " |"
    separator = "| " + " | ".join("---" for _ in visible) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in visible) + " |"
        for _, row in rows.iterrows()
    ]
    return "\n".join([header, separator, *body])


lines = [
    "# 1d TAAR Sample TTT vs 1C Chunk TTT",
    "",
    headline,
    "",
    "## Summary",
    "",
    f"- total_1c_rows: {total_1c}",
    f"- total_1d_rows: {total_1d}",
    f"- ok_1c: {ok_count_1c}",
    f"- ok_1d: {ok_count_1d}",
    f"- shared_ok_intersection: {intersection}",
    f"- mean_accuracy_1c: {fmt_float(mean_1c)}",
    f"- mean_accuracy_1d: {fmt_float(mean_1d)}",
    f"- mean_delta_accuracy: {fmt_float(mean_delta)}",
    f"- median_delta_accuracy: {fmt_float(median_delta)}",
    f"- improved_datasets: {improved}",
    f"- regressed_datasets: {regressed}",
    f"- tied_datasets: {tied}",
    "",
    "## Top Gains",
    "",
    markdown_table(top_gain, ["dataset_name", "accuracy_1c", "accuracy_1d", "delta_accuracy"]),
    "",
    "## Top Losses",
    "",
    markdown_table(top_loss, ["dataset_name", "accuracy_1c", "accuracy_1d", "delta_accuracy"]),
    "",
    "## Outputs",
    "",
    f"- comparison_csv: `{compare_csv}`",
    f"- 1c_results: `{one_c_csv}`",
    f"- 1d_results: `{one_d_csv}`",
    "",
]

compare_md.write_text("\n".join(lines), encoding="utf-8")
print(f"saved_compare_csv: {compare_csv}")
print(f"saved_compare_md: {compare_md}")
PY

echo "done"
echo "compare_csv: ${COMPARE_CSV}"
echo "compare_md: ${COMPARE_MD}"
echo "1c_log: ${ONE_C_LOG}"
echo "1d_log: ${ONE_D_LOG}"

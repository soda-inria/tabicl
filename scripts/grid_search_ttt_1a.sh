#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-result/ttt_1a_grid_search}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STEPS=(1 5 10 20 30 40 50)
LRS=(1e-4 3.7606e-5 1.4142e-5 5.3183e-6 )

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

for step in "${STEPS[@]}"; do
    for lr in "${LRS[@]}"; do
        out_dir="${OUTPUT_ROOT}/iclv1.1_ttt_step${step}_lr${lr}"

        echo "========================================"
        echo "Running TTT_1A_ICL.py with step=${step}, lr=${lr}"
        echo "out_dir=${out_dir}"
        echo "========================================"

        "${PYTHON_BIN}" TTT_1A_ICL.py \
            --ttt-steps "${step}" \
            --ttt-lr "${lr}" \
            --out-dir "${out_dir}" \
            "$@"
    done
done

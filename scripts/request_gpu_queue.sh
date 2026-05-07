#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-crc00006699}"
PARTITION="${PARTITION:-gpu-rtx3090}"
NODE="${NODE:-gpu1}"
GPUS="${GPUS:-4}"
TIME_LIMIT="${TIME_LIMIT:-144:00:00}"
JOB_NAME="${JOB_NAME:-interact}"
SHOW_QUEUE="${SHOW_QUEUE:-1}"
USE_NODE=1

usage() {
    cat <<'USAGE'
Request an interactive Slurm GPU allocation and wait in the queue.

Defaults match the current RTX3090 server setup:
  account   crc00006699
  partition gpu-rtx3090
  node      gpu4
  gpus      2
  time      144:00:00

Usage:
  scripts/request_gpu_queue.sh [options] [-- command ...]

Options:
  -A, --account ACCOUNT      Slurm account
  -p, --partition PARTITION  Slurm partition
  -w, --node NODE            Requested node, for example gpu4
      --any-node             Do not pin to a specific node
  -g, --gpus N               Number of GPUs
  -t, --time HH:MM:SS        Time limit
  -J, --job-name NAME        Job name shown by squeue
      --no-squeue            Do not print queue status before salloc
  -h, --help                 Show this help

Examples:
  scripts/request_gpu_queue.sh
  scripts/request_gpu_queue.sh --any-node
  scripts/request_gpu_queue.sh -w gpu4 -g 2 -t 72:00:00
  scripts/request_gpu_queue.sh --any-node -- bash -lc 'nvidia-smi && zsh'

You can also override defaults with environment variables:
  ACCOUNT=crc00006699 PARTITION=gpu-rtx3090 NODE=gpu4 GPUS=2 TIME_LIMIT=72:00:00 scripts/request_gpu_queue.sh
USAGE
}

cmd=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -A|--account)
            ACCOUNT="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -w|--node)
            NODE="$2"
            USE_NODE=1
            shift 2
            ;;
        --any-node)
            USE_NODE=0
            shift
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -J|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --no-squeue)
            SHOW_QUEUE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            cmd=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v salloc >/dev/null 2>&1; then
    echo "salloc was not found. Run this script on the Slurm login/workstation node." >&2
    exit 127
fi

if [[ "$SHOW_QUEUE" == "1" ]]; then
    echo "Current queue for partition ${PARTITION}:"
    squeue -p "$PARTITION" || true
    echo
    if command -v sinfo >/dev/null 2>&1; then
        echo "Node state for partition ${PARTITION}:"
        sinfo -p "$PARTITION" -N -o "%N %t %G %E" || true
        echo
    fi
fi

salloc_cmd=(
    salloc
    --account="$ACCOUNT"
    -p "$PARTITION"
    --gres="gpu:${GPUS}"
    -t "$TIME_LIMIT"
    -J "$JOB_NAME"
)

if [[ "$USE_NODE" == "1" ]]; then
    salloc_cmd+=(-w "$NODE")
fi

echo "Requesting allocation:"
printf ' %q' "${salloc_cmd[@]}"
if [[ ${#cmd[@]} -gt 0 ]]; then
    printf ' --'
    printf ' %q' "${cmd[@]}"
fi
echo
echo
echo "If this stays pending with ReqNodeNotAvail, retry with --any-node or another -w node."

if [[ ${#cmd[@]} -gt 0 ]]; then
    exec "${salloc_cmd[@]}" "${cmd[@]}"
else
    exec "${salloc_cmd[@]}"
fi

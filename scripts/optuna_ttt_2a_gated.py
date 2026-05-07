#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TTT_SCRIPT = REPO_ROOT / "TTT_2A_gated.py"
DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "result"
    / "ttt_step_result"
    / "iclv1.1_baseline"
    / "tabiclv1_1_classification_3gpu"
    / "all_classification_results.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "result" / "ttt_2a_optuna"
STEP_CHOICES = (1, 5, 10, 20, 30, 40, 50)
RESERVED_TTT_ARGS = {
    "--ttt-steps",
    "--ttt-lr",
    "--ttt-b-fraction",
    "--ttt-c-fraction",
    "--ttt-d-fraction",
    "--out-dir",
}


def load_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "optuna is required for scripts/optuna_ttt_2a_gated.py. "
            "Install it with `pip install optuna` and rerun."
        ) from exc
    return optuna


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna hyperparameter search for TTT_2A_gated.py. "
            "Unknown arguments are forwarded to TTT_2A_gated.py."
        )
    )
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--study-name", default="ttt_2a_gated_optuna")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python-bin", default=sys.executable or "python")
    parser.add_argument("--load-if-exists", dest="load_if_exists", action="store_true")
    parser.add_argument("--no-load-if-exists", dest="load_if_exists", action="store_false")
    parser.add_argument("--resume", dest="load_if_exists", action="store_true")
    parser.add_argument("--no-resume", dest="load_if_exists", action="store_false")
    parser.set_defaults(load_if_exists=True)
    args, forwarded_args = parser.parse_known_args(argv)
    validate_forwarded_args(forwarded_args)
    return args, forwarded_args


def validate_forwarded_args(forwarded_args: Sequence[str]) -> None:
    conflicts = []
    for arg in forwarded_args:
        if not arg.startswith("-"):
            continue
        key = arg.split("=", 1)[0]
        if key in RESERVED_TTT_ARGS:
            conflicts.append(arg)
    if conflicts:
        raise ValueError(
            "Do not pass trial-controlled arguments through this script: "
            + ", ".join(conflicts)
        )


def default_storage_uri(output_root: Path, study_name: str) -> str:
    db_path = output_root.resolve() / f"{study_name}.db"
    return f"sqlite:///{db_path.as_posix()}"


def parse_summary(summary_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value in {"", "(none)"}:
        return None
    return float(value)


def format_out_dir_name(step: int, lr: float, b_size: float, c_size: float, d_size: float) -> str:
    return (
        f"ae82_step{int(step)}_lr{float(lr):.2e}"
        f"_b{float(b_size):.4f}_c{float(c_size):.4f}_d{float(d_size):.4f}"
    )


def compute_d_bounds(b_size: float) -> tuple[float, float]:
    lower = max(0.05, 0.7 - float(b_size))
    upper = min(0.15, 0.8 - float(b_size))
    if lower > upper:
        raise ValueError(f"No feasible d_size range for b_size={b_size}")
    return lower, upper


def validate_ttt_fractions(b_size: float, c_size: float, d_size: float) -> None:
    eps = 1e-8
    total = float(b_size) + float(c_size) + float(d_size)
    if not (0.5 - eps) <= b_size <= (0.7 + eps):
        raise ValueError(f"b_size out of range: {b_size}")
    if not (0.2 - eps) <= c_size <= (0.3 + eps):
        raise ValueError(f"c_size out of range: {c_size}")
    if not (0.05 - eps) <= d_size <= (0.15 + eps):
        raise ValueError(f"d_size out of range: {d_size}")
    if abs(total - 1.0) > eps:
        raise ValueError(f"b_size + c_size + d_size must equal 1.0, got {total}")


def sample_ttt_fractions(trial: Any) -> tuple[float, float, float]:
    b_size = float(trial.suggest_float("b_size", 0.55, 0.70))
    d_min, d_max = compute_d_bounds(b_size)
    d_size = float(trial.suggest_float("d_size", d_min, d_max))
    c_size = float(round(1.0 - b_size - d_size, 12))
    validate_ttt_fractions(b_size, c_size, d_size)
    return b_size, c_size, d_size


def load_ok_accuracy_frame(csv_path: Path, *, accuracy_column: str = "accuracy") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ok_df = df.loc[df["status"] == "ok", ["dataset_name", accuracy_column]].copy()
    if ok_df.empty:
        raise ValueError(f"No successful datasets found in {csv_path}")
    return ok_df


def is_valid_trial_output(out_dir: Path) -> bool:
    csv_path = out_dir / "all_classification_results.csv"
    summary_path = out_dir / "summary.txt"
    if not csv_path.exists() or not summary_path.exists():
        return False
    try:
        summary = parse_summary(summary_path)
        if "avg_accuracy_ok" not in summary:
            return False
        pd.read_csv(csv_path, nrows=1)
    except Exception:
        return False
    return True


def build_trial_command(
    *,
    python_bin: str,
    forwarded_args: Sequence[str],
    out_dir: Path,
    step: int,
    lr: float,
    b_size: float,
    c_size: float,
    d_size: float,
) -> list[str]:
    return [
        python_bin,
        str(TTT_SCRIPT),
        *forwarded_args,
        "--ttt-steps",
        str(int(step)),
        "--ttt-lr",
        str(float(lr)),
        "--ttt-b-fraction",
        f"{float(b_size):.10f}",
        "--ttt-c-fraction",
        f"{float(c_size):.10f}",
        "--ttt-d-fraction",
        f"{float(d_size):.10f}",
        "--out-dir",
        str(out_dir),
    ]


def ensure_trial_outputs(
    *,
    out_dir: Path,
    command: Sequence[str],
    cwd: Path = REPO_ROOT,
    runner: Any = subprocess.run,
) -> bool:
    if is_valid_trial_output(out_dir):
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    runner(command, cwd=str(cwd), check=True)

    if not is_valid_trial_output(out_dir):
        raise RuntimeError(f"Trial run did not produce valid outputs in {out_dir}")
    return False


def evaluate_trial_outputs(baseline_ok_df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    csv_path = out_dir / "all_classification_results.csv"
    summary_path = out_dir / "summary.txt"
    trial_df = pd.read_csv(csv_path)
    trial_ok = trial_df.loc[trial_df["status"] == "ok", ["dataset_name", "accuracy"]].copy()
    merged = baseline_ok_df.merge(
        trial_ok.rename(columns={"accuracy": "trial_accuracy"}),
        on="dataset_name",
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No common successful datasets between baseline and {csv_path}")

    summary = parse_summary(summary_path)
    objective = float(merged["trial_accuracy"].mean() - merged["baseline_accuracy"].mean())
    return {
        "common_ok_count": int(len(merged)),
        "common_avg_baseline": float(merged["baseline_accuracy"].mean()),
        "common_avg_step": float(merged["trial_accuracy"].mean()),
        "objective": objective,
        "avg_delta_pp": float(objective * 100.0),
        "avg_accuracy_ok": parse_optional_float(summary.get("avg_accuracy_ok")),
        "ok_count": int((trial_df["status"] == "ok").sum()),
        "failed_count": int((trial_df["status"] == "fail").sum()),
        "wall_seconds": parse_optional_float(summary.get("wall_seconds")),
        "summary_path": str(summary_path),
        "all_csv_path": str(csv_path),
    }


def trial_to_record(trial: Any) -> dict[str, Any]:
    attrs = dict(getattr(trial, "user_attrs", {}))
    state = str(getattr(trial, "state", "UNKNOWN"))
    return {
        "trial_number": getattr(trial, "number", None),
        "state": state,
        "value": getattr(trial, "value", None),
        "step": getattr(trial, "params", {}).get("step"),
        "lr": getattr(trial, "params", {}).get("lr"),
        "b_size": getattr(trial, "params", {}).get("b_size"),
        "c_size": attrs.get("c_size"),
        "d_size": getattr(trial, "params", {}).get("d_size"),
        "common_ok_count": attrs.get("common_ok_count"),
        "common_avg_baseline": attrs.get("common_avg_baseline"),
        "common_avg_step": attrs.get("common_avg_step"),
        "avg_delta_pp": attrs.get("avg_delta_pp"),
        "avg_accuracy_ok": attrs.get("avg_accuracy_ok"),
        "ok_count": attrs.get("ok_count"),
        "failed_count": attrs.get("failed_count"),
        "wall_seconds": attrs.get("wall_seconds"),
        "cached": attrs.get("cached"),
        "out_dir": attrs.get("out_dir"),
        "summary_path": attrs.get("summary_path"),
        "all_csv_path": attrs.get("all_csv_path"),
        "command_json": attrs.get("command_json"),
        "error": attrs.get("error"),
    }


def build_best_payload(best_trial: Any) -> dict[str, Any]:
    record = trial_to_record(best_trial)
    return {
        "trial_number": record["trial_number"],
        "objective": record["value"],
        "step": record["step"],
        "lr": record["lr"],
        "b_size": record["b_size"],
        "c_size": record["c_size"],
        "d_size": record["d_size"],
        "common_ok_count": record["common_ok_count"],
        "common_avg_baseline": record["common_avg_baseline"],
        "common_avg_step": record["common_avg_step"],
        "avg_delta_pp": record["avg_delta_pp"],
        "avg_accuracy_ok": record["avg_accuracy_ok"],
        "ok_count": record["ok_count"],
        "failed_count": record["failed_count"],
        "wall_seconds": record["wall_seconds"],
        "out_dir": record["out_dir"],
        "summary_path": record["summary_path"],
        "all_csv_path": record["all_csv_path"],
    }


def write_study_outputs(study: Any, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    trials = [trial_to_record(trial) for trial in getattr(study, "trials", [])]
    trials_df = pd.DataFrame(trials)
    if not trials_df.empty and "trial_number" in trials_df.columns:
        trials_df = trials_df.sort_values("trial_number")
    trials_df.to_csv(output_root / "trials.csv", index=False)

    completed_trials = [
        trial
        for trial in getattr(study, "trials", [])
        if str(getattr(trial, "state", "")) in {"TrialState.COMPLETE", "COMPLETE"}
    ]
    if completed_trials:
        best_payload = build_best_payload(study.best_trial)
        (output_root / "best_params.json").write_text(
            json.dumps(
                {
                    "step": best_payload["step"],
                    "lr": best_payload["lr"],
                    "b_size": best_payload["b_size"],
                    "c_size": best_payload["c_size"],
                    "d_size": best_payload["d_size"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_root / "best_trial_summary.json").write_text(
            json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        best_line = f"best_objective: {best_payload['objective']:.10f}"
        best_trial_line = f"best_trial_number: {best_payload['trial_number']}"
    else:
        (output_root / "best_params.json").write_text("{}\n", encoding="utf-8")
        (output_root / "best_trial_summary.json").write_text("{}\n", encoding="utf-8")
        best_line = "best_objective: (none)"
        best_trial_line = "best_trial_number: (none)"

    lines = [
        f"study_name: {getattr(study, 'study_name', '(unknown)')}",
        f"total_trials: {len(getattr(study, 'trials', []))}",
        f"completed_trials: {len(completed_trials)}",
        best_trial_line,
        best_line,
    ]
    (output_root / "study_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_objective(
    *,
    args: argparse.Namespace,
    forwarded_args: Sequence[str],
    baseline_ok_df: pd.DataFrame,
) -> Any:
    def objective(trial: Any) -> float:
        step = int(trial.suggest_categorical("step", STEP_CHOICES))
        lr = float(trial.suggest_float("lr", 2e-6, 1e-4, log=True))
        b_size, c_size, d_size = sample_ttt_fractions(trial)
        out_dir = args.output_root / "runs" / format_out_dir_name(step, lr, b_size, c_size, d_size)
        command = build_trial_command(
            python_bin=args.python_bin,
            forwarded_args=forwarded_args,
            out_dir=out_dir,
            step=step,
            lr=lr,
            b_size=b_size,
            c_size=c_size,
            d_size=d_size,
        )

        trial.set_user_attr("c_size", c_size)
        trial.set_user_attr("out_dir", str(out_dir))
        trial.set_user_attr("command_json", json.dumps(command, ensure_ascii=False))

        try:
            cached = ensure_trial_outputs(out_dir=out_dir, command=command, cwd=REPO_ROOT)
            metrics = evaluate_trial_outputs(baseline_ok_df, out_dir)
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

        trial.set_user_attr("cached", cached)
        for key, value in metrics.items():
            if key == "objective":
                continue
            trial.set_user_attr(key, value)
        return float(metrics["objective"])

    return objective


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded_args = parse_args(argv)
    if not args.baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {args.baseline_csv}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.storage is None:
        args.storage = default_storage_uri(args.output_root, args.study_name)

    baseline_ok_df = load_ok_accuracy_frame(args.baseline_csv).rename(
        columns={"accuracy": "baseline_accuracy"}
    )
    optuna = load_optuna()
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=args.load_if_exists,
    )
    objective = make_objective(
        args=args,
        forwarded_args=forwarded_args,
        baseline_ok_df=baseline_ok_df,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        catch=(Exception,),
    )
    write_study_outputs(study, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

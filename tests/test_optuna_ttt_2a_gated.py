from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "optuna_ttt_2a_gated.py"


def load_module():
    module_name = "module_optuna_ttt_2a_gated"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


search = load_module()


class FakeFractionTrial:
    def __init__(self, *, b_size: float, d_size: float):
        self.params: dict[str, float] = {}
        self._values = {"b_size": b_size, "d_size": d_size}

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        value = self._values[name]
        assert low <= value <= high
        self.params[name] = value
        return value


class FakeTrial:
    def __init__(self):
        self.number = 0
        self.params: dict[str, float | int] = {}
        self.user_attrs: dict[str, object] = {}
        self.state = "RUNNING"
        self.value = None

    def suggest_categorical(self, name: str, choices):
        value = 20
        self.params[name] = value
        assert value in choices
        return value

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if name == "lr":
            value = 2e-5
        elif name == "b_size":
            value = 0.6
        elif name == "d_size":
            value = 0.1
        else:
            raise AssertionError(f"Unexpected suggest_float call for {name}")
        assert low <= value <= high
        self.params[name] = value
        return value

    def set_user_attr(self, key: str, value) -> None:
        self.user_attrs[key] = value


class FakeStudy:
    def __init__(self, study_name: str):
        self.study_name = study_name
        self.trials: list[FakeTrial] = []
        self.best_trial: FakeTrial | None = None

    def optimize(self, objective, n_trials: int, timeout: int | None, catch):
        for trial_number in range(n_trials):
            trial = FakeTrial()
            trial.number = trial_number
            try:
                trial.value = objective(trial)
                trial.state = "COMPLETE"
                if self.best_trial is None or float(trial.value) > float(self.best_trial.value):
                    self.best_trial = trial
            except catch as exc:
                trial.state = "FAIL"
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            self.trials.append(trial)


class FakeSampler:
    def __init__(self, seed: int):
        self.seed = seed


class FakeOptuna:
    class samplers:
        TPESampler = FakeSampler

    @staticmethod
    def create_study(study_name: str, storage: str, direction: str, sampler, load_if_exists: bool):
        assert direction == "maximize"
        assert isinstance(sampler, FakeSampler)
        assert load_if_exists is True
        assert storage.startswith("sqlite:///")
        return FakeStudy(study_name=study_name)


def write_trial_outputs(out_dir: Path, *, accuracies: dict[str, float], avg_accuracy_ok: float, wall_seconds: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"dataset_name": name, "accuracy": accuracy, "status": "ok"}
            for name, accuracy in accuracies.items()
        ]
    ).to_csv(out_dir / "all_classification_results.csv", index=False)
    (out_dir / "summary.txt").write_text(
        f"avg_accuracy_ok: {avg_accuracy_ok:.6f}\nwall_seconds: {wall_seconds:.3f}\n",
        encoding="utf-8",
    )


def test_format_out_dir_name_contains_search_params():
    out_dir = search.format_out_dir_name(20, 2e-5, 0.6, 0.3, 0.1)

    assert out_dir == "ae82_step20_lr2.00e-05_b0.6000_c0.3000_d0.1000"


def test_sample_ttt_fractions_respects_constraints():
    trial = FakeFractionTrial(b_size=0.61, d_size=0.12)

    b_size, c_size, d_size = search.sample_ttt_fractions(trial)

    assert abs(b_size + c_size + d_size - 1.0) < 1e-10
    assert 0.5 <= b_size <= 0.7
    assert 0.2 <= c_size <= 0.3
    assert 0.05 <= d_size <= 0.15


def test_evaluate_trial_outputs_uses_common_ok_datasets(tmp_path):
    baseline_ok = pd.DataFrame(
        [
            {"dataset_name": "alpha", "baseline_accuracy": 0.80},
            {"dataset_name": "beta", "baseline_accuracy": 0.70},
            {"dataset_name": "gamma", "baseline_accuracy": 0.60},
        ]
    )
    out_dir = tmp_path / "trial"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"dataset_name": "alpha", "accuracy": 0.85, "status": "ok"},
            {"dataset_name": "beta", "accuracy": 0.65, "status": "ok"},
            {"dataset_name": "delta", "accuracy": 0.90, "status": "ok"},
            {"dataset_name": "omega", "accuracy": 0.00, "status": "fail"},
        ]
    ).to_csv(out_dir / "all_classification_results.csv", index=False)
    (out_dir / "summary.txt").write_text(
        "avg_accuracy_ok: 0.800000\nwall_seconds: 12.500\n",
        encoding="utf-8",
    )

    metrics = search.evaluate_trial_outputs(baseline_ok, out_dir)

    assert metrics["common_ok_count"] == 2
    assert metrics["common_avg_baseline"] == 0.75
    assert metrics["common_avg_step"] == 0.75
    assert metrics["objective"] == 0.0
    assert metrics["avg_delta_pp"] == 0.0
    assert metrics["avg_accuracy_ok"] == 0.8
    assert metrics["ok_count"] == 3
    assert metrics["failed_count"] == 1
    assert metrics["wall_seconds"] == 12.5


def test_ensure_trial_outputs_reuses_valid_cache(tmp_path):
    out_dir = tmp_path / "cached"
    write_trial_outputs(
        out_dir,
        accuracies={"alpha": 0.8},
        avg_accuracy_ok=0.8,
        wall_seconds=1.0,
    )

    def fail_runner(*args, **kwargs):
        raise AssertionError("runner should not be called when cache is valid")

    cached = search.ensure_trial_outputs(
        out_dir=out_dir,
        command=["python", "TTT_2A_gated.py"],
        runner=fail_runner,
    )

    assert cached is True


def test_build_trial_command_contains_trial_specific_args(tmp_path):
    out_dir = tmp_path / "run"

    command = search.build_trial_command(
        python_bin="python",
        forwarded_args=["--gpus", "0,1", "--max-datasets", "5"],
        out_dir=out_dir,
        step=40,
        lr=1e-5,
        b_size=0.6,
        c_size=0.3,
        d_size=0.1,
    )

    assert command[:2] == ["python", str(search.TTT_SCRIPT)]
    assert "--gpus" in command
    assert "--max-datasets" in command
    assert command[command.index("--ttt-steps") + 1] == "40"
    assert command[command.index("--ttt-lr") + 1] == "1e-05"
    assert command[command.index("--ttt-b-fraction") + 1] == "0.6000000000"
    assert command[command.index("--ttt-c-fraction") + 1] == "0.3000000000"
    assert command[command.index("--ttt-d-fraction") + 1] == "0.1000000000"
    assert command[command.index("--out-dir") + 1] == str(out_dir)


def test_main_writes_study_outputs_with_fake_optuna(tmp_path, monkeypatch):
    baseline_csv = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {"dataset_name": "alpha", "accuracy": 0.80, "status": "ok"},
            {"dataset_name": "beta", "accuracy": 0.70, "status": "ok"},
        ]
    ).to_csv(baseline_csv, index=False)
    output_root = tmp_path / "optuna_out"

    monkeypatch.setattr(search, "load_optuna", lambda: FakeOptuna)

    def fake_ensure_trial_outputs(*, out_dir: Path, command, cwd, runner=None):
        write_trial_outputs(
            out_dir,
            accuracies={"alpha": 0.84, "beta": 0.74},
            avg_accuracy_ok=0.79,
            wall_seconds=9.5,
        )
        return False

    monkeypatch.setattr(search, "ensure_trial_outputs", fake_ensure_trial_outputs)

    exit_code = search.main(
        [
            "--baseline-csv",
            str(baseline_csv),
            "--output-root",
            str(output_root),
            "--study-name",
            "demo_study",
            "--n-trials",
            "1",
            "--max-datasets",
            "5",
        ]
    )

    assert exit_code == 0
    trials_df = pd.read_csv(output_root / "trials.csv")
    assert len(trials_df) == 1
    assert trials_df.loc[0, "state"] == "COMPLETE"
    assert trials_df.loc[0, "common_ok_count"] == 2
    assert abs(float(trials_df.loc[0, "value"]) - 0.04) < 1e-12
    best_params = json.loads((output_root / "best_params.json").read_text(encoding="utf-8"))
    assert best_params["step"] == 20
    assert best_params["lr"] == 2e-5
    assert best_params["b_size"] == 0.6
    assert best_params["c_size"] == 0.3
    assert best_params["d_size"] == 0.1
    best_summary = json.loads((output_root / "best_trial_summary.json").read_text(encoding="utf-8"))
    assert abs(best_summary["objective"] - 0.04) < 1e-12
    summary_text = (output_root / "study_summary.txt").read_text(encoding="utf-8")
    assert "study_name: demo_study" in summary_text
    assert "completed_trials: 1" in summary_text

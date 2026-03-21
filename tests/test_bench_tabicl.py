from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import types

import numpy as np

import bench_tabicl


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTabICLClassifier:
    init_kwargs: list[dict] = []
    fit_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    predict_shapes: list[tuple[int, ...]] = []

    def __init__(self, **kwargs):
        type(self).init_kwargs.append(dict(kwargs))
        self.kwargs = kwargs
        self.model_ = object()

    def fit(self, X, y):
        type(self).fit_shapes.append((tuple(np.asarray(X).shape), tuple(np.asarray(y).shape)))
        return self

    def predict(self, X):
        X = np.asarray(X)
        type(self).predict_shapes.append(tuple(X.shape))
        return X[:, 0].astype(int)

    @classmethod
    def reset(cls):
        cls.init_kwargs.clear()
        cls.fit_shapes.clear()
        cls.predict_shapes.clear()


def install_fake_classifier(monkeypatch) -> None:
    FakeTabICLClassifier.reset()

    tabicl_module = types.ModuleType("tabicl")
    sklearn_module = types.ModuleType("tabicl.sklearn")
    classifier_module = types.ModuleType("tabicl.sklearn.classifier")
    classifier_module.TabICLClassifier = FakeTabICLClassifier
    sklearn_module.classifier = classifier_module
    tabicl_module.sklearn = sklearn_module

    monkeypatch.setitem(sys.modules, "tabicl", tabicl_module)
    monkeypatch.setitem(sys.modules, "tabicl.sklearn", sklearn_module)
    monkeypatch.setitem(sys.modules, "tabicl.sklearn.classifier", classifier_module)


def write_split(dataset_dir: Path, split: str, X_num, y, X_cat=None) -> None:
    np.save(dataset_dir / f"N_{split}.npy", np.asarray(X_num))
    if X_cat is not None:
        np.save(dataset_dir / f"C_{split}.npy", np.asarray(X_cat, dtype=object))
    np.save(dataset_dir / f"y_{split}.npy", np.asarray(y))


def make_standard_dataset(
    root: Path,
    name: str,
    *,
    task_type: str = "binclass",
    with_val: bool = False,
    add_nan: bool = False,
) -> Path:
    dataset_dir = root / name
    dataset_dir.mkdir()
    (dataset_dir / "info.json").write_text(f'{{"task_type": "{task_type}"}}', encoding="utf-8")

    train_num = np.array([[0.0, 10.0], [1.0, 11.0], [0.0, 12.0]], dtype=float)
    test_num = np.array([[1.0, 20.0], [0.0, 21.0]], dtype=float)
    if add_nan:
        train_num[1, 1] = np.nan

    train_cat = np.array([["low"], ["high"], ["low"]], dtype=object)
    test_cat = np.array([["high"], ["low"]], dtype=object)

    write_split(dataset_dir, "train", train_num, np.array([0, 1, 0]), train_cat)
    write_split(dataset_dir, "test", test_num, np.array([1, 0]), test_cat)

    if with_val:
        val_num = np.array([[1.0, 30.0], [0.0, 31.0]], dtype=float)
        val_cat = np.array([["high"], ["low"]], dtype=object)
        write_split(dataset_dir, "val", val_num, np.array([1, 0]), val_cat)

    return dataset_dir


def build_classifier_kwargs(tmp_path: Path, extra_args: list[str] | None = None) -> dict:
    parser = bench_tabicl.build_arg_parser()
    args = parser.parse_args(
        ["--data-root", str(tmp_path), "--outdir", str(tmp_path / "out"), "--device", "cpu"] + (extra_args or [])
    )
    return bench_tabicl.classifier_kwargs_from_args(args)


def test_help_runs_without_optional_nvml():
    result = subprocess.run(
        [sys.executable, "bench_tabicl.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--checkpoint-version" in result.stdout
    assert "--allow-auto-download" in result.stdout
    assert "--debug" in result.stdout


def test_classifier_kwargs_from_args_align_with_readme_defaults(tmp_path):
    parser = bench_tabicl.build_arg_parser()

    default_args = parser.parse_args(["--data-root", str(tmp_path)])
    default_kwargs = bench_tabicl.classifier_kwargs_from_args(default_args)
    assert default_kwargs["model_path"] is None
    assert default_kwargs["checkpoint_version"] == bench_tabicl.DEFAULT_CHECKPOINT_VERSION
    assert default_kwargs["device"] is None
    assert default_kwargs["allow_auto_download"] is True
    assert default_kwargs["kv_cache"] is False
    assert default_kwargs["batch_size"] == 8
    assert default_kwargs["norm_methods"] is None

    custom_args = parser.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--device",
            "cuda",
            "--device-id",
            "2",
            "--n-estimators",
            "16",
            "--norm-methods",
            "none",
            "power",
            "--feat-shuffle-method",
            "random",
            "--class-shuffle-method",
            "latin",
            "--outlier-threshold",
            "3.5",
            "--softmax-temperature",
            "0.75",
            "--no-average-logits",
            "--no-support-many-classes",
            "--batch-size",
            "none",
            "--kv-cache",
            "repr",
            "--use-amp",
            "true",
            "--use-fa3",
            "false",
            "--offload-mode",
            "disk",
            "--disk-offload-dir",
            "/tmp/offload",
            "--random-state",
            "7",
            "--n-jobs",
            "-1",
            "--verbose",
            "--no-allow-auto-download",
        ]
    )
    custom_kwargs = bench_tabicl.classifier_kwargs_from_args(custom_args)
    assert custom_kwargs == {
        "n_estimators": 16,
        "norm_methods": ["none", "power"],
        "feat_shuffle_method": "random",
        "class_shuffle_method": "latin",
        "outlier_threshold": 3.5,
        "softmax_temperature": 0.75,
        "average_logits": False,
        "support_many_classes": False,
        "batch_size": None,
        "kv_cache": "repr",
        "model_path": None,
        "allow_auto_download": False,
        "checkpoint_version": bench_tabicl.DEFAULT_CHECKPOINT_VERSION,
        "device": "cuda:2",
        "use_amp": True,
        "use_fa3": False,
        "offload_mode": "disk",
        "disk_offload_dir": "/tmp/offload",
        "random_state": 7,
        "n_jobs": -1,
        "verbose": True,
    }


def test_evaluate_datasets_writes_outputs_and_skips_nonstandard_layouts(tmp_path, monkeypatch):
    install_fake_classifier(monkeypatch)

    data_root = tmp_path / "data"
    data_root.mkdir()
    make_standard_dataset(data_root, "alpha", with_val=True)
    make_standard_dataset(data_root, "beta_nan", add_nan=True)
    make_standard_dataset(data_root, "gamma_regression", task_type="regression")

    single_file_dir = data_root / "single_file_only"
    single_file_dir.mkdir()
    np.save(single_file_dir / "dataset.npy", np.array([[0, 1], [1, 2]]))

    outdir = tmp_path / "out"
    results = bench_tabicl.evaluate_datasets(
        classifier_kwargs=build_classifier_kwargs(tmp_path),
        data_root=data_root,
        outdir=outdir,
        merge_val=False,
        coerce_numeric=True,
        device_id=0,
    )

    assert [result.name for result in results] == ["alpha", "beta_nan"]
    assert all(result.accuracy == 1.0 for result in results)
    assert results[0].val_accuracy == 1.0
    assert results[0].peak_vram_mib is None
    assert results[1].val_accuracy is None

    assert len(FakeTabICLClassifier.init_kwargs) == 2
    assert all(kwargs["device"] == "cpu" for kwargs in FakeTabICLClassifier.init_kwargs)

    detailed_text = (outdir / "talent_detailed.txt").read_text(encoding="utf-8")
    assert "dataset\taccuracy\tfit_time_s\tpredict_time_s\ttotal_time_s\tpeak_vram_mib\tval_accuracy\tval_time_s" in detailed_text
    assert "alpha" in detailed_text
    assert "beta_nan" in detailed_text
    assert "gamma_regression" not in detailed_text
    assert "single_file_only" not in detailed_text
    assert "N/A" in detailed_text

    summary_text = (outdir / "talent_summary.txt").read_text(encoding="utf-8")
    assert "Total datasets: 2" in summary_text
    assert "Datasets with NaN values: 1" in summary_text
    assert "List (NaN datasets): beta_nan" in summary_text
    assert "Average validation accuracy:" in summary_text


def test_merge_val_extends_training_split(tmp_path, monkeypatch):
    install_fake_classifier(monkeypatch)

    data_root = tmp_path / "data"
    data_root.mkdir()
    make_standard_dataset(data_root, "alpha", with_val=True)

    bench_tabicl.evaluate_datasets(
        classifier_kwargs=build_classifier_kwargs(tmp_path),
        data_root=data_root,
        outdir=tmp_path / "out",
        merge_val=True,
        coerce_numeric=True,
        device_id=0,
    )

    assert FakeTabICLClassifier.fit_shapes[0][0][0] == 5

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

import compare_methods_benchmark as cmb


def write_split(dataset_dir: Path, split: str, X_num, y, X_cat=None) -> None:
    np.save(dataset_dir / f"N_{split}.npy", np.asarray(X_num, dtype=float))
    if X_cat is not None:
        np.save(dataset_dir / f"C_{split}.npy", np.asarray(X_cat, dtype=object))
    np.save(dataset_dir / f"y_{split}.npy", np.asarray(y))


def make_dataset(root: Path, name: str, *, task_type: str = "binclass", with_val: bool = False) -> Path:
    dataset_dir = root / name
    dataset_dir.mkdir()
    (dataset_dir / "info.json").write_text(f'{{"task_type": "{task_type}"}}', encoding="utf-8")
    write_split(
        dataset_dir,
        "train",
        np.array([[0.0, 10.0], [1.0, 11.0], [0.0, 12.0]]),
        np.array([0, 1, 0]),
        np.array([["low"], ["high"], ["low"]], dtype=object),
    )
    write_split(
        dataset_dir,
        "test",
        np.array([[1.0, 20.0], [0.0, 21.0]]),
        np.array([1, 0]),
        np.array([["high"], ["low"]], dtype=object),
    )
    if with_val:
        write_split(
            dataset_dir,
            "val",
            np.array([[1.0, 30.0], [0.0, 31.0]]),
            np.array([1, 0]),
            np.array([["high"], ["low"]], dtype=object),
        )
    return dataset_dir


class EchoFirstColumnAdapter(cmb.MethodAdapter):
    method_name = "echo"
    backend = "test"
    model_version = "test"

    def fit_predict(self, X_train, y_train, X_test, categorical_feature_indices, dataset_name):
        return cmb.PredictionResult(
            y_pred=X_test.iloc[:, 0].to_numpy().astype(int),
            fit_seconds=0.1,
            predict_seconds=0.2,
        )


def test_load_classification_dataset_matches_benchmark_shape_and_val_merge(tmp_path):
    dataset_dir = make_dataset(tmp_path, "alpha", with_val=True)

    loaded = cmb.load_classification_dataset(dataset_dir)

    assert loaded.dataset_name == "alpha"
    assert loaded.task_type == "binclass"
    assert loaded.X_train.shape == (5, 3)
    assert loaded.X_test.shape == (2, 3)
    assert loaded.n_val == 2
    assert loaded.categorical_feature_indices == [2]
    assert loaded.n_classes == 2


def test_result_frame_keeps_benchmark_columns_first(tmp_path):
    dataset_dir = make_dataset(tmp_path, "alpha")
    adapter = EchoFirstColumnAdapter(types.SimpleNamespace(), device="cpu")

    row = cmb.evaluate_one_dataset(adapter, dataset_dir, worker_id=0, gpu_id=0)
    frame = cmb.rows_to_frame([row])

    assert frame.columns.tolist()[: len(cmb.BASE_RESULT_COLUMNS)] == cmb.BASE_RESULT_COLUMNS
    assert frame.columns.tolist()[len(cmb.BASE_RESULT_COLUMNS) :] == cmb.EXTRA_RESULT_COLUMNS
    assert frame.loc[0, "status"] == "ok"
    assert frame.loc[0, "accuracy"] == 1.0
    assert frame.loc[0, "method_name"] == "echo"


def test_missing_tabpfn_dependency_becomes_fail_row(tmp_path, monkeypatch):
    dataset_dir = make_dataset(tmp_path, "alpha")
    monkeypatch.setitem(sys.modules, "tabpfn", None)
    args = types.SimpleNamespace(tabpfn25_model_path=None, tabpfnv2_model_path=None)
    adapter = cmb.TabPFNAdapter(args, device="cpu", version_name="v2_5")

    row = cmb.evaluate_one_dataset(adapter, dataset_dir, worker_id=0, gpu_id=0)

    assert row.status == "fail"
    assert "Failed to import tabpfn" in (row.error or "")
    assert row.method_name == "tabpfn25"


def test_resolve_workers_and_gpu_ids_auto_uses_detected_gpus(monkeypatch):
    monkeypatch.setattr(cmb, "detect_gpu_count", lambda: 3)
    args = types.SimpleNamespace(gpus="auto", workers=None)

    workers, gpu_ids = cmb.resolve_workers_and_gpu_ids(args)

    assert workers == 3
    assert gpu_ids == [0, 1, 2]


def test_resolve_workers_and_gpu_ids_accepts_explicit_ids():
    args = types.SimpleNamespace(gpus="2,4", workers=2)

    workers, gpu_ids = cmb.resolve_workers_and_gpu_ids(args)

    assert workers == 2
    assert gpu_ids == [2, 4]


def test_write_method_outputs_include_summary_and_comparison(tmp_path):
    rows = [
        cmb.ResultRow(
            dataset_name="alpha",
            dataset_dir="data/alpha",
            task_type="binclass",
            n_train=3,
            n_val=0,
            n_test=2,
            n_features=2,
            n_classes=2,
            accuracy=0.5,
            fit_seconds=1.0,
            predict_seconds=2.0,
            status="ok",
            error=None,
            method_name="m1",
            backend="test",
            model_version="v",
            model_path=None,
            categorical_feature_indices="[]",
            worker_id=0,
            gpu_id=0,
        )
    ]
    frame = cmb.rows_to_frame(rows)
    summary_path = tmp_path / "summary.txt"
    cmb.write_method_summary(summary_path, frame, [tmp_path / "alpha"], wall_seconds=3.0)
    summary = cmb.summarize_method_frame("m1", frame, wall_seconds=3.0)
    summary_df = pd.DataFrame([summary])
    md_path = tmp_path / "method_comparison.md"
    cmb.write_comparison_markdown(md_path, summary_df)

    assert "avg_accuracy_ok: 0.500000" in summary_path.read_text(encoding="utf-8")
    assert "| m1 | 1 | 0 | 0 | 0.500000 |" in md_path.read_text(encoding="utf-8")

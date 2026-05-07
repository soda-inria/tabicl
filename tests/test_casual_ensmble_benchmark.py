from __future__ import annotations

import numpy as np

from casual_ensmble_benchmark import (
    PCMarkovBlanketFeatureSelector,
    PartialFeatureShuffleEnsembleGenerator,
    build_arg_parser,
    build_common_model_kwargs,
)


def test_partial_feature_shuffle_only_permutates_selected_columns():
    X = np.array(
        [
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [1.0, 11.0, 21.0, 31.0, 41.0],
            [2.0, 12.0, 22.0, 32.0, 42.0],
        ]
    )
    y = np.array([0, 1, 0])

    generator = PartialFeatureShuffleEnsembleGenerator(
        classification=True,
        n_estimators=4,
        norm_methods="none",
        feat_shuffle_method="shift",
        class_shuffle_method="none",
        shuffle_feature_indices=[1, 3],
        random_state=0,
    ).fit(X, y)

    patterns = list(generator.feature_shuffles_.values())[0]
    assert patterns
    for pattern in patterns:
        assert pattern[0] == 0
        assert pattern[2] == 2
        assert pattern[4] == 4
        assert sorted([pattern[1], pattern[3]]) == [1, 3]


def test_partial_feature_shuffle_remaps_after_constant_feature_filter():
    X = np.array(
        [
            [0.0, 1.0, 20.0, 30.0],
            [1.0, 1.0, 21.0, 31.0],
            [2.0, 1.0, 22.0, 32.0],
        ]
    )
    y = np.array([0, 1, 0])

    generator = PartialFeatureShuffleEnsembleGenerator(
        classification=True,
        n_estimators=4,
        norm_methods="none",
        feat_shuffle_method="shift",
        class_shuffle_method="none",
        shuffle_feature_indices=[1, 3],
        random_state=0,
    ).fit(X, y)

    assert generator.unique_filter_.features_to_keep_.tolist() == [True, False, True, True]
    assert generator.shuffle_feature_indices_filtered_.tolist() == [2]
    assert list(generator.feature_shuffles_.values())[0] == [[0, 1, 2]]


def test_pc_markov_blanket_selector_is_deterministic_and_selects_signal_feature():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=400)
    X = rng.normal(size=(400, 8))
    X[:, 0] = y + rng.normal(scale=0.05, size=400)
    X[:, 3] = 0.5 * y + rng.normal(scale=0.1, size=400)

    selector_a = PCMarkovBlanketFeatureSelector(
        alpha=0.01,
        max_samples=200,
        max_candidates=6,
        max_cond_set=1,
        min_features=1,
        top_k=3,
        random_state=42,
    ).fit(X, y)
    selector_b = PCMarkovBlanketFeatureSelector(
        alpha=0.01,
        max_samples=200,
        max_candidates=6,
        max_cond_set=1,
        min_features=1,
        top_k=3,
        random_state=42,
    ).fit(X, y)

    assert 0 in selector_a.selected_indices_.tolist()
    assert selector_a.selected_indices_.tolist() == selector_b.selected_indices_.tolist()
    assert selector_a.selector_seconds_ >= 0.0
    assert selector_a.status_.startswith("ok selected=")


def test_causal_cli_kwargs_are_passed_to_wrapper_config(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--causal-alpha",
            "0.02",
            "--causal-max-samples",
            "123",
            "--causal-max-candidates",
            "17",
            "--causal-max-cond-set",
            "1",
            "--causal-min-features",
            "3",
            "--causal-top-k",
            "9",
        ]
    )

    kwargs = build_common_model_kwargs(args)

    assert kwargs["causal_alpha"] == 0.02
    assert kwargs["causal_max_samples"] == 123
    assert kwargs["causal_max_candidates"] == 17
    assert kwargs["causal_max_cond_set"] == 1
    assert kwargs["causal_min_features"] == 3
    assert kwargs["causal_top_k"] == 9


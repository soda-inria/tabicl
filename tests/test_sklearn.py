import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICLClassifier, TabICLRegressor


# n_estimators=2 ensures the full preprocessing and ensembling pipeline is tested:
# n_estimators=1 skips shuffling and uses only one norm method, while n_estimators=2
# exercises feature/class shuffling, multiple normalization methods, and ensemble averaging.
@parametrize_with_checks([TabICLClassifier(n_estimators=2), TabICLRegressor(n_estimators=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


class TestClassifierKVCache:
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        clf = TabICLClassifier(n_estimators=2)
        clf.fit(X_train, y_train)
        pred_no_cache = clf.predict_proba(X_test)

        clf_cached = TabICLClassifier(n_estimators=2, kv_cache=kv_cache)
        clf_cached.fit(X_train, y_train)
        pred_cached = clf_cached.predict_proba(X_test)

        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)


class TestRegressorKVCache:
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        reg = TabICLRegressor(n_estimators=2)
        reg.fit(X_train, y_train)
        pred_no_cache = reg.predict(X_test)

        reg_cached = TabICLRegressor(n_estimators=2, kv_cache=kv_cache)
        reg_cached.fit(X_train, y_train)
        pred_cached = reg_cached.predict(X_test)

        # Relaxed tolerance: kv cache changes float32 computation order
        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)


class TestClassifierMultiTarget:
    def test_multi_target_output_shape(self):
        """1D y -> proba (n, nc), pred (n,). 2D y -> proba is list of 3, pred (n, 3)."""
        rng = np.random.RandomState(42)
        X, y0 = make_classification(n_samples=50, n_features=5, random_state=42)
        y_multi = np.column_stack([y0, rng.randint(0, 3, 50), rng.randint(0, 2, 50)])
        X_train, X_test = X[:40], X[40:]

        # 1D y
        clf_1d = TabICLClassifier(n_estimators=2)
        clf_1d.fit(X_train, y0[:40])
        proba_1d = clf_1d.predict_proba(X_test)
        pred_1d = clf_1d.predict(X_test)
        assert proba_1d.shape == (10, 2)
        assert pred_1d.shape == (10,)

        # 2D y
        clf_2d = TabICLClassifier(n_estimators=2)
        clf_2d.fit(X_train, y_multi[:40])
        proba_2d = clf_2d.predict_proba(X_test)
        pred_2d = clf_2d.predict(X_test)
        assert isinstance(proba_2d, list)
        assert len(proba_2d) == 3
        assert proba_2d[0].shape == (10, 2)  # binary
        assert proba_2d[1].shape == (10, 3)  # 3-class
        assert proba_2d[2].shape == (10, 2)  # binary
        assert pred_2d.shape == (10, 3)

    def test_multi_target_fit_attributes(self):
        """2D: n_targets_==3, len(y_encoders_)==3. 1D: n_targets_==1, has y_encoder_."""
        rng = np.random.RandomState(42)
        X, y0 = make_classification(n_samples=50, n_features=5, random_state=42)
        y_multi = np.column_stack([y0, rng.randint(0, 3, 50), rng.randint(0, 2, 50)])

        # 2D y
        clf_multi = TabICLClassifier(n_estimators=2)
        clf_multi.fit(X, y_multi)
        assert clf_multi.n_targets_ == 3
        assert len(clf_multi.y_encoders_) == 3
        assert clf_multi.y_train_encoded_.shape == (50, 3)

        # 1D y
        clf_single = TabICLClassifier(n_estimators=2)
        clf_single.fit(X, y0)
        assert clf_single.n_targets_ == 1
        assert hasattr(clf_single, "y_encoder_")
        assert not hasattr(clf_single, "y_encoders_")

    def test_multi_target_matches_single_target(self):
        """Multi-target and single-target classifiers should largely agree on labels."""
        rng = np.random.RandomState(42)
        X, y0 = make_classification(n_samples=50, n_features=5, random_state=42)
        y_multi = np.column_stack([y0, rng.randint(0, 3, 50), rng.randint(0, 2, 50)])
        X_train, X_test = X[:40], X[40:]

        # Multi-target
        clf_multi = TabICLClassifier(n_estimators=2, random_state=42)
        clf_multi.fit(X_train, y_multi[:40])
        pred_multi = clf_multi.predict(X_test)

        # Single-target per column
        for t in range(3):
            clf_single = TabICLClassifier(n_estimators=2, random_state=42)
            clf_single.fit(X_train, y_multi[:40, t])
            pred_single = clf_single.predict(X_test)
            agreement = np.mean(pred_multi[:, t] == pred_single)
            assert agreement >= 0.5, (
                f"Target {t}: only {agreement:.0%} agreement between multi- and single-target"
            )

    def test_multi_target_predict(self):
        """Labels belong to correct per-target classes."""
        rng = np.random.RandomState(42)
        X, y0 = make_classification(n_samples=50, n_features=5, random_state=42)
        y_multi = np.column_stack([y0, rng.randint(0, 3, 50), rng.randint(0, 2, 50)])
        X_train, X_test = X[:40], X[40:]

        clf = TabICLClassifier(n_estimators=2)
        clf.fit(X_train, y_multi[:40])
        pred = clf.predict(X_test)
        for t in range(3):
            assert set(pred[:, t]).issubset(set(y_multi[:40, t]))


class TestRegressorMultiTarget:
    def test_multi_target_matches_single_target(self):
        """Multi-target predictions should match individual single-target models."""
        X, y = make_regression(n_samples=50, n_features=5, n_targets=3, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]

        # Fit separate single-target models
        single_preds = []
        for t in range(3):
            reg = TabICLRegressor(n_estimators=2, random_state=42)
            reg.fit(X_train, y_train[:, t])
            single_preds.append(reg.predict(X_test))

        # Fit one multi-target model
        reg_multi = TabICLRegressor(n_estimators=2, random_state=42)
        reg_multi.fit(X_train, y_train)
        multi_preds = reg_multi.predict(X_test)

        for t in range(3):
            np.testing.assert_allclose(multi_preds[:, t], single_preds[t], rtol=1e-4, atol=1e-4)

    def test_multi_target_output_shape(self):
        """Verify output shapes for 1D and 2D y."""
        X, y = make_regression(n_samples=50, n_features=5, n_targets=3, random_state=42)
        X_train, X_test = X[:40], X[40:]

        # 1D y -> (n_test,)
        reg_1d = TabICLRegressor(n_estimators=2, random_state=42)
        reg_1d.fit(X_train, y[:40, 0])
        pred_1d = reg_1d.predict(X_test)
        assert pred_1d.shape == (10,)

        # 2D y -> (n_test, 3)
        reg_2d = TabICLRegressor(n_estimators=2, random_state=42)
        reg_2d.fit(X_train, y[:40])
        pred_2d = reg_2d.predict(X_test)
        assert pred_2d.shape == (10, 3)

    def test_multi_target_fit_attributes(self):
        """Verify fit() sets correct attributes for single vs multi-target."""
        X, y = make_regression(n_samples=50, n_features=5, n_targets=3, random_state=42)

        # 2D y
        reg_multi = TabICLRegressor(n_estimators=2, random_state=42)
        reg_multi.fit(X, y)
        assert reg_multi.n_targets_ == 3
        assert len(reg_multi.y_scalers_) == 3
        assert reg_multi.y_train_scaled_.shape == (50, 3)

        # 1D y
        reg_single = TabICLRegressor(n_estimators=2, random_state=42)
        reg_single.fit(X, y[:, 0])
        assert reg_single.n_targets_ == 1
        assert hasattr(reg_single, "y_scaler_")
        assert not hasattr(reg_single, "y_scalers_")

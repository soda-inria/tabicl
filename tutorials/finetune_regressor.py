"""
Fine-tuning TabICL for regression
=================================

Adapt a pretrained TabICL regressor to a single dataset with
:class:`tabicl.FinetunedTabICLRegressor` (pinball loss on raw quantiles,
same objective the pretrained head was fit with).

.. note::

    A CUDA GPU is recommended for large-scale fine-tuning. Multi-GPU via
    ``torchrun --nproc-per-node=N`` (auto-detected).
"""

# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabicl import FinetunedTabICLRegressor, TabICLRegressor


# %%
# Target: one easy feature (sine), one hard feature (spike)
# ---------------------------------------------------------
def target_fn(x: np.ndarray) -> np.ndarray:
    return 0.8 * np.sin(1.2 * x) + 2.5 * np.exp(-80.0 * (x - 1.0) ** 2)


def make_dataset(n_samples: int = 1_000, random_state: int = 0):
    rng = np.random.RandomState(random_state)
    x = rng.uniform(-3.0, 3.0, size=n_samples)
    y = target_fn(x) + rng.normal(0.0, 0.08, size=n_samples)
    X = x.reshape(-1, 1).astype(np.float32)
    return X, y.astype(np.float32)


X, y = make_dataset(n_samples=1000, random_state=0)

# Split: 40 train (sparse at the spike) / 200 val (early stopping) / 760 test.
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=40, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, train_size=200, random_state=0)

is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0


def _metrics(pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_squared_error(y_true, pred)),
        float(mean_absolute_error(y_true, pred)),
        float(r2_score(y_true, pred)),
    )


# %%
# Baseline — zero-shot TabICL
# ---------------------------
#
# Expected: draws the sine, smears the spike.
base = TabICLRegressor(n_estimators=4, random_state=0)
base.fit(X_train, y_train)
base_pred = base.predict(X_test)
base_mse, base_mae, base_r2 = _metrics(base_pred, y_test)
# Captured for the training-curve reference line in Figure 2.
base_val_mse = float(mean_squared_error(y_val, base.predict(X_val)))


# %%
# Fine-tune
# ---------
#
# ``_HistoryLogger`` below is installed via the same
# ``_make_experiment_logger`` hook ``wandb_kwargs`` uses, to capture per-epoch
# val metrics for Figure 2 without pulling in W&B.
history: dict[str, list[float]] = {
    "epoch": [],
    "val_mse": [],
    "val_mae": [],
    "val_r2": [],
    "train_loss": [],
}


class _HistoryLogger:
    """Record per-epoch validation metrics into ``history``."""

    def setup(self, config):
        del config

    def log_step(self, metrics, step):
        del metrics, step

    def log_epoch(self, metrics, step):
        del step
        history["epoch"].append(int(metrics.get("train/epoch", len(history["epoch"]))) + 1)
        history["val_mse"].append(float(metrics.get("val/mse", np.nan)))
        history["val_mae"].append(float(metrics.get("val/mae", np.nan)))
        history["val_r2"].append(float(metrics.get("val/r2", np.nan)))
        history["train_loss"].append(float(metrics.get("train/mean_loss", np.nan)))

    def finish(self):
        pass


reg = FinetunedTabICLRegressor(
    epochs=60,
    learning_rate=1e-5,
    n_estimators_finetune=2,
    n_estimators_validation=2,
    n_estimators_inference=4,
    early_stopping=True,
    patience=10,
    random_state=0,
    verbose=True,
)
reg._make_experiment_logger = lambda: _HistoryLogger()
reg.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# %%
# Evaluate on the held-out test set
# ---------------------------------
ft_pred = reg.predict(X_test)
ft_mse, ft_mae, ft_r2 = _metrics(ft_pred, y_test)

if is_main_process:
    header = f"{'metric':<10}{'pretrained':>14}{'fine-tuned':>14}{'Δ':>14}"
    rule = "=" * len(header)
    print()
    print(rule)
    print(f"Test-set metrics  (n_train={len(X_train)}, n_test={len(X_test)})")
    print(rule)
    print(header)
    print("-" * len(header))
    print(f"{'MSE ↓':<10}{base_mse:>14.4f}{ft_mse:>14.4f}{ft_mse - base_mse:>+14.4f}")
    print(f"{'MAE ↓':<10}{base_mae:>14.4f}{ft_mae:>14.4f}{ft_mae - base_mae:>+14.4f}")
    print(f"{'R² ↑':<10}{base_r2:>14.4f}{ft_r2:>14.4f}{ft_r2 - base_r2:>+14.4f}")
    print(rule)

# %%
# Figure 1 — Predictions + residuals
# ----------------------------------
#
# Yellow band = spike FWHM; the residual gap should collapse there
# under fine-tuning while the rest of the panel stays flat.
if is_main_process:
    x_grid = np.linspace(-3.0, 3.0, 600).reshape(-1, 1).astype(np.float32)
    alphas = [0.1, 0.5, 0.9]
    q_base = base.predict(x_grid, output_type="quantiles", alphas=alphas)
    q_ft = reg.predict(x_grid, output_type="quantiles", alphas=alphas)

    # Quantiles on the test grid so the residual panel can show the
    # 10–90% band around zero (a calibration read at a glance).
    qt_base = base.predict(X_test, output_type="quantiles", alphas=alphas)
    qt_ft = reg.predict(X_test, output_type="quantiles", alphas=alphas)
    order = np.argsort(X_test.ravel())
    x_sorted = X_test.ravel()[order]

    fig1, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharex=True, sharey="row", constrained_layout=True)
    top = axes[0]
    bot = axes[1]

    # Same emerald as the classifier tutorial for the "ground truth"
    # reference, so the two figures share a visual vocabulary.
    TRUTH_COLOR = "#10b981"
    for ax, title, q, (mse, r2) in [
        (top[0], "Pretrained TabICL", q_base, (base_mse, base_r2)),
        (top[1], "Fine-tuned TabICL", q_ft, (ft_mse, ft_r2)),
    ]:
        ax.fill_between(
            x_grid.ravel(),
            q[:, 0],
            q[:, 2],
            color="#60a5fa",
            alpha=0.25,
            label="10–90 % quantile band",
        )
        ax.plot(x_grid.ravel(), q[:, 1], color="#1d4ed8", lw=2.2, label="predicted median")
        ax.plot(x_grid.ravel(), target_fn(x_grid.ravel()), color=TRUTH_COLOR, lw=2.0, ls="--", label="true target")
        ax.scatter(
            X_train.ravel(),
            y_train,
            c="#b45309",
            edgecolor="white",
            s=32,
            linewidths=0.8,
            label=f"train (n={len(X_train)})",
        )
        # Shade the FWHM of the sharp spike to flag where the failure mode
        # lives. Both panels share the band so the comparison is direct.
        ax.axvspan(0.905, 1.095, color="#fde68a", alpha=0.45, zorder=0, label="spike FWHM")
        ax.set_title(f"{title}\nMSE={mse:.3f}  R²={r2:.3f}", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25)
    top[0].set_ylabel("y", fontsize=11)
    top[0].legend(loc="lower right", framealpha=0.92, fontsize=9)

    # Residual panels: predicted − true, with the 10–90% band relative to
    # the predicted median so the shaded region is centered on zero.
    for ax, title, pred, qt in [
        (bot[0], "Residuals — pretrained", base_pred, qt_base),
        (bot[1], "Residuals — fine-tuned", ft_pred, qt_ft),
    ]:
        residual = pred - y_test
        lo = (qt[:, 0] - qt[:, 1])[order]
        hi = (qt[:, 2] - qt[:, 1])[order]
        ax.fill_between(x_sorted, lo, hi, color="#60a5fa", alpha=0.22, label="10–90 % band (centered)")
        ax.scatter(X_test.ravel(), residual, c="#334155", s=10, alpha=0.65, label="residual (pred − y)")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvspan(0.905, 1.095, color="#fde68a", alpha=0.45, zorder=0, label="spike FWHM")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25)
    bot[0].set_ylabel("residual", fontsize=11)
    bot[0].legend(loc="lower right", framealpha=0.92, fontsize=9)

    # sharey="row" already aligns the two residual panels; just widen the
    # limits symmetrically around zero so max |residual| from either model
    # fits in both.
    y_res_lim = max(np.abs(base_pred - y_test).max(), np.abs(ft_pred - y_test).max())
    bot[0].set_ylim(-y_res_lim * 1.1, y_res_lim * 1.1)

    fig1.suptitle("Predictions + residuals: pretrained vs. fine-tuned", fontsize=14)

# %%
# Figure 2 — Training dynamics + metric comparison
# ------------------------------------------------
#
# Left: val MSE per epoch; dashed line = pretrained floor, star = best
# epoch kept by the safety net. Right: test-set MSE / MAE / R² bars.
if is_main_process and history["epoch"]:
    fig2, (ax_tr, ax_bar) = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    ep = history["epoch"]
    val_mse = history["val_mse"]
    ax_tr.plot(ep, val_mse, "o-", color="#0f766e", lw=2.0, markersize=5, label="fine-tuning: val MSE")
    ax_tr.axhline(
        base_val_mse,
        ls="--",
        color="#64748b",
        lw=1.5,
        label=f"pretrained baseline ({base_val_mse:.3f})",
    )
    best_idx = int(np.nanargmin(val_mse))
    ax_tr.scatter(
        [ep[best_idx]],
        [val_mse[best_idx]],
        marker="*",
        s=220,
        color="#f59e0b",
        edgecolor="black",
        linewidths=0.8,
        zorder=5,
        label=f"best epoch ({val_mse[best_idx]:.3f} @ epoch {ep[best_idx]})",
    )
    ax_tr.set_xlabel("epoch")
    ax_tr.set_ylabel("validation MSE (lower is better)")
    ax_tr.set_title("Validation metric across fine-tuning epochs")
    ax_tr.grid(alpha=0.3)
    ax_tr.legend(fontsize=9, loc="upper right")

    metric_names = ["MSE ↓", "MAE ↓", "R² ↑"]
    base_vals = [base_mse, base_mae, base_r2]
    ft_vals = [ft_mse, ft_mae, ft_r2]
    x_pos = np.arange(len(metric_names))
    w = 0.38
    bars_b = ax_bar.bar(x_pos - w / 2, base_vals, w, color="#64748b", label="pretrained")
    bars_f = ax_bar.bar(x_pos + w / 2, ft_vals, w, color="#0f766e", label="fine-tuned")
    for bars, vals in [(bars_b, base_vals), (bars_f, ft_vals)]:
        for rect, v in zip(bars, vals):
            y_anchor = v + (0.02 if v >= 0 else -0.04)
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2,
                y_anchor,
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_title("Test-set metrics: pretrained vs. fine-tuned")
    ax_bar.set_ylabel("metric value")
    ax_bar.axhline(0, color="black", lw=0.5)
    ax_bar.grid(alpha=0.25, axis="y")
    ax_bar.legend(fontsize=9, loc="upper right")

    fig2.suptitle("Training dynamics & test-set gains", fontsize=13)
    plt.show()

# %%

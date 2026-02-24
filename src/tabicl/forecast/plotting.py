from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tabicl.forecast.ts_dataframe import TimeSeriesDataFrame


_COLORS = {
    "context": "royalblue",
    "forecast": "tomato",
    "actual": "darkslateblue",
}


def _index_contained_in(tsdf_a: TimeSeriesDataFrame, tsdf_b: TimeSeriesDataFrame) -> bool:
    """Check whether the index of ``tsdf_a`` is contained in ``tsdf_b``.

    Parameters
    ----------
    tsdf_a : TimeSeriesDataFrame
        First time series DataFrame.

    tsdf_b : TimeSeriesDataFrame
        Second time series DataFrame.

    Returns
    -------
    bool
        ``True`` if all index entries in ``tsdf_a`` are also in ``tsdf_b``.
    """
    return tsdf_a.index.isin(tsdf_b.index).all()


def _resolve_item_ids(df: TimeSeriesDataFrame, item_ids: list | None) -> pd.Index:
    """Return validated item IDs or all available ones."""
    available = df.index.get_level_values("item_id").unique()
    if item_ids is None:
        return available
    if not set(item_ids).issubset(available):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")
    return pd.Index(item_ids)


def _create_subplots(n: int):
    """Create vertically stacked subplots, always returning a list of axes."""
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)
    return fig, axes


def plot_forecast(
    context_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    item_ids: list | None = None,
    context_length: int = 100,
    show_quantiles: bool = True,
    show_points: bool = False,
    linewidth: float = 1.8,
) -> None:
    """Plot forecast with historical context and optional ground truth.

    Converts pandas DataFrames from the ``predict_df`` API to
    ``TimeSeriesDataFrame`` and delegates to ``plot_predictions``.

    Parameters
    ----------
    context_df : pd.DataFrame
        Historical data with columns ``timestamp``, ``target``, and
        optionally ``item_id``.

    pred_df : pd.DataFrame
        Predictions from ``predict_df``, with multi-index
        ``(item_id, timestamp)``.

    test_df : pd.DataFrame or None, default=None
        Optional ground truth for the forecast horizon.

    item_ids : list or None, default=None
        Item IDs to plot. If ``None``, plots all unique items.

    context_length : int, default=100
        Number of historical points to show before the forecast.

    show_quantiles : bool, default=True
        Whether to show the quantile prediction range.

    show_points : bool, default=False
        Whether to show individual data points.

    linewidth : float, default=1.8
        Line thickness for all plot lines.
    """
    if "item_id" not in context_df.columns:
        context_df = context_df.copy()
        context_df["item_id"] = 0

    context_df = context_df.groupby("item_id").tail(context_length)

    if test_df is None:
        # Build test_df from prediction timestamps using MultiIndex extraction
        unique_items = pred_df.index.get_level_values(0).unique()
        records = []
        for item_id in unique_items:
            pred_timestamps = pred_df.loc[item_id].index
            item_records = pd.DataFrame({
                "item_id": item_id,
                "timestamp": pred_timestamps,
                "target": np.nan,
            })
            records.append(item_records)
        test_df = pd.concat(records, ignore_index=True)
    else:
        if "item_id" not in test_df.columns:
            test_df = test_df.copy()
            test_df["item_id"] = 0

    train_tsdf = TimeSeriesDataFrame.from_data_frame(context_df)
    test_tsdf = TimeSeriesDataFrame.from_data_frame(test_df)
    pred_tsdf = TimeSeriesDataFrame(pred_df)

    return plot_predictions(
        pred=pred_tsdf,
        train=train_tsdf,
        test=test_tsdf,
        item_ids=item_ids,
        show_quantiles=show_quantiles,
        show_points=show_points,
        linewidth=linewidth,
    )


def plot_series(
    df: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    in_single_plot: bool = False,
    y_limit: tuple[float, float] | None = None,
    show_points: bool = False,
    target_col: str = "target",
):
    """Plot one or more time series.

    Parameters
    ----------
    df : TimeSeriesDataFrame
        Time series data to plot.

    item_ids : list[int] | None, default=None
        Item IDs to plot. If ``None``, plots all items.

    in_single_plot : bool, default=False
        If ``True``, overlay all time series in a single plot.

    y_limit : tuple[float, float] | None, default=None
        Y-axis limits ``(min, max)``.

    show_points : bool, default=False
        Whether to show individual data points.

    target_col : str, default="target"
        Name of the target column to plot.

    Raises
    ------
    ValueError
        If any ``item_ids`` are not found in the DataFrame.
    """
    ids = _resolve_item_ids(df, item_ids)

    if not in_single_plot:
        _, axes = _create_subplots(len(ids))

        for ax, item_id in zip(axes, ids):
            df_item = df.xs(item_id, level="item_id")
            ax.plot(df_item.index, df_item[target_col])
            if show_points:
                ax.scatter(
                    df_item.index,
                    df_item[target_col],
                    color="lightcoral",
                    s=8,
                    alpha=0.8,
                )
            ax.set_title(f"Item ID: {item_id}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Target")
            if y_limit is not None:
                ax.set_ylim(*y_limit)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        for item_id in ids:
            df_item = df.xs(item_id, level="item_id")
            ax.plot(df_item.index, df_item[target_col], label=f"Item ID: {item_id}")
            if show_points:
                ax.scatter(
                    df_item.index,
                    df_item[target_col],
                    color="lightcoral",
                    s=8,
                    alpha=0.8,
                )
        ax.legend()
        if y_limit is not None:
            ax.set_ylim(*y_limit)

    plt.tight_layout()
    plt.show()


def plot_splits(
    train: TimeSeriesDataFrame,
    test: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    show_points: bool = False,
):
    """Plot actual time series with train/test split.

    Parameters
    ----------
    train : TimeSeriesDataFrame
        Training (historical) time series data.

    test : TimeSeriesDataFrame
        Test (future) time series data.

    item_ids : list[int] | None, default=None
        Item IDs to plot. If ``None``, plots all items.

    show_points : bool, default=False
        Whether to show individual data points.

    Raises
    ------
    ValueError
        If any ``item_ids`` are not found in the training DataFrame.
    """
    ids = _resolve_item_ids(train, item_ids)
    _, axes = _create_subplots(len(ids))

    def _plot_single_item(ax, item_id):
        train_item = train.xs(item_id, level="item_id")
        test_item = test.xs(item_id, level="item_id")

        if _index_contained_in(train_item, test_item):
            ground_truth = test_item["target"]
        else:
            ground_truth = pd.concat([train_item[["target"]], test_item[["target"]]])
        ax.plot(ground_truth.index, ground_truth, label="Ground Truth")
        if show_points:
            ax.scatter(ground_truth.index, ground_truth, color="lightblue", s=8, alpha=0.8)

        train_item_length = train.xs(item_id, level="item_id").iloc[-1].name
        ax.axvline(x=train_item_length, color="r", linestyle="--", label="Train/Test Split")

        ax.set_title(f"Item ID: {item_id}")
        ax.legend()

    for i, item_id in enumerate(ids):
        _plot_single_item(axes[i], item_id)

    plt.tight_layout()
    plt.show()


def plot_predictions(
    pred: TimeSeriesDataFrame,
    train: TimeSeriesDataFrame,
    test: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    show_quantiles: bool = True,
    show_points: bool = False,
    linewidth: float = 1.8,
):
    """Plot predictions alongside historical context and actual values.

    Shows three distinct elements:

    - **History**: Historical data used as input.
    - **Future**: Ground truth values for the forecast period.
    - **Forecast**: Model predictions with optional quantile intervals.

    Parameters
    ----------
    pred : TimeSeriesDataFrame
        Model predictions.

    train : TimeSeriesDataFrame
        Historical (training) time series data.

    test : TimeSeriesDataFrame
        Actual (test) time series data.

    item_ids : list[int] | None, default=None
        Item IDs to plot. If ``None``, plots all items.

    show_quantiles : bool, default=True
        Whether to show quantile prediction intervals.

    show_points : bool, default=False
        Whether to show individual data points.

    linewidth : float, default=1.8
        Line thickness for all plot lines.

    Raises
    ------
    ValueError
        If any ``item_ids`` are not found in the training DataFrame, or
        if ``pred`` and ``test`` have mismatched shapes and ``pred`` is
        not a subset of ``test``.
    """
    ids = _resolve_item_ids(train, item_ids)

    if pred.shape[0] != test.shape[0]:
        if not _index_contained_in(pred, test):
            raise ValueError("Pred and Test have different number of items and Pred is not a subset of Test")

        filled_pred = test.copy()
        filled_pred["target"] = np.nan
        for col in pred.columns:
            filled_pred.loc[pred.index, col] = pred[col]
        pred = filled_pred

    if pred.shape[0] != test.shape[0]:
        raise ValueError("Pred and Test shapes still mismatch after alignment")

    fig, axes = _create_subplots(len(ids))

    def _plot_single_item(ax, item_id):
        pred_item = pred.xs(item_id, level="item_id")
        train_item = train.xs(item_id, level="item_id")
        test_item = test.xs(item_id, level="item_id")

        # Plot context (historical data)
        ax.plot(
            train_item.index,
            train_item["target"],
            label="History",
            color=_COLORS["context"],
            linewidth=linewidth,
        )
        if show_points:
            ax.scatter(
                train_item.index,
                train_item["target"],
                color=_COLORS["context"],
                s=8,
                alpha=0.6,
            )

        # Plot actual test values (ground truth) if available
        test_has_values = test_item["target"].notna().any()
        if test_has_values:
            ax.plot(
                test_item.index,
                test_item["target"],
                label="Future",
                color=_COLORS["actual"],
                linewidth=linewidth,
                alpha=0.8,
            )
            if show_points:
                ax.scatter(
                    test_item.index,
                    test_item["target"],
                    color=_COLORS["actual"],
                    s=8,
                    alpha=0.6,
                )

        # Plot prediction
        ax.plot(
            pred_item.index,
            pred_item["target"],
            label="Forecast",
            color=_COLORS["forecast"],
            linewidth=linewidth,
        )

        # Plot quantile range
        if show_quantiles:
            quantile_cols = [c for c in pred_item.columns if c != "target"]
            if len(quantile_cols) >= 2:
                quantile_config = sorted(quantile_cols, key=lambda x: float(x))
                lower_quantile = quantile_config[0]
                upper_quantile = quantile_config[-1]
                ax.fill_between(
                    pred_item.index,
                    pred_item[lower_quantile],
                    pred_item[upper_quantile],
                    color=_COLORS["forecast"],
                    alpha=0.2,
                    label=f"{lower_quantile}-{upper_quantile} Quantile",
                )

        # Add train/test split line
        split_point = train_item.iloc[-1].name
        ax.axvline(x=split_point, color="gray", linestyle="--", alpha=0.7)

        ax.set_title(f"Item ID: {item_id}")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1))

    for i, item_id in enumerate(ids):
        _plot_single_item(axes[i], item_id)

    plt.tight_layout()
    return fig, axes

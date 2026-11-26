# This file is almost entirely AI-generated (DeepSeek V4.1 Flash 1017) from
# the gallery plotting code; it is reviewed and maintained by the TabICL
# maintainers.
"""Plotly figures of the compute-time gallery page.

Renders the gallery snippets in both themes (light/dark) and both layouts
(desktop/mobile), from the results table shipped in
``docs/_data/tabicl_benchmarks_data.csv``.

Each row of that table is one benchmark run: a dataset shape (train/test rows,
features, classes), a model configuration (ensemble size, device, kv cache) and
the measured durations.

The snippets are regenerated at the start of every HTML build by the Sphinx
extension in ``setup()`` below, so the page cannot drift from its data. A file
is only rewritten when its content changes -- Plotly's random div ids are
replaced by stable ones for that -- so incremental and live-reload builds stay
quiet.

The four variants of a figure are swapped by viewport width and page theme,
through the ``plotly-desktop``/``plotly-mobile`` and ``plotly-light``/
``plotly-dark`` rules of ``docs/_static/css/custom.css``.

Adding a figure means writing a builder with the signature of the ones below,
registering it in ``FIGURES``, and including its snippet from the gallery page.

Regenerate by hand with::

    python docs/_ext/compute_tradeoffs_figures.py [--csv CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Where the data and the generated snippets live
# ---------------------------------------------------------------------------
_DOCS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = _DOCS_DIR / "_data" / "tabicl_benchmarks_data.csv"
DEFAULT_OUT = _DOCS_DIR / "_static" / "compute_tradeoffs"

# ---------------------------------------------------------------------------
# Columns of the results table
# ---------------------------------------------------------------------------
RUN_LABEL = "Run label"
NB_TRAIN_SAMPLES = "Nb train samples"
NB_FEATURES = "Nb features"
NB_TEST_SAMPLES = "Nb test samples"
NB_ESTIMATORS = "Nb estimators"
KV_CACHE = "KV cache"
# Canonical values of the KV_CACHE column after ``load_results``.
KV_CACHE_OFF = "off"
KV_CACHE_REPR = "repr"
KV_CACHE_ON = "on"
DEVICE = "Device"
DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
AVAILABLE_MEMORY = "Available Device Memory (MiB)"
FIT_TIME = "Fit time"
PREDICT_TIME = "Predict time"
# Name of the derived fit + predict total, computed by the builders.
TOTAL_TIME = "Total (fit + predict)"
TABICL_VERSION = "TabICL version"

# Ensemble size of every run the figures use. The builders filter on it and the
# subtitles quote it.
BENCHMARK_ESTIMATORS = 8

# ---------------------------------------------------------------------------
# Families of runs
# ---------------------------------------------------------------------------
# Each figure is drawn from one family of runs. The table ships several
# families, and a builder only reads the one it needs; the names are matched as
# substrings of the run labels.
LABEL_SCALING = "rows/cols/test scaling"
LABEL_SMALL = "small workloads"
LABEL_MEMORY = "available memory sweep"

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
#
# Plotly bakes colours into the rendered SVG, so a single figure cannot follow
# the page theme. Each figure is therefore rendered once per theme
# (``theme="light"`` / ``"dark"``) and the documentation page swaps the
# pre-rendered variants with CSS.
#
# Both palettes are Okabe-Ito (colour-blind safe). ``light`` is the canonical
# palette. ``dark`` keeps each light colour's *hue* and only raises its
# lightness so every series stays above 6:1 contrast on the dark documentation
# background (#14181e).
#
# The light palette keeps Okabe-Ito's published lightness: darkening its orange
# until it reaches 3:1 on white collapses it into the vermillion under
# deuteranopia (ΔE 18.3 -> 2.3), and darkening its sky blue collapses it into
# the blue (ΔE 26.4 -> 14.4). Extra separation is encoded with stroke width and
# marker shape instead.
#
# Hover labels use an *opaque* bgcolor: plotly composites a translucent
# hoverlabel.bgcolor against white while the figure paper is transparent, so an
# alpha colour renders as a solid white tooltip under light text in dark mode.
# For the same reason the traces put the series name in the tooltip *body* and
# end their hovertemplate with <extra></extra>: the trace-name tag plotly draws
# next to the body has a hard-coded white 80%-opacity fill that no attribute
# can recolour.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "background": "#ffffff",
        "text": "#1f2937",
        "muted": "#6b7280",
        "tick": "#374151",
        "faint": "#9ca3af",
        "grid": "rgba(0,0,0,0.10)",
        "hover_bg": "#ffffff",  # opaque: see note above
        "hover_border": "#9ca3af",
        "kv_off": "#009E73",  # bluish green
        "kv_repr": "#CC79A7",  # reddish purple
        "kv_on": "#D55E00",  # vermillion
        "cpu": "#0072B2",  # blue
        "gpu": "#D55E00",  # vermillion
    },
    "dark": {
        "background": "#14181e",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "tick": "#9ca3af",
        "faint": "#8b93a1",
        "grid": "rgba(255,255,255,0.14)",
        "hover_bg": "#262b33",  # opaque: see note above
        "hover_border": "#4b5563",
        "kv_off": "#00C48C",  # bluish green, lightened
        "kv_repr": "#E8A0C4",  # reddish purple, lightened
        "kv_on": "#F2703A",  # vermillion, lightened
        "cpu": "#56B4E9",  # blue, lightened
        "gpu": "#F2703A",  # vermillion, lightened
    },
}

# Marker shape is a second channel besides colour.
MARKER_CACHE_OFF = "circle"
MARKER_CACHE_REPR = "diamond"
MARKER_CACHE_ON = "square"
MARKER_CPU = "circle"
MARKER_GPU = "square"

THEME_NAMES = tuple(THEMES)


def _theme(name: str = "light") -> dict[str, str]:
    """Return the colour tokens of ``name`` (``"light"`` or ``"dark"``)."""
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(
            f"Unknown theme {name!r}; expected one of {', '.join(THEME_NAMES)}."
        ) from None


# ---------------------------------------------------------------------------
# Legend and line-style encoding
# ---------------------------------------------------------------------------
# A 40 px swatch shows enough repetitions of the 10-on/6-off dash for the
# "predict" series to read as dashed at a 2 px stroke, in the desktop and the
# mobile variant alike. The dash cycle and the swatch length are independent:
# a short swatch simply shows fewer repetitions of the cycle.
LEGEND_ITEMWIDTH = 40
DASH_PREDICT = "10px,6px"
# Desktop-only: a fixed fraction of the plot width per legend entry, so the
# grid keeps the same number of columns at any window width, with the columns
# aligned.
LEGEND_ENTRYWIDTH = 0.36

# ---------------------------------------------------------------------------
# Margin profiles. Two rendering targets share the same figure builders:
#   * ``default`` — generous margins for direct/standalone HTML viewing.
#   * ``gallery``  — tight margins tuned for the narrow sphinx-gallery column.
# The ``default`` and ``gallery`` profiles share the same vertical layout; they
# differ in horizontal margins, inter-subplot spacing and y-axis title standoff.
# The plot area height is a per-figure choice (``plot_area_px``).
# ---------------------------------------------------------------------------
_DEFAULT_LR = (80, 80)  # generous left/right for standalone viewing
_GALLERY_LR = (20, 5)  # tight for the narrow gallery column
_DEFAULT_YSTANDOFF = None  # plotly default
_GALLERY_YSTANDOFF = 6


# ---------------------------------------------------------------------------
# Reading the results table
# ---------------------------------------------------------------------------
def _parse_mean(value) -> float:
    """Extract the mean from a ``mean±std`` repetition-average string.

    Returns the value as-is if already numeric, and NaN for missing or
    unparsable values.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    if "±" in s:
        s = s.split("±")[0].strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_results(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load the results table with coerced dtypes.

    Numeric metric columns may be stored as ``mean±std`` strings when a value
    was averaged over several repetitions — the mean is extracted so the
    figures receive plain floats.
    """
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])

    if KV_CACHE in df.columns:
        # The column holds the boolean-looking modes plus the ``repr`` caching
        # level; map them onto the three canonical modes so that ``repr`` stays
        # distinct from a missing value.
        df[KV_CACHE] = (
            df[KV_CACHE]
            .astype(str)
            .str.strip()
            .map(
                {
                    "True": KV_CACHE_ON,
                    "true": KV_CACHE_ON,
                    "repr": KV_CACHE_REPR,
                    "False": KV_CACHE_OFF,
                    "false": KV_CACHE_OFF,
                }
            )
        )

    for col in (FIT_TIME, PREDICT_TIME):
        if col in df.columns:
            df[col] = df[col].map(_parse_mean)

    return df


def _family(
    df: pd.DataFrame,
    label: str,
    *,
    device: str | None = None,
    kv_cache: str | None = None,
) -> pd.DataFrame:
    """Rows of one benchmark family, at the ensemble size the figures use.

    ``device`` and ``kv_cache`` restrict the selection further when a figure
    only reads one device or one caching mode.
    """
    sub = df[df[RUN_LABEL].str.contains(label, na=False)]
    sub = sub[sub[NB_ESTIMATORS] == BENCHMARK_ESTIMATORS]
    if device is not None:
        sub = sub[sub[DEVICE] == device]
    if kv_cache is not None:
        sub = sub[sub[KV_CACHE] == kv_cache]
    return sub.copy()


def _tabicl_version_label(df: pd.DataFrame) -> str:
    """Return the ``"TabICL <version>"`` footnote printed under the figures.

    All rows were produced by runs of the same TabICL version, reported in the
    ``TabICL version`` column. The footnote is part of the figure, so a missing
    or empty column is a data error that is reported here, which keeps a broken
    results table visible.
    """
    if TABICL_VERSION not in df.columns:
        raise ValueError(
            f"the results table has no {TABICL_VERSION!r} column, which the "
            f"figure footnote needs; it has: {', '.join(sorted(df.columns))}"
        )
    vals = df[TABICL_VERSION].dropna()
    vals = vals[vals.astype(str).str.strip() != ""]
    if vals.empty:
        raise ValueError(
            f"the {TABICL_VERSION!r} column of the results table is empty, but "
            "the figure footnote needs the TabICL version the runs were made "
            "with"
        )
    return f"TabICL {vals.iloc[0]}"


# ---------------------------------------------------------------------------
# Shared layout of the three-panel scaling figures (figures 1 and 2)
# ---------------------------------------------------------------------------
class _Panel(NamedTuple):
    """One cell of a scaling figure.

    ``index`` is the panel position, left to right on desktop and top to bottom
    on mobile; panels are numbered from 1. ``fixed`` holds the columns whose
    value defines the cell, and the panel plots the ``x_col`` column against
    time.
    """

    index: int
    x_col: str
    fixed: dict[str, int]
    x_title: str
    title: str


class _Series(NamedTuple):
    """One line of a scaling figure.

    ``key`` is the value the figure's split column must have for a row to
    belong to this line, and ``label`` names the line in the legend and the
    hover box.
    """

    metric: str
    metric_name: str
    key: str
    label: str
    color: str
    dash: str
    width: float
    marker: str


# The two phases every scaling figure plots: (metric column, metric name,
# dash, stroke width).
_METRIC_STYLES = (
    (PREDICT_TIME, "predict", DASH_PREDICT, 2.2),
    (TOTAL_TIME, "fit + predict", "solid", 3.0),
)


def _build_series(
    tokens: dict[str, str], groups: tuple[tuple[str, str, str, str], ...]
) -> list[_Series]:
    """Cross-combine the phase styles with the group styles into ``_Series``.

    ``groups`` holds one ``(key, label, colour token, marker)`` entry per cache
    mode or device. Predict lines come first, then fit + predict lines, and
    within a phase the groups keep their order, so the legend is grouped by
    line style.
    """
    return [
        _Series(
            metric=metric,
            metric_name=metric_name,
            key=key,
            label=f"{group_label}, {metric_name}",
            color=tokens[token],
            dash=dash,
            width=width,
            marker=marker,
        )
        for metric, metric_name, dash, width in _METRIC_STYLES
        for key, group_label, token, marker in groups
    ]


def _hspacing(gallery: bool) -> float:
    """Inter-panel horizontal spacing of the desktop layout."""
    return 0.04 if gallery else 0.07


def _panel_grid(panels: tuple[_Panel, ...], gallery: bool, mobile: bool) -> go.Figure:
    """Empty canvas: one row of panels on desktop, one column on mobile."""
    titles = [panel.title for panel in panels]
    if mobile:
        return make_subplots(
            rows=len(panels), cols=1, subplot_titles=titles, vertical_spacing=0.18
        )
    return make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=titles,
        horizontal_spacing=_hspacing(gallery),
    )


def _panel_cell(sub: pd.DataFrame, panel: _Panel) -> pd.DataFrame:
    """Rows of ``sub`` that belong to ``panel``."""
    cell = sub
    for column, value in panel.fixed.items():
        cell = cell[cell[column] == value]
    return cell


def _draw_panels(
    fig: go.Figure,
    sub: pd.DataFrame,
    panels: tuple[_Panel, ...],
    series: list[_Series],
    split_col: str,
    mobile: bool,
) -> None:
    """Draw ``series`` in every panel and style the log axes.

    A row belongs to a series when its ``split_col`` value equals the series
    key. Each panel must hold one run per key, otherwise a line would silently
    disappear from the figure.
    """
    expected = {s.key for s in series}
    for panel in panels:
        cell = _panel_cell(sub, panel)
        present = set(cell[split_col].dropna())
        if not expected <= present:
            raise ValueError(
                f"panel {panel.x_title!r} has no run with {split_col}="
                f"{sorted(expected - present)}; the results table has "
                f"{sorted(present)}"
            )

        row, col = (panel.index, 1) if mobile else (1, panel.index)
        for s in series:
            d = cell[cell[split_col] == s.key].sort_values(panel.x_col)
            fig.add_trace(
                go.Scatter(
                    x=d[panel.x_col],
                    y=d[s.metric],
                    name=s.label,
                    legendgroup=s.label,
                    showlegend=(panel.index == 1),
                    mode="lines+markers",
                    line={"color": s.color, "width": s.width, "dash": s.dash},
                    marker={"symbol": s.marker, "size": 7, "color": s.color},
                    hovertemplate=(
                        f"<b>{s.label}</b>"
                        f"<br>{panel.x_title}=%{{x}}"
                        f"<br>{s.metric_name}=%{{y:.3g}} s<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(type="log", title_text=panel.x_title, row=row, col=col)
        fig.update_yaxes(type="log", row=row, col=col)

    # Stacked panels each need their own y-axis title; on desktop the panels
    # share one y scale, so the leftmost title labels all of them.
    rows = range(1, len(panels) + 1) if mobile else (1,)
    for row in rows:
        fig.update_yaxes(title_text="Time (s)", row=row, col=1)


# ---------------------------------------------------------------------------
# Figure 1 — KV cache: predict and fit+predict time vs dataset size
# ---------------------------------------------------------------------------
# One cell per axis: train rows, features and test rows, with the other two
# dimensions held fixed.
_SCALING_PANELS = (
    _Panel(
        index=1,
        x_col=NB_TRAIN_SAMPLES,
        fixed={NB_FEATURES: 100, NB_TEST_SAMPLES: 2000},
        x_title="Number of train rows",
        title="Time vs number of train rows<br>(n_features=100, n_test=2000)",
    ),
    _Panel(
        index=2,
        x_col=NB_FEATURES,
        fixed={NB_TRAIN_SAMPLES: 1000, NB_TEST_SAMPLES: 1000},
        x_title="Number of features",
        title="Time vs number of features<br>(n_train=1000, n_test=1000)",
    ),
    _Panel(
        index=3,
        x_col=NB_TEST_SAMPLES,
        fixed={NB_TRAIN_SAMPLES: 4000, NB_FEATURES: 100},
        x_title="Test-set size",
        title="Time vs test-set size<br>(n_train=4000, n_features=100)",
    ),
)

# Cache modes from no caching to full caching: (key, label, colour token,
# marker).
_CACHE_MODE_STYLES = (
    (KV_CACHE_OFF, "no cache", "kv_off", MARKER_CACHE_OFF),
    (KV_CACHE_REPR, "repr cache", "kv_repr", MARKER_CACHE_REPR),
    (KV_CACHE_ON, "full cache", "kv_on", MARKER_CACHE_ON),
)


def build_fig1_kv_cache(
    df: pd.DataFrame,
    gallery: bool = False,
    mobile: bool = False,
    theme: str = "light",
) -> go.Figure:
    """Predict and fit+predict time vs dataset size, per KV cache mode.

    Three panels (train rows, features, test rows), each with one dashed
    predict line and one solid fit+predict line per cache mode, in the mode
    order ``no cache`` (green), ``repr cache`` (purple), ``full cache``
    (vermillion). Prediction is the outer legend group and the cache mode the
    inner one.

    The fixed dimensions of each panel are in ``_SCALING_PANELS``. Runs: GPU,
    n_estimators=8, 10 classes, ``offload_mode=auto``. Log axes; ``theme``
    selects the colour token set (``"light"``/``"dark"``).
    """
    tokens = _theme(theme)
    sub = _family(df, LABEL_SCALING, device=DEVICE_GPU)
    sub[TOTAL_TIME] = sub[FIT_TIME] + sub[PREDICT_TIME]

    fig = _panel_grid(_SCALING_PANELS, gallery, mobile)
    _draw_panels(
        fig,
        sub,
        _SCALING_PANELS,
        _build_series(tokens, _CACHE_MODE_STYLES),
        split_col=KV_CACHE,
        mobile=mobile,
    )

    _finalize(
        fig,
        title="Figure 1 — KV cache: predict and fit+predict time vs dataset size",
        subtitle="GPU (NVIDIA L4), 10 classes, n_estimators=8.",
        legend_title=None,
        gallery=gallery,
        plot_area_px=600 if mobile else 240,
        mobile=mobile,
        footnote=_tabicl_version_label(sub),
        theme=theme,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — CPU vs GPU with the kv cache on: predict and fit+predict time
# ---------------------------------------------------------------------------
# The small-workload sweep, one cell per axis, with the other two dimensions
# held at the smallest shape of the sweep.
_SMALL_WORKLOAD_PANELS = (
    _Panel(
        index=1,
        x_col=NB_TRAIN_SAMPLES,
        fixed={NB_FEATURES: 10, NB_TEST_SAMPLES: 100},
        x_title="Number of train rows",
        title="Time vs number of train rows<br>(n_features=10, n_test=100)",
    ),
    _Panel(
        index=2,
        x_col=NB_FEATURES,
        fixed={NB_TRAIN_SAMPLES: 100, NB_TEST_SAMPLES: 100},
        x_title="Number of features",
        title="Time vs number of features<br>(n_train=100, n_test=100)",
    ),
    _Panel(
        index=3,
        x_col=NB_TEST_SAMPLES,
        fixed={NB_TRAIN_SAMPLES: 100, NB_FEATURES: 10},
        x_title="Test-set size",
        title="Time vs test-set size<br>(n_train=100, n_features=10)",
    ),
)

# Devices, in legend and bar order: (key, label, colour token, marker).
_DEVICE_STYLES = (
    (DEVICE_CPU, "CPU", "cpu", MARKER_CPU),
    (DEVICE_GPU, "GPU", "gpu", MARKER_GPU),
)


def build_fig2_cpu_vs_gpu(
    df: pd.DataFrame,
    gallery: bool = False,
    mobile: bool = False,
    theme: str = "light",
) -> go.Figure:
    """Predict and fit+predict time on CPU vs GPU, kv cache on (3 axes).

    Three panels (train rows, features, test rows), each with one dashed
    predict line and one solid fit+predict line per device, in the order CPU
    (blue), GPU (vermillion). The fixed dimensions of each panel are in
    ``_SMALL_WORKLOAD_PANELS``.

    n_estimators=8, classification, 10 classes. Log axes; ``theme`` selects
    the colour token set (``"light"``/``"dark"``).
    """
    tokens = _theme(theme)
    sub = _family(df, LABEL_SMALL, kv_cache=KV_CACHE_ON)
    sub[TOTAL_TIME] = sub[FIT_TIME] + sub[PREDICT_TIME]

    fig = _panel_grid(_SMALL_WORKLOAD_PANELS, gallery, mobile)
    _draw_panels(
        fig,
        sub,
        _SMALL_WORKLOAD_PANELS,
        _build_series(tokens, _DEVICE_STYLES),
        split_col=DEVICE,
        mobile=mobile,
    )

    _finalize(
        fig,
        title="Figure 2 — CPU vs GPU with the kv cache on: predict and fit+predict time",
        subtitle="CPU (AMD EPYC 7R13) and GPU (NVIDIA L4), 10 classes, n_estimators=8.",
        legend_title=None,
        gallery=gallery,
        plot_area_px=600 if mobile else 240,
        mobile=mobile,
        footnote=_tabicl_version_label(sub),
        theme=theme,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — CPU vs GPU on large matched shapes (bars)
# ---------------------------------------------------------------------------
def build_fig3_large_shapes(
    df: pd.DataFrame,
    gallery: bool = False,
    mobile: bool = False,
    theme: str = "light",
) -> go.Figure:
    """Total fit + predict time, CPU vs GPU, on the large matched shapes.

    One group of bars per dataset shape measured on both devices with the kv
    cache on, and one bar per device inside a group. The bars show the total
    fit + predict time; the hover box keeps the fit/predict split.

    n_estimators=8, classification, 10 classes. ``theme`` selects the colour
    token set (``"light"``/``"dark"``).
    """
    tokens = _theme(theme)
    sub = _family(df, LABEL_SCALING, kv_cache=KV_CACHE_ON)
    sub[TOTAL_TIME] = sub[FIT_TIME] + sub[PREDICT_TIME]

    shape_cols = [NB_TRAIN_SAMPLES, NB_FEATURES, NB_TEST_SAMPLES]
    per_device = {
        device: sub[sub[DEVICE] == device].set_index(shape_cols)
        for device, _, _, _ in _DEVICE_STYLES
    }
    shapes = sorted(
        set(per_device[DEVICE_CPU].index) & set(per_device[DEVICE_GPU].index)
    )
    if not shapes:
        raise ValueError(
            "figure 3 needs at least one dataset shape measured on both the "
            "CPU and the GPU with the kv cache on; the results table has none"
        )
    labels = [
        f"{int(rows)} × {int(features)} × {int(test)}"
        for rows, features, test in shapes
    ]

    fig = go.Figure()
    totals: dict[str, list[float]] = {}
    for device, device_name, token, _ in _DEVICE_STYLES:
        d = per_device[device]
        totals[device] = [float(d.loc[shape, TOTAL_TIME]) for shape in shapes]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=totals[device],
                name=device_name,
                marker={"color": tokens[token]},
                customdata=[
                    [float(d.loc[shape, FIT_TIME]), float(d.loc[shape, PREDICT_TIME])]
                    for shape in shapes
                ],
                text=[f"{v:.3g} s" for v in totals[device]],
                # The CPU bar of every pair is the tallest, so its label fits
                # inside; the GPU label sits outside to stay readable.
                textposition="inside" if device == DEVICE_CPU else "outside",
                insidetextfont={"color": tokens["background"]},
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{device_name}</b><br>shape=%{{x}}"
                    "<br>fit=%{customdata[0]:.3g} s"
                    "<br>predict=%{customdata[1]:.3g} s"
                    "<br>total=%{y:.3g} s<extra></extra>"
                ),
            )
        )

    for i, label in enumerate(labels):
        ratio = totals[DEVICE_CPU][i] / totals[DEVICE_GPU][i]
        fig.add_annotation(
            x=label,
            y=totals[DEVICE_CPU][i] * 1.06,
            xref="x",
            yref="y",
            text=f"GPU {ratio:.0f}× faster",
            showarrow=False,
            font={"size": 11, "color": tokens["muted"]},
        )

    fig.update_layout(barmode="group", bargap=0.45, bargroupgap=0.1)
    fig.update_xaxes(title_text="Dataset shape (rows × features × test)")
    fig.update_yaxes(
        title_text="Time (s)", range=[0, max(map(max, totals.values())) * 1.2]
    )

    _finalize(
        fig,
        title="Figure 3 — CPU vs GPU, kv cache on: large matched shapes",
        subtitle="Total fit + predict time, 10 classes, n_estimators=8.",
        legend_title=None,
        gallery=gallery,
        plot_area_px=300 if mobile else 240,
        mobile=mobile,
        subplot_titles=False,
        footnote=_tabicl_version_label(sub),
        theme=theme,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Predict time vs available GPU memory
# ---------------------------------------------------------------------------
def build_fig4_available_memory(
    df: pd.DataFrame,
    gallery: bool = False,
    mobile: bool = False,
    theme: str = "light",
) -> go.Figure:
    """Predict time vs the memory budget the solver may use.

    Single panel: the available-memory sweep, one capped budget per tick plus
    the uncapped run (``available_memory=None``, shown as ``"no cap"``). The
    x-axis is categorical because the ticks are the budgets that were
    measured, and the uncapped run has no budget to place on a numeric axis.
    The axis runs from the uncapped device down to the tightest budget, so the
    page reads as memory *shrinking*.

    Only the predict time is plotted: with the kv cache off, ``fit`` is weight
    loading plus light preprocessing and stays flat across the sweep, so it
    would only add a constant offset.

    Fixed workload: n_train=8000, n_features=200, n_test=1000, 10 classes,
    n_estimators=8, kv cache off, GPU. ``theme`` selects the colour token set
    (``"light"``/``"dark"``).
    """
    tokens = _theme(theme)
    sub = _family(df, LABEL_MEMORY, device=DEVICE_GPU, kv_cache=KV_CACHE_OFF)
    if sub.empty:
        raise ValueError(
            "figure 4 needs the available-memory sweep (GPU, kv cache off); "
            "the results table has none"
        )

    # Read left to right as a *decreasing* budget: the sweep starts from the
    # unconstrained device and tightens towards the 450 MiB cliff, the
    # direction the reader cares about.
    sub = sub.sort_values(AVAILABLE_MEMORY, na_position="last")
    sub = sub.iloc[::-1].reset_index(drop=True)
    labels = ["no cap" if pd.isna(v) else f"{int(v)}" for v in sub[AVAILABLE_MEMORY]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=sub[PREDICT_TIME],
            name="predict",
            mode="lines+markers",
            line={"color": tokens["gpu"], "width": 3.0},
            marker={"symbol": "circle", "size": 8, "color": tokens["gpu"]},
            hovertemplate=(
                "<b>predict</b><br>available memory=%{x} MiB"
                "<br>predict=%{y:.3g} s<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(type="category", title_text="Available device memory (MiB)")
    # Frame the y-axis on the data: the sweep spans roughly 8 s to 16 s, so a
    # zero baseline would squeeze the 450 MiB cliff into a few pixels. A line
    # chart may crop its baseline; the range keeps 10% of the span as headroom
    # below the lowest point.
    lo = float(sub[PREDICT_TIME].min())
    hi = float(sub[PREDICT_TIME].max())
    pad = 0.1 * (hi - lo)
    fig.update_yaxes(title_text="Predict time (s)", range=[lo - pad, hi + pad])

    _finalize(
        fig,
        title="Figure 4 — Predict time vs available GPU memory",
        subtitle=(
            "GPU (NVIDIA L4), 10 classes, n_estimators=8, kv cache off."
            "<br>n_train=8000, n_features=200, n_test=1000."
        ),
        legend_title=None,
        gallery=gallery,
        plot_area_px=300 if mobile else 240,
        mobile=mobile,
        subplot_titles=False,
        legend=False,
        footnote=_tabicl_version_label(sub),
        theme=theme,
    )
    return fig


# ---------------------------------------------------------------------------
# Shared layout of every figure
# ---------------------------------------------------------------------------
def _legend_entries(fig: go.Figure) -> int:
    """Number of traces the legend shows."""
    return sum(
        1 for trace in fig.data if getattr(trace, "showlegend", None) is not False
    )


def _legend_kwargs(
    legend_y: float,
    legend_yanchor: str,
    legend_title: str | None,
    gallery: bool,
    mobile: bool,
    n_entries: int,
) -> dict:
    """Legend layout, shared by every figure.

    The legend is horizontal, centred, and anchored at the top of the bottom
    band so that it grows downward into the margin. ``itemwidth`` fixes the
    swatch length for the desktop and mobile variants alike.

    In the gallery, legends of three entries or more get a fixed entry width so
    that their columns line up: two columns up to four entries, three columns
    above that. The builders list the predict series first, so the wider grid
    puts one line style per row. One- and two-entry legends keep their natural
    width, and the mobile variants always wrap naturally, because a fixed
    fraction of a narrow figure would leave less room than swatch plus label
    needs.
    """
    kwargs = {
        "title": (
            {"text": f"<b>{legend_title}</b>", "font_size": 13}
            if legend_title
            else None
        ),
        "orientation": "h",
        "y": legend_y,
        "yanchor": legend_yanchor,
        "x": 0.5,
        "xanchor": "center",
        "font_size": 12,
        "itemwidth": LEGEND_ITEMWIDTH,
    }
    if gallery and not mobile and n_entries > 2:
        columns = 2 if n_entries <= 4 else 3
        kwargs["entrywidthmode"] = "fraction"
        kwargs["entrywidth"] = min(LEGEND_ENTRYWIDTH, 1.0 / columns - 0.02)
    return kwargs


def _finalize(
    fig: go.Figure,
    *,
    title: str,
    subtitle: str,
    legend_title: str | None = None,
    gallery: bool = False,
    plot_area_px: int = 240,
    mobile: bool = False,
    subplot_titles: bool = True,
    legend: bool = True,
    footnote: str | None = None,
    theme: str = "light",
) -> None:
    """Apply the shared layout, title block and footnote of a figure.

    The plot area stays ``plot_area_px`` tall in every variant, so the figures
    keep their proportions across layouts; the total height is that plot area
    plus a top band (title, subtitle and subplot titles) and a bottom band
    (x-axis titles, legend and footnote).

    ``title`` and ``subtitle`` are drawn as one annotation pinned to the top
    edge of the top band, the subtitle in a smaller muted font; ``subtitle``
    may contain ``<br>`` line breaks. ``legend`` reserves the legend band and
    ``subplot_titles`` tells the top band whether the figure carries subplot
    titles. ``footnote`` is pinned to the bottom-right corner. ``theme``
    selects the colour tokens of ``THEMES``.
    """
    tokens = _theme(theme)

    # Top band. The title block is an annotation pinned to the top edge of the
    # band with ``yanchor="top"``, so it grows downward from there and stays
    # clear of the subplot titles at the bottom of the band.
    #
    # The pixel budgets follow the rendered text metrics: a bold 18 px title
    # line takes ~30 px, a 13 px subtitle ~22 px per line, and a two-line 14 px
    # subplot title ~48 px. The gap before the subplot titles grows a little
    # with the number of subtitle lines, so taller subtitle blocks keep their
    # distance.
    n_subtitle_lines = subtitle.count("<br>") + 1
    title_block_px = 30 + n_subtitle_lines * 22
    subplot_title_px = 48 if subplot_titles else 8
    gap_px = 16 + n_subtitle_lines * 2
    top_margin = title_block_px + gap_px + subplot_title_px

    # Bottom band. In the gallery the legend hangs from the top of the band and
    # grows downward, while the footnote sits on the bottom edge and grows
    # upward, so a wrapped legend cannot run into the note. The band holds two
    # legend rows plus the footnote; a figure without a legend only needs the
    # x-axis title and the footnote.
    if gallery:
        bottom_margin = 148 if legend else 76
        # The legend starts a fixed pixel distance below the plot, just past
        # the tick labels and the x-axis title; the plot area height converts
        # that distance to paper coordinates.
        legend_y = -58 / plot_area_px
    else:
        bottom_margin = 160 if legend else 96
        legend_y = -0.22

    height = plot_area_px + top_margin + bottom_margin
    # Paper y of the figure's top edge: paper coordinates cover the plot area,
    # so the band above it is measured in plot-area heights.
    band_top = 1 + top_margin / plot_area_px
    left_margin, right_margin = _GALLERY_LR if gallery else _DEFAULT_LR
    y_standoff = _GALLERY_YSTANDOFF if gallery else _DEFAULT_YSTANDOFF
    # Gallery snippets sit on the documentation page, whose background the CSS
    # swaps with the theme, so the figure paper stays transparent. Standalone
    # HTML opens on a blank browser page and needs an opaque background.
    paper_bgcolor = "rgba(0,0,0,0)" if gallery else tokens["background"]

    # The subplot titles are the annotations already on the figure; remember
    # how many there are so that the restyling loop below leaves the title
    # annotation, appended after them, untouched.
    n_subplot_anns = len(fig.layout.annotations)
    fig.add_annotation(
        text=(
            f"<b>{title}</b><br>"
            f"<span style='font-size:13px;color:{tokens['muted']}'>{subtitle}</span>"
        ),
        x=0.5,
        xref="paper",
        y=band_top,
        yref="paper",
        yanchor="top",
        xanchor="center",
        showarrow=False,
        font={"size": 18, "color": tokens["text"]},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=paper_bgcolor,
        # Plotly's default text colour is a dark navy, unreadable on a dark
        # background.
        font={"color": tokens["text"]},
        hoverlabel={
            "bgcolor": tokens["hover_bg"],
            "bordercolor": tokens["hover_border"],
            "font": {"color": tokens["text"]},
        },
        hovermode="closest",
        showlegend=legend,
        legend=_legend_kwargs(
            legend_y=legend_y,
            legend_yanchor="top",
            legend_title=legend_title,
            gallery=gallery,
            mobile=mobile,
            n_entries=_legend_entries(fig),
        ),
        margin={
            "l": left_margin,
            "r": right_margin,
            "t": top_margin,
            "b": bottom_margin,
        },
        height=height,
    )
    # Subplot titles sit just above the plot area: they are anchored at their
    # baseline and lifted a little into the top band. Their position and font
    # are the only things changed here, so the plot area keeps its size.
    for i in range(n_subplot_anns):
        fig.layout.annotations[i].update(
            y=fig.layout.annotations[i].y + 0.03,
            yanchor="bottom",
            font={"size": 12, "color": tokens["text"]},
        )
    yaxis_kwargs = {
        "zeroline": False,
        "gridcolor": tokens["grid"],
        "title_font": {"size": 13, "color": tokens["text"]},
        "tickfont": {"size": 11, "color": tokens["tick"]},
    }
    if y_standoff is not None:
        yaxis_kwargs["title_standoff"] = y_standoff
    fig.update_yaxes(**yaxis_kwargs)
    # Share one vertical scale across the subplots: matched y axes autorange
    # to the union of their data and zoom together. The first subplot's axis is
    # the reference the others match.
    y_axes = list(fig.select_yaxes())
    for axis in y_axes[1:]:
        axis.matches = "y"
    fig.update_xaxes(
        zeroline=False,
        gridcolor=tokens["grid"],
        title_font={"size": 13, "color": tokens["text"]},
        tickfont={"size": 11, "color": tokens["tick"]},
    )
    # Footnote pinned at the bottom-right corner of the figure, below the
    # legend. The bottom margin is in pixels; convert it to paper coordinates
    # through the fixed plot area height.
    if footnote:
        fig.add_annotation(
            text=(
                f"<span style='font-size:10px;color:{tokens['faint']}'>"
                f"{footnote}</span>"
            ),
            x=1,
            xref="paper",
            xanchor="right",
            y=-(bottom_margin - 8) / plot_area_px,
            yref="paper",
            yanchor="bottom",
            showarrow=False,
        )


# ---------------------------------------------------------------------------
# Gallery embed snippets
# ---------------------------------------------------------------------------
# Builders by snippet name: the key is the snippet filename (``fig1.html``),
# and the order here is the order they are written in. Every builder is called
# as ``build(df, gallery=..., mobile=..., theme=...)``.
_Builder = Callable[..., go.Figure]
FIGURES: dict[str, _Builder] = {
    "fig1": build_fig1_kv_cache,
    "fig2": build_fig2_cpu_vs_gpu,
    "fig3": build_fig3_large_shapes,
    "fig4": build_fig4_available_memory,
}

# Output order of the variants: plotly is loaded from the CDN once, by the
# first snippet; the others reuse it.
VARIANTS = ((False, "light"), (False, "dark"), (True, "light"), (True, "dark"))

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _snippet(fig: go.Figure, *, slug: str, include_plotlyjs: bool) -> str:
    """Render ``fig`` as an embeddable <div>+<script> block.

    ``include_plotlyjs`` embeds the plotly library in this block; the other
    blocks of the page reuse the copy loaded by the first one. ``slug``
    replaces the random element id plotly generates, so that an unchanged
    figure renders byte-identically and the file is left untouched.
    """
    html = fig.to_html(
        full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False
    )
    return _UUID.sub(slug, html)


def build_gallery_html(df: pd.DataFrame, name: str, build: _Builder) -> str:
    """Return the four-variant gallery snippet of one figure."""
    parts = []
    for mobile, theme in VARIANTS:
        layout = "plotly-mobile" if mobile else "plotly-desktop"
        parts.append(
            f'<div class="{layout} plotly-{theme}">\n'
            + _snippet(
                build(df, gallery=True, mobile=mobile, theme=theme),
                slug=f"{name}-{layout}-{theme}",
                include_plotlyjs=not parts,
            )
            + "\n</div>\n"
        )
    return "".join(parts)


def generate(
    csv_path: str | Path = DEFAULT_CSV,
    out_dir: str | Path = DEFAULT_OUT,
    verbose: bool = True,
) -> list[Path]:
    """(Re)write every gallery snippet; return the paths that changed."""
    df = load_results(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, build in FIGURES.items():
        path = out_dir / f"{name}.html"
        html = build_gallery_html(df, name, build)
        if path.exists() and path.read_text() == html:
            if verbose:
                print(f"{path}: up to date")
            continue
        path.write_text(html)
        written.append(path)
        if verbose:
            print(f"{path}: written")
    return written


# ---------------------------------------------------------------------------
# Sphinx integration
# ---------------------------------------------------------------------------
def _generate_at_build(app) -> None:
    """Regenerate the snippets before any source is read (HTML builds only)."""
    if app.builder.format != "html":
        return
    srcdir = Path(app.srcdir)
    generate(
        srcdir / DEFAULT_CSV.relative_to(_DOCS_DIR),
        srcdir / DEFAULT_OUT.relative_to(_DOCS_DIR),
        verbose=False,
    )


def setup(app):
    """Sphinx entry point: regenerate the snippets at the start of each build."""
    app.connect("builder-inited", _generate_at_build)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}


def main(argv: list[str] | None = None) -> None:
    """Regenerate the snippets from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, help="results table to read"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory the snippets are written to",
    )
    args = parser.parse_args(argv)
    generate(args.csv, args.out)


if __name__ == "__main__":
    main()

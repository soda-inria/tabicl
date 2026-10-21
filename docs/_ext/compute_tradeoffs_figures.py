# This file is almost entirely AI-generated (DeepSeek V4.1 Flash 1017) from
# the gallery plotting code; it is reviewed and maintained by the TabICL
# maintainers.
"""Figures of the "Compute time and memory footprint" gallery page.

Renders the gallery snippets of figures 1 and 2, each in both themes
(light/dark) and both layouts (desktop/mobile), from the results table shipped
in ``docs/_data/tabicl_benchmarks_data.csv``.

Each row of that table is one benchmark run: a dataset shape (train/test rows,
features, classes), a model configuration (ensemble size, device, kv cache) and
the measured durations and peak memory.

The snippets are regenerated at the start of every HTML build by the Sphinx
extension in ``setup()`` below, so the page can never drift from its data. A
file is only rewritten when its content changes -- Plotly's random div ids are
replaced by stable ones for that -- so incremental and live-reload builds stay
quiet.

Regenerate by hand with::

    python docs/_ext/compute_tradeoffs_figures.py [--csv CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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
NB_TRAIN_SAMPLES = "Nb train samples"
NB_FEATURES = "Nb features"
NB_TEST_SAMPLES = "Nb test samples"
NB_ESTIMATORS = "Nb estimators"
KV_CACHE = "KV cache"
FIT_TIME = "Fit time"
PREDICT_TIME = "Predict time"
RAM_PEAK = "RAM peak (MB)"
VRAM_PEAK = "VRAM peak (MB)"
VRAM_PEAK_FIT = "VRAM peak fit (MB)"
VRAM_PEAK_PREDICT = "VRAM peak predict (MB)"
TABICL_VERSION = "TabICL version"

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
#
# Plotly bakes colours into the rendered SVG, so a single figure cannot follow
# the page theme. Each figure is therefore rendered once per theme
# (``theme="light"`` / ``"dark"``) and the documentation page swaps the
# pre-rendered variants with CSS (see ``--gallery-html``).
#
# Both palettes are Okabe-Ito (colour-blind safe). ``light`` is the canonical
# palette. ``dark`` keeps each light colour's *hue* and only raises its
# lightness so every series stays above 6:1 contrast on the dark documentation
# background (#14181e).
#
# Hover labels must use an *opaque* bgcolor: plotly composites a translucent
# hoverlabel.bgcolor against white (not against the figure's paper, and our
# paper is transparent), so an alpha colour renders as a solid white tooltip
# under light text in dark mode. For the same reason the traces put the series
# name in the tooltip *body* and end their hovertemplate with <extra></extra>:
# the trace-name tag plotly draws next to the body has a hard-coded white
# 80%-opacity fill that no attribute can recolour.
#
# Do NOT "fix" the light palette by darkening its orange or sky blue to reach
# 3:1 on white: measured under simulated colour-blindness (Machado matrices),
# darkening Okabe-Ito orange until it reaches 3:1 collapses it into the
# vermillion (ΔE 18.3 -> 2.3 under deuteranopia), and darkening the sky blue
# collapses it into the blue (ΔE 26.4 -> 14.4). The lightness *relations*
# between those hue families are what keeps them separable; encode extra
# distinction with stroke width or marker shape instead.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "background": "#ffffff",
        "text": "#1f2937",
        "muted": "#6b7280",
        "tick": "#374151",
        "faint": "#9ca3af",
        "grid": "rgba(0,0,0,0.10)",
        "marker_ring": "#1f2937",
        "hover_bg": "#ffffff",  # opaque: see note above
        "hover_border": "#9ca3af",
        "kv_off": "#009E73",  # bluish green
        "kv_on": "#D55E00",  # vermillion
        "gpu_gpu": "#0072B2",  # blue (GPU, offload=gpu)
        "gpu_cpu": "#56B4E9",  # sky blue (GPU, offload=cpu)
        "gpu_disk": "#CC79A7",  # reddish purple (GPU, offload=disk)
        "cpu_cpu": "#E69F00",  # orange (CPU, offload=cpu)
        "cpu_disk": "#D55E00",  # vermillion (CPU, offload=disk)
        "bar_time": "#000000",  # black (predict time)
        "bar_ram": "#0072B2",  # blue (RAM peak)
        "bar_vram": "#D55E00",  # vermillion (VRAM peak)
    },
    "dark": {
        "background": "#14181e",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "tick": "#9ca3af",
        "faint": "#8b93a1",
        "grid": "rgba(255,255,255,0.14)",
        "marker_ring": "#14181e",
        "hover_bg": "#262b33",  # opaque: see note above
        "hover_border": "#4b5563",
        "kv_off": "#00C48C",  # bluish green, lightened
        "kv_on": "#F2703A",  # vermillion, lightened
        "gpu_gpu": "#56B4E9",  # sky blue (was blue), lightened
        "gpu_cpu": "#8FD8F7",  # sky blue, lightened further
        "gpu_disk": "#E29BC9",  # reddish purple, lightened
        "cpu_cpu": "#E69F00",  # orange, already 7.9:1 on dark
        "cpu_disk": "#F2703A",  # vermillion, lightened
        "bar_time": "#e5e7eb",  # light grey: black vanishes on dark
        "bar_ram": "#56B4E9",  # sky blue, lightened
        "bar_vram": "#F2703A",  # vermillion, lightened
    },
}

MARKER_CACHE_OFF = "circle"
MARKER_CACHE_ON = "square"
MARKER_GPU_GPU = "circle"
MARKER_GPU_CPU = "diamond"
MARKER_GPU_DISK = "square"
MARKER_CPU_CPU = "triangle-up"
MARKER_CPU_DISK = "triangle-down"

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
# Legend / line-style encoding
# ---------------------------------------------------------------------------
# The dashed "predict" series is the one readers confuse with the solid
# "fit + predict" one, because plotly's default swatch is only 30 px wide and
# its default 6-on/6-off "dash" cycle then reads as a smudge at a 2 px stroke.
# Give the sample room (one value for both the desktop and the mobile variant)
# and use a coarser custom dash so the gap survives the small scale. The
# swatch is only a *sample* of the line: shortening it does not require a
# shorter dash cycle, it just shows fewer repetitions of it.
LEGEND_ITEMWIDTH = 40
DASH_PREDICT = "10px,6px"
# Desktop-only: give each legend entry a fixed fraction of the plot width so
# exactly two fit per row at any width. Without it the four-entry legend
# collapses into a single long row on wide screens (and wraps 3 + 1 in
# between), whereas the fixed grid keeps it at 2 + 2 with aligned columns.
LEGEND_ENTRYWIDTH = 0.36

# ---------------------------------------------------------------------------
# Margin profiles. Two rendering targets share the same figure builders:
#   * ``default`` — generous margins for direct/standalone HTML viewing.
#   * ``gallery``  — tight margins tuned for the narrow sphinx-gallery column.
# The ``<br>`` title/subtitle wrapping and the annotation-based title
# positioning are correctness (identical in both); only the horizontal margins,
# inter-subplot spacing, and y-axis title standoff differ. The plot area is
# fixed at 240 px tall in both, so vertical proportions never change.
# ---------------------------------------------------------------------------
_DEFAULT_LR = (80, 80)  # generous left/right for standalone viewing
_GALLERY_LR = (20, 5)  # tight for the narrow gallery column
_DEFAULT_YSTANDOFF = None  # plotly default
_GALLERY_YSTANDOFF = 6


def _hspacing(gallery: bool) -> float:
    """Inter-subplot horizontal spacing of the three-panel layout."""
    return 0.04 if gallery else 0.07


# Figures 1 and 2 are drawn from the runs that sweep the dataset shape; the
# table may also hold runs they do not use.
LABEL_SCALING = "rows & cols scaling"


def _parse_mean(value):
    """Extract the mean from a ``mean±std`` repetition-average string.

    Returns the value as-is if already numeric/empty.
    """
    import math

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

    for col in (KV_CACHE,):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map({"True": True, "False": False, "true": True, "false": False})
            )

    for col in (
        FIT_TIME,
        PREDICT_TIME,
        RAM_PEAK,
        VRAM_PEAK,
        VRAM_PEAK_FIT,
        VRAM_PEAK_PREDICT,
    ):
        if col in df.columns:
            df[col] = df[col].map(_parse_mean)

    return df


def _tabicl_version_label(df: pd.DataFrame) -> str:
    """Return the ``"TabICL <version>"`` footnote printed under the figures.

    All rows were produced by runs of the same TabICL version, reported in the
    ``TabICL version`` column. The footnote is part of the figure, so a missing
    or empty column is a data error: report what is wrong instead of silently
    dropping the note.
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
# Figure 1 — KV cache: predict & fit+predict vs rows, cols, test-set size
# ---------------------------------------------------------------------------


def build_fig1_kv_cache(
    df: pd.DataFrame,
    gallery: bool = False,
    mobile: bool = False,
    theme: str = "light",
) -> go.Figure:
    """Predict and fit+predict time vs dataset size (3 axes), KV cache OFF vs ON.

    Layout: 1 row × 3 cols. Each cell plots four lines:

        * predict       — kv OFF (green, dashed)
        * predict       — kv ON  (vermillion, dashed)
        * fit+predict   — kv OFF (green, solid)
        * fit+predict   — kv ON  (vermillion, solid)

      * row 1: vs n_train     (rows sweep; n_features=100, n_test=2000)
      * row 2: vs n_features  (cols sweep; n_train=1000,  n_test=2000)
      * row 3: vs n_test      (rows sweep; n_train=4000,  n_features=100)

    n_estimators=4, classification, 10 classes, cuda, offload=gpu.
    Log axes. ``theme`` selects the colour token set (``"light"``/``"dark"``).
    """
    tokens = _theme(theme)
    sub = df[df["Run label"].str.contains(LABEL_SCALING, na=False)].copy()
    sub = sub[sub[NB_ESTIMATORS] == 4]
    sub["Total (fit + predict)"] = sub[FIT_TIME] + sub[PREDICT_TIME]

    # (row, x-axis column, filter dict, x-axis title, subplot title)
    rows_spec = [
        (
            1,
            NB_TRAIN_SAMPLES,
            {NB_FEATURES: 100, NB_TEST_SAMPLES: 2000},
            "Number of train rows",
            "Time vs number of train rows<br>(n_features=100, n_test=2000)",
        ),
        (
            2,
            NB_FEATURES,
            {NB_TRAIN_SAMPLES: 1000, NB_TEST_SAMPLES: 2000},
            "Number of features",
            "Time vs number of features<br>(n_train=1000, n_test=2000)",
        ),
        (
            3,
            NB_TEST_SAMPLES,
            {NB_TRAIN_SAMPLES: 4000, NB_FEATURES: 100},
            "Test-set size",
            "Time vs test-set size<br>(n_train=4000, n_features=100)",
        ),
    ]

    if mobile:
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            vertical_spacing=0.18,
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            horizontal_spacing=_hspacing(gallery),
        )

    # (kv value, color, marker, kv_name)
    kv_settings = [
        (False, tokens["kv_off"], MARKER_CACHE_OFF, "no kv cache"),
        (True, tokens["kv_on"], MARKER_CACHE_ON, "kv cache"),
    ]

    # (metric column, dash, width, metric_name). Metrics are the outer loop so
    # the legend groups identical line styles together: the two "predict"
    # entries (dashed) first, then the two "fit + predict" entries (solid).
    # Readers then compare colours within a group instead of having to tell a
    # dash from a solid between neighbours.
    metrics = [
        (PREDICT_TIME, DASH_PREDICT, 2.2, "predict"),
        ("Total (fit + predict)", "solid", 3.0, "fit + predict"),
    ]

    for col_idx, x_col, filt, x_title, _ in rows_spec:
        cell = sub
        for k, v in filt.items():
            cell = cell[cell[k] == v]

        r, c = (col_idx, 1) if mobile else (1, col_idx)
        for metric, dash, lw, metric_name in metrics:
            for kv, color, marker, kv_name in kv_settings:
                d = cell[cell[KV_CACHE] == kv].sort_values(x_col)
                fig.add_trace(
                    go.Scatter(
                        x=d[x_col],
                        y=d[metric],
                        name=f"{kv_name}, {metric_name}",
                        legendgroup=f"{kv_name}, {metric_name}",
                        showlegend=(col_idx == 1),
                        mode="lines+markers",
                        line={"color": color, "width": lw, "dash": dash},
                        marker={"symbol": marker, "size": 7, "color": color},
                        hovertemplate=(
                            f"<b>{kv_name}, {metric_name}</b>"
                            f"<br>{x_title}=%{{x}}<br>{metric_name}=%{{y:.3g}} s"
                            "<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )

        fig.update_xaxes(type="log", title_text=x_title, row=r, col=c)
        fig.update_yaxes(type="log", row=r, col=c)

    # Label every subplot's y-axis in mobile mode (stacked vertically, each
    # needs its own label); in desktop mode label only the leftmost (the three
    # share one y-scale, so labeling each is redundant).
    for r in range(1, len(rows_spec) + 1) if mobile else (1,):
        fig.update_yaxes(title_text="Time (s)", row=r, col=1)

    _finalize(
        fig,
        title="Figure 1 — KV cache: predict and fit+predict time vs dataset size",
        subtitle="GPU (NVIDIA L4), 10 classes, n_estimators=4.",
        height=560,
        legend_title=None,
        gallery=gallery,
        plot_area_px=600 if mobile else 240,
        mobile=mobile,
        footnote=_tabicl_version_label(sub),
        theme=theme,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Peak VRAM vs dataset size (kv cache OFF vs ON)
# ---------------------------------------------------------------------------


def build_fig2_vram(
    df: pd.DataFrame,
    gallery: bool = False,
    mobile: bool = False,
    theme: str = "light",
) -> go.Figure:
    """Peak VRAM vs three dataset-size axes, KV cache OFF vs ON.

    Layout: 1 row × 3 cols. Each cell plots three lines:

        * predict       — kv OFF (green, dashed)
        * predict       — kv ON  (vermillion, dashed)
        * fit           — kv ON  (vermillion, solid)

    ``kv OFF — fit`` is omitted: it is constant at ~110 MB across all axes
    (the estimator's small footprint, no kv cache built) and would just sit
    flat at the bottom of the log plot.

    Note: the kv cache is per-estimator — each estimator in the ensemble uses
    shifted columns, so cached context does not carry over across estimators.
    n_estimators is therefore fixed (to 4) here; sweeping it just scales both
    the cache cost and benefit linearly without amortization.

    Phase definitions:

        * ``fit``         = ``VRAM peak fit (MB)``  — incremental peak within
          the fit window only (the kv cache build lives here).
        * ``predict``     = ``VRAM peak predict (MB)`` — absolute footprint
          during predict = (VRAM retained at end of fit) + (incremental peak
          within the predict window). This is the real-workload inference
          footprint, including the resident kv cache.

      * panel 1: vs n_train     (rows sweep; n_features=100, n_test=2000)
      * panel 2: vs n_features  (cols sweep; n_train=1000,  n_test=2000)
      * panel 3: vs n_test      (rows sweep; n_train=4000,  n_features=100)

    n_estimators=4, classification, 10 classes, cuda, offload=gpu. Log axes.
    ``theme`` selects the colour token set (``"light"``/``"dark"``).
    """
    tokens = _theme(theme)
    sub = df[df["Run label"].str.contains(LABEL_SCALING, na=False)].copy()
    sub = sub[sub[NB_ESTIMATORS] == 4]

    # (col, x-axis column, filter dict, x-axis title, subplot title)
    rows_spec = [
        (
            1,
            NB_TRAIN_SAMPLES,
            {NB_FEATURES: 100, NB_TEST_SAMPLES: 2000},
            "Number of train rows",
            "VRAM vs number of train rows<br>(n_features=100, n_test=2000)",
        ),
        (
            2,
            NB_FEATURES,
            {NB_TRAIN_SAMPLES: 1000, NB_TEST_SAMPLES: 2000},
            "Number of features",
            "VRAM vs number of features<br>(n_train=1000, n_test=2000)",
        ),
        (
            3,
            NB_TEST_SAMPLES,
            {NB_TRAIN_SAMPLES: 4000, NB_FEATURES: 100},
            "Test-set size",
            "VRAM vs test-set size<br>(n_train=4000, n_features=100)",
        ),
    ]

    if mobile:
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            vertical_spacing=0.18,
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[title for _, _, _, _, title in rows_spec],
            horizontal_spacing=_hspacing(gallery),
        )

    # Per-kv metric sets. ``kv OFF — fit`` is omitted: it is constant at
    # ~110 MB across all axes (the estimator's small footprint, no kv cache
    # built), so it would just sit flat at the bottom of the log plot.
    #
    # ``trace_specs`` order doubles as the legend order: the two kv-cache
    # entries come first so they share the left legend column (predict on the
    # first row, fit on the second), the no-cache predict line fills the right
    # column.
    kv_style = {
        True: (tokens["kv_on"], MARKER_CACHE_ON, "kv cache"),
        False: (tokens["kv_off"], MARKER_CACHE_OFF, "no kv cache"),
    }
    trace_specs = [
        # (kv value, metric column, dash, width, metric_name)
        (True, VRAM_PEAK_PREDICT, DASH_PREDICT, 2.5, "predict"),
        (False, VRAM_PEAK_PREDICT, DASH_PREDICT, 2.5, "predict"),
        (True, VRAM_PEAK_FIT, "solid", 3.5, "fit"),
    ]

    for col_idx, x_col, filt, x_title, _ in rows_spec:
        cell = sub
        for k, v in filt.items():
            cell = cell[cell[k] == v]

        r, c = (col_idx, 1) if mobile else (1, col_idx)
        for kv, metric, dash, lw, metric_name in trace_specs:
            color, marker, kv_name = kv_style[kv]
            d = cell[cell[KV_CACHE] == kv].sort_values(x_col)
            fig.add_trace(
                go.Scatter(
                    x=d[x_col],
                    y=d[metric],
                    name=f"{kv_name}, {metric_name}",
                    legendgroup=f"{kv_name}, {metric_name}",
                    showlegend=(col_idx == 1),
                    mode="lines+markers",
                    line={"color": color, "width": lw, "dash": dash},
                    marker={"symbol": marker, "size": 10, "color": color},
                    hovertemplate=(
                        f"<b>{kv_name}, {metric_name}</b>"
                        f"<br>{x_title}=%{{x}}<br>VRAM {metric_name}=%{{y:.0f}} MB"
                        "<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(type="log", title_text=x_title, row=r, col=c)
        fig.update_yaxes(type="log", row=r, col=c)

    # Label every subplot's y-axis in mobile mode (stacked vertically, each
    # needs its own label); in desktop mode label only the leftmost (the three
    # share one y-scale, so labeling each is redundant).
    for r in range(1, len(rows_spec) + 1) if mobile else (1,):
        fig.update_yaxes(title_text="VRAM peak (MB)", row=r, col=1)

    _finalize(
        fig,
        title=(
            "Figure 2 — KV cache: peak device memory (fit, predict) vs dataset size"
        ),
        subtitle="GPU (NVIDIA L4), 10 classes, n_estimators=4.",
        height=560,
        legend_title=None,
        gallery=gallery,
        plot_area_px=600 if mobile else 240,
        mobile=mobile,
        footnote=_tabicl_version_label(sub),
        theme=theme,
    )
    return fig


# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------


def _legend_kwargs(
    legend_y: float,
    legend_yanchor: str,
    legend_title: str | None,
    gallery: bool,
    mobile: bool,
) -> dict:
    """Legend layout, shared by every figure.

    The swatch length (``itemwidth``) is identical for the desktop and mobile
    variants. The fixed entry width is desktop-only: on the mobile variant an
    entry would end up narrower than swatch + label and the labels would spill
    out of the figure, while natural wrapping already yields a tidy 2 + 2
    grid at that width.
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
    if gallery and not mobile:
        kwargs["entrywidthmode"] = "fraction"
        kwargs["entrywidth"] = LEGEND_ENTRYWIDTH
    return kwargs


def _finalize(
    fig: go.Figure,
    title: str,
    subtitle: str,
    height: int,
    legend_title: str | None = None,
    gallery: bool = False,
    extra_bottom: int = 0,
    plot_area_px: int = 240,
    mobile: bool = False,
    legend_offset_px: int | None = None,
    footnote: str | None = None,
    theme: str = "light",
) -> None:
    # Keep the plot area IDENTICAL to the HEAD baseline (240 px tall) so the
    # plot proportions never change. The figure title block (bold title +
    # ``<br>``-wrapped subtitle) is rendered as a *layout annotation* pinned
    # at the top of the margin band with ``yanchor="top"`` (grows down),
    # because plotly's ``layout.title`` is capped at y=1 (the domain top) and
    # a multi-line block would grow down from there into the subplot titles.
    #
    # The top margin must hold: the figure title block + a gap + the subplot
    # titles (which grow up from y=1). A 4-line subtitle needs more band than
    # a 2-line one, so ``t`` scales with the subtitle line count. To keep the
    # plot area fixed at 240 px while ``t`` changes, ``height`` is adjusted by
    # the same delta (height = 240 + t + b). So:
    #   - 2-line subtitle -> smaller t, smaller total height (less gap).
    #   - 4-line subtitle -> larger t, larger total height (more gap).
    #
    # Horizontal margins and y-axis title standoff depend on the target:
    # ``gallery`` tightens them for the narrow sphinx-gallery column, while
    # the default profile uses generous margins for direct/standalone viewing.
    n_subtitle_lines = subtitle.count("<br>") + 1
    # Per-line budgets tuned to plotly's actual rendered text metrics (font
    # 18 bold title ~30px/line, 13px subtitle ~22px/line, two-line subplot
    # title at font 14 ~48px). The gap between the title block and the subplot
    # titles scales up slightly with the subtitle line count so 4-row
    # subtitles get a bit more breathing room than 2-row ones.
    title_block_px = 30 + n_subtitle_lines * 22
    subplot_title_px = 48
    gap_px = 16 + n_subtitle_lines * 2
    # Bottom margin + legend position depend on the target. In gallery mode
    # the legend sits just below the plot and grows *downward* into the
    # bottom margin (anchored at top). This way, when the legend wraps to
    # multiple lines on a narrow screen, the extra lines grow down into the
    # margin (harmless) instead of up into the plot. In default mode the
    # legend stays at a comfortable mid-band position with a generous bottom
    # margin.
    if gallery:
        # Budget TWO legend rows plus the footnote. Since the swatches were
        # widened (LEGEND_ITEMWIDTH) a four-entry legend no longer fits on one
        # row in the gallery column, and the legend is anchored at the top of
        # the band while the footnote is anchored at its bottom, so the margin
        # must hold both without the two overlapping.
        bottom_margin = 148 + extra_bottom
        # Legend sits a fixed *pixel* offset below the plot domain (just past
        # the x-axis tick labels + title), converted to paper coords via the
        # plot area height. This keeps the pixel gap consistent across the
        # desktop (plot_area 240) and mobile (400-600) layouts.
        # ``extra_bottom`` and ``legend_offset_px`` let a caller push it lower
        # when a figure needs more room below the plot.
        legend_offset = (
            legend_offset_px if legend_offset_px is not None else 58 + extra_bottom
        )
        legend_y = -legend_offset / plot_area_px
        legend_yanchor = "top"
    else:
        bottom_margin = 160 + extra_bottom
        legend_y = -0.22 - extra_bottom / 240.0
        legend_yanchor = "top"
    top_margin = title_block_px + gap_px + subplot_title_px
    height = plot_area_px + top_margin + bottom_margin
    band_top = 1 + top_margin / plot_area_px  # paper y of the figure's top edge
    left_margin, right_margin = _GALLERY_LR if gallery else _DEFAULT_LR
    y_standoff = _GALLERY_YSTANDOFF if gallery else _DEFAULT_YSTANDOFF
    tokens = _theme(theme)
    # Gallery embeds sit on the documentation page, whose background the CSS
    # swaps with the theme: keep the paper transparent so the figure inherits
    # it. Standalone HTML opens on a blank browser page, so it needs the
    # theme's own opaque background (otherwise dark text on white, or light
    # text on white, is unreadable).
    paper_bgcolor = "rgba(0,0,0,0)" if gallery else tokens["background"]
    # Capture the existing (subplot-title) annotations BEFORE adding the
    # figure-title annotation, so the baseline-positioning loop below only
    # touches the subplot titles, not the figure title.
    n_subplot_anns = len(fig.layout.annotations)
    # Render the figure title block as an annotation pinned at the band top.
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
        # layout.title left empty: the figure title is the annotation above.
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=paper_bgcolor,
        # Legend text (and any unstyled text) follows the theme; plotly's
        # default is a dark navy that vanishes on a dark background.
        font={"color": tokens["text"]},
        hoverlabel={
            "bgcolor": tokens["hover_bg"],
            "bordercolor": tokens["hover_border"],
            "font": {"color": tokens["text"]},
        },
        hovermode="closest",
        legend=_legend_kwargs(
            legend_y=legend_y,
            legend_yanchor=legend_yanchor,
            legend_title=legend_title,
            gallery=gallery,
            mobile=mobile,
        ),
        margin={
            "l": left_margin,
            "r": right_margin,
            "t": top_margin,
            "b": bottom_margin,
        },
        height=height,
    )
    # Subplot titles: keep the baseline positioning (lifted +0.03, growing
    # up). Do NOT reposition them — moving them changes the plot's apparent
    # proportions. Only touch the pre-existing annotations (by index), not
    # the figure-title annotation we just appended.
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
    # Link all subplot y-axes to the first (shared y scale across subplots):
    # matched axes autorange to the union of their data, so every subplot
    # uses the same vertical scale and zooms together.
    for ax_key in fig.layout.to_plotly_json():
        if ax_key.startswith("yaxis") and ax_key != "yaxis":
            getattr(fig.layout, ax_key).matches = "y"
    fig.update_xaxes(
        zeroline=False,
        gridcolor=tokens["grid"],
        title_font={"size": 13, "color": tokens["text"]},
        tickfont={"size": 11, "color": tokens["tick"]},
    )
    # Small footnote (e.g. the TabICL version) pinned at the bottom-right of
    # the figure, below the legend. The legend is centered, anchored at top,
    # and grows *downward* into the bottom margin; the footnote is anchored at
    # the bottom of the margin and grows upward, so the two never collide
    # (the bottom margin is sized to hold the legend, leaving the lowest band
    # free for the note). ``bottom_margin`` is in pixels; convert to paper
    # coords via the fixed plot area height.
    if footnote:
        note_color = tokens["faint"]
        fig.add_annotation(
            text=f"<span style='font-size:10px;color:{note_color}'>{footnote}</span>",
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
FIGURES = {
    "fig1": build_fig1_kv_cache,
    "fig2": build_fig2_vram,
}

# Output order of the variants: plotly is loaded from the CDN once, by the
# first snippet; the other three reuse it.
VARIANTS = ((False, "light"), (False, "dark"), (True, "light"), (True, "dark"))

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _snippet(fig: go.Figure, *, slug: str, include_plotlyjs: bool) -> str:
    """Render ``fig`` as an embeddable <div>+<script> block.

    ``slug`` replaces the random element id plotly generates, so that an
    unchanged figure renders byte-identically and the file is left untouched.
    """
    html = fig.to_html(
        full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False
    )
    return _UUID.sub(slug, html)


def build_gallery_html(df: pd.DataFrame, name: str, build) -> str:
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
    """(Re)write the gallery snippets of every figure; return the written paths."""
    df = load_results(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
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
    out_dir = Path(app.srcdir) / "_static" / "compute_tradeoffs"
    generate(Path(app.srcdir) / "_data" / DEFAULT_CSV.name, out_dir, verbose=False)


def setup(app):
    """Sphinx entry point: regenerate the snippets at the start of each build."""
    app.connect("builder-inited", _generate_at_build)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}


def main(argv: list[str] | None = None) -> None:
    """Regenerate the snippets from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    generate(args.csv, args.out)


if __name__ == "__main__":
    main()

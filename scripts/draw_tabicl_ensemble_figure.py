from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"


COLORS = {
    "input": "#F7F7F2",
    "prep": "#E8F1F2",
    "perm": "#F3E7D3",
    "member": "#E9E6F7",
    "model": "#E3F0DF",
    "agg": "#F6E4E8",
    "ink": "#232323",
    "muted": "#666666",
    "line": "#474747",
}


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    face,
    edge=None,
    fontsize=8.8,
    weight="regular",
    align="center",
    linestyle="-",
    lw=1.25,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.18,rounding_size=0.55",
        linewidth=lw,
        edgecolor=edge or COLORS["line"],
        facecolor=face,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight=weight,
        linespacing=1.22,
    )
    return patch


def add_arrow(ax, start, end, rad=0.0, lw=1.25, style="-", color=None, mutation_scale=10):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=style,
        color=color or COLORS["line"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arr)
    return arr


def add_section_label(ax, x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )


def matrix_glyph(ax, x, y, cols, rows=3, cell=1.05, gap=0.08, colors=None):
    colors = colors or ["#6BAED6", "#74C476", "#FDAE6B", "#9E9AC8", "#F768A1"]
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (x + c * (cell + gap), y - r * (cell + gap)),
                    cell,
                    cell,
                    linewidth=0.35,
                    edgecolor="white",
                    facecolor=colors[c % len(colors)],
                )
            )


def draw_member_row(ax, y, idx, norm, pi_x, pi_y, color):
    add_box(
        ax,
        5,
        y,
        13.5,
        5.2,
        f"member {idx}\n$m={norm}$",
        "#FFFFFF",
        edge=color,
        fontsize=8.2,
        weight="bold",
    )
    add_box(
        ax,
        22,
        y,
        17.5,
        5.2,
        "$T_m(X)$\nfit on train, apply to train/test",
        COLORS["prep"],
        fontsize=7.5,
    )
    add_box(
        ax,
        43,
        y,
        19,
        5.2,
        f"feature reorder\n$X^e = T_m(X)[:, {pi_x}]$",
        COLORS["perm"],
        fontsize=7.5,
    )
    add_box(
        ax,
        66,
        y,
        18,
        5.2,
        f"label remap\n$y^e = {pi_y}[y]$",
        COLORS["agg"],
        fontsize=7.5,
    )
    add_arrow(ax, (18.5, y + 2.6), (22, y + 2.6))
    add_arrow(ax, (39.5, y + 2.6), (43, y + 2.6))
    add_arrow(ax, (62, y + 2.6), (66, y + 2.6))
    matrix_glyph(ax, 44.2, y + 3.9, cols=4, rows=2, cell=0.62)
    ax.text(51.4, y + 4.32, "$\\pi_x$", fontsize=8.5, ha="center", va="center", color=COLORS["ink"])


def available_font_family():
    names = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["PingFang HK", "Songti SC", "Arial Unicode MS", "STHeiti", "DejaVu Sans"]
    return [name for name in preferred if name in names]


def draw_member_row_zh(ax, y, idx, norm, pi_x, pi_y, color):
    add_box(
        ax,
        5,
        y,
        13.5,
        5.2,
        f"成员 {idx}\n$m={norm}$",
        "#FFFFFF",
        edge=color,
        fontsize=8.2,
        weight="bold",
    )
    add_box(
        ax,
        22,
        y,
        17.5,
        5.2,
        "$T_m(X)$\n在训练集拟合\n同时作用于 train/test",
        COLORS["prep"],
        fontsize=7.0,
    )
    add_box(
        ax,
        43,
        y,
        19,
        5.2,
        f"特征列重排\n$X^e = T_m(X)[:, {pi_x}]$",
        COLORS["perm"],
        fontsize=7.1,
    )
    add_box(
        ax,
        66,
        y,
        18,
        5.2,
        f"标签重映射\n$y^e = {pi_y}[y]$",
        COLORS["agg"],
        fontsize=7.1,
    )
    add_arrow(ax, (18.5, y + 2.6), (22, y + 2.6))
    add_arrow(ax, (39.5, y + 2.6), (43, y + 2.6))
    add_arrow(ax, (62, y + 2.6), (66, y + 2.6))
    matrix_glyph(ax, 44.2, y + 3.9, cols=4, rows=2, cell=0.62)
    ax.text(51.4, y + 4.32, "$\\pi_x$", fontsize=8.5, ha="center", va="center", color=COLORS["ink"])


def main():
    plt.rcParams.update(
        {
            "font.family": available_font_family(),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(14.2, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(
        50,
        63.6,
        "TabICL classifier ensemble: marginalizing feature order and class-index order",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        50,
        61.4,
        "Derived from src/tabicl/sklearn/classifier.py and EnsembleGenerator in preprocessing.py",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    # Fit-time pipeline.
    add_section_label(ax, 3, 58.2, "(a) Fit-time construction")
    add_box(ax, 3, 49.7, 15.5, 6.4, "raw data\n$X_{train}, y_{raw}$", COLORS["input"], fontsize=8.6, weight="bold")
    add_box(
        ax,
        22,
        49.7,
        17,
        6.4,
        "encoders\nLabelEncoder: $y \\to [0,C)$\nTransformToNumerical",
        COLORS["prep"],
        fontsize=7.8,
    )
    add_box(
        ax,
        42.5,
        49.7,
        18.5,
        6.4,
        "feature filter + preprocessors\nUniqueFeatureFilter\n$T_m$: scaler, norm, outlier clip",
        COLORS["prep"],
        fontsize=7.4,
    )
    add_box(
        ax,
        65,
        49.7,
        29.5,
        6.4,
        "ensemble configs\n$\\Pi_x = Shuffler(F, feat\\_method)$\n$\\Pi_y = Shuffler(C, class\\_method)$\nshuffle product with norm methods, truncate to $E$",
        COLORS["perm"],
        fontsize=7.4,
    )
    add_arrow(ax, (18.5, 52.9), (22, 52.9))
    add_arrow(ax, (39, 52.9), (42.5, 52.9))
    add_arrow(ax, (61, 52.9), (65, 52.9))

    add_box(
        ax,
        4,
        41.5,
        34,
        5.5,
        "Shuffler options\nidentity | circular shift | sampled random | Latin-square rows\nDefaults: norm=[none, power], feature=latin, class=shift",
        "#FFFFFF",
        edge="#9A9A9A",
        fontsize=6.9,
    )
    add_box(
        ax,
        42,
        41.5,
        52.5,
        5.5,
        "Each selected config is a tuple $e=(m, \\pi_x^e, \\pi_y^e)$.\n$\\pi_x^e$ reorders columns; $\\pi_y^e$ maps original encoded labels to the model-visible label indices.",
        "#FFFFFF",
        edge="#9A9A9A",
        fontsize=7.7,
    )
    add_arrow(ax, (79.8, 49.7), (79.8, 47.1), style="--", color="#8A8A8A")

    # Per-member views.
    add_section_label(ax, 3, 38.6, "(b) Per-member dataset views")
    draw_member_row(ax, 31.6, 1, "none", "[2,0,3,1]", "[1,2,0]", "#5B8DEF")
    draw_member_row(ax, 24.6, 2, "power", "[0,3,1,2]", "[2,0,1]", "#D08C36")
    draw_member_row(ax, 17.6, 3, "...", "...", "...", "#7C6DC7")
    add_box(
        ax,
        85.6,
        20.3,
        11.6,
        12.2,
        "",
        "#FFFFFF",
        edge="#9A9A9A",
        fontsize=6.8,
    )
    ax.text(91.4, 29.1, "stack outputs", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    ax.text(91.4, 28.0, "by norm", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    ax.text(91.4, 24.8, "$X_s$: $E_m \\times N \\times F$", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    ax.text(91.4, 23.6, "$y_s$: $E_m \\times N_{tr}$", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    # Predict-time forward and aggregation.
    add_section_label(ax, 3, 14.2, "(c) Predict-time alignment and aggregation")
    add_box(
        ax,
        5,
        6.4,
        18,
        5.9,
        "forward batch\nTabICL($X^e$, $y^e$,\nfeature_shuffles=$\\pi_x^e$)",
        COLORS["model"],
        fontsize=7.6,
        weight="bold",
    )
    add_box(
        ax,
        28,
        6.4,
        18.5,
        5.9,
        "member output\n$z^e \\in \\mathbb{R}^{N_{te}\\times C}$\ncolumns are remapped labels",
        "#FFFFFF",
        edge="#8AA879",
        fontsize=7.35,
    )
    add_box(
        ax,
        51,
        6.4,
        20,
        5.9,
        "class realignment\n$q^e_k = z^e_{\\pi_y^e(k)}$\ncode: out[..., shuffle]",
        COLORS["agg"],
        fontsize=7.55,
        weight="bold",
    )
    add_box(
        ax,
        75.5,
        6.4,
        19,
        5.9,
        "ensemble prediction\n$\\bar q = \\frac{1}{E}\\sum_e q^e$\nsoftmax if logits, normalize\ninverse LabelEncoder",
        COLORS["input"],
        fontsize=7.35,
        weight="bold",
    )
    add_arrow(ax, (23, 9.35), (28, 9.35))
    add_arrow(ax, (46.5, 9.35), (51, 9.35))
    add_arrow(ax, (71, 9.35), (75.5, 9.35))
    ax.text(
        14,
        13.0,
        "batched by norm method",
        fontsize=7.1,
        ha="center",
        va="center",
        color=COLORS["muted"],
    )

    # Optional cache branch.
    add_box(
        ax,
        4,
        1.0,
        42,
        4.1,
        "Optional KV cache\nfit: transform(mode='train') caches training views\npredict: transform(mode='test') unless feature_mask is active",
        "#FFFFFF",
        edge="#B3B3B3",
        fontsize=6.7,
        linestyle="--",
    )
    add_box(
        ax,
        51,
        1.0,
        43.5,
        4.1,
        "Core invariance\naverage over views with different feature orders and label-index orders\nafter undoing the label-index permutation",
        "#FFFFFF",
        edge="#B3B3B3",
        fontsize=6.7,
        linestyle="--",
    )

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "tabicl_ensemble_pipeline.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "tabicl_ensemble_pipeline.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "tabicl_ensemble_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main_zh():
    plt.rcParams.update(
        {
            "font.family": available_font_family(),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(14.2, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(
        50,
        63.6,
        "TabICL 分类器 Ensemble：边缘化特征顺序与类别编号顺序",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        50,
        61.4,
        "基于 src/tabicl/sklearn/classifier.py 与 preprocessing.py 中的 EnsembleGenerator",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    add_section_label(ax, 3, 58.2, "(a) 拟合阶段：构造 ensemble 配置")
    add_box(ax, 3, 49.7, 15.5, 6.4, "原始数据\n$X_{train}, y_{raw}$", COLORS["input"], fontsize=8.6, weight="bold")
    add_box(
        ax,
        22,
        49.7,
        17,
        6.4,
        "编码器\nLabelEncoder: $y \\to [0,C)$\nTransformToNumerical",
        COLORS["prep"],
        fontsize=7.5,
    )
    add_box(
        ax,
        42.5,
        49.7,
        18.5,
        6.4,
        "特征过滤 + 预处理器\nUniqueFeatureFilter\n$T_m$: 缩放/归一化/异常值裁剪",
        COLORS["prep"],
        fontsize=7.0,
    )
    add_box(
        ax,
        65,
        49.7,
        29.5,
        6.4,
        "ensemble 配置\n$\\Pi_x = Shuffler(F, feat\\_method)$\n$\\Pi_y = Shuffler(C, class\\_method)$\n与 norm method 组合，打乱后截断到 $E$ 个",
        COLORS["perm"],
        fontsize=7.1,
    )
    add_arrow(ax, (18.5, 52.9), (22, 52.9))
    add_arrow(ax, (39, 52.9), (42.5, 52.9))
    add_arrow(ax, (61, 52.9), (65, 52.9))

    add_box(
        ax,
        4,
        41.5,
        34,
        5.5,
        "Shuffler 选项\n不打乱 | 循环平移 | 随机排列 | Latin-square 行\n默认：norm=[none, power], feature=latin, class=shift",
        "#FFFFFF",
        edge="#9A9A9A",
        fontsize=6.7,
    )
    add_box(
        ax,
        42,
        41.5,
        52.5,
        5.5,
        "每个被选中的配置都是 $e=(m, \\pi_x^e, \\pi_y^e)$。\n$\\pi_x^e$ 重排列；$\\pi_y^e$ 把原始 encoded label 映射成模型看到的类别编号。",
        "#FFFFFF",
        edge="#9A9A9A",
        fontsize=7.2,
    )
    add_arrow(ax, (79.8, 49.7), (79.8, 47.1), style="--", color="#8A8A8A")

    add_section_label(ax, 3, 38.6, "(b) 每个成员生成一个数据视图")
    draw_member_row_zh(ax, 31.6, 1, "none", "[2,0,3,1]", "[1,2,0]", "#5B8DEF")
    draw_member_row_zh(ax, 24.6, 2, "power", "[0,3,1,2]", "[2,0,1]", "#D08C36")
    draw_member_row_zh(ax, 17.6, 3, "...", "...", "...", "#7C6DC7")
    add_box(
        ax,
        85.6,
        20.3,
        11.6,
        12.2,
        "",
        "#FFFFFF",
        edge="#9A9A9A",
        fontsize=6.8,
    )
    ax.text(91.4, 29.1, "按 norm", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    ax.text(91.4, 28.0, "堆叠输出", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    ax.text(91.4, 24.8, "$X_s$: $E_m \\times N \\times F$", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
    ax.text(91.4, 23.6, "$y_s$: $E_m \\times N_{tr}$", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])

    add_section_label(ax, 3, 14.2, "(c) 预测阶段：类别对齐与聚合")
    add_box(
        ax,
        5,
        6.4,
        18,
        5.9,
        "批量前向\nTabICL($X^e$, $y^e$,\nfeature_shuffles=$\\pi_x^e$)",
        COLORS["model"],
        fontsize=7.1,
        weight="bold",
    )
    add_box(
        ax,
        28,
        6.4,
        18.5,
        5.9,
        "单成员输出\n$z^e \\in \\mathbb{R}^{N_{te}\\times C}$\n列含义是重映射后的类别",
        "#FFFFFF",
        edge="#8AA879",
        fontsize=6.9,
    )
    add_box(
        ax,
        51,
        6.4,
        20,
        5.9,
        "类别顺序对齐\n$q^e_k = z^e_{\\pi_y^e(k)}$\n代码：out[..., shuffle]",
        COLORS["agg"],
        fontsize=7.1,
        weight="bold",
    )
    add_box(
        ax,
        75.5,
        6.4,
        19,
        5.9,
        "ensemble 预测\n$\\bar q = \\frac{1}{E}\\sum_e q^e$\n若平均 logits 则 softmax\n归一化并 inverse LabelEncoder",
        COLORS["input"],
        fontsize=6.8,
        weight="bold",
    )
    add_arrow(ax, (23, 9.35), (28, 9.35))
    add_arrow(ax, (46.5, 9.35), (51, 9.35))
    add_arrow(ax, (71, 9.35), (75.5, 9.35))
    ax.text(14, 13.0, "按 norm method 分批", fontsize=7.0, ha="center", va="center", color=COLORS["muted"])

    add_box(
        ax,
        4,
        1.0,
        42,
        4.1,
        "可选 KV cache\nfit: transform(mode='train') 缓存训练视图\npredict: 无 feature_mask 时只做 transform(mode='test')",
        "#FFFFFF",
        edge="#B3B3B3",
        fontsize=6.5,
        linestyle="--",
    )
    add_box(
        ax,
        51,
        1.0,
        43.5,
        4.1,
        "核心不变性\n对不同特征顺序、不同类别编号顺序的等价视图求平均\n但平均前必须先撤销类别编号 permutation",
        "#FFFFFF",
        edge="#B3B3B3",
        fontsize=6.5,
        linestyle="--",
    )

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "tabicl_ensemble_pipeline_zh.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "tabicl_ensemble_pipeline_zh.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "tabicl_ensemble_pipeline_zh.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
    main_zh()

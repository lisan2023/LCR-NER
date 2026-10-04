"""
Top-K 候选路径覆盖率分析图
- 蓝色柱状图：最优比例 (%)
- 红色折线：累计覆盖率 (%)
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_topk_coverage(
    ranks=("Top-1", "Top-2", "Top-3", "Top-4", "Top-5"),
    bar_values=(67.90, 19.30, 7.30, 2.80, 2.80),
    line_values=(67.90, 87.20, 94.50, 97.30, 100.00),
    save_path="topk_coverage.png",
    show=True,
):
    # ---------- 中文字体（兼容 macOS / Windows / Linux）----------
    try:
        plt.rcParams["font.sans-serif"] = [
            "PingFang SC",        # macOS
            "Microsoft YaHei",    # Windows
            "WenQuanYi Zen Hei",  # Linux
            "Noto Sans CJK SC",
            "SimHei",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # ---------- 数据 ----------
    x = np.arange(len(ranks))
    bar_values = np.asarray(bar_values, dtype=float)
    line_values = np.asarray(line_values, dtype=float)

    bar_color = "#4A7DC9"   # 蓝色柱
    line_color = "#D62728"  # 红色折线

    # ---------- 画布 ----------
    # 缩到原尺寸的 30%（之前是 (9, 5.5)，现在 (9*0.30, 5.5*0.30)）。
    # 只缩 figsize 像素数才会真正按比例缩小；只缩 dpi 只是把字号放大、
    # 像素密度变小，但物理尺寸不变（在 1× 屏幕上看起来一样）。
    fig, ax = plt.subplots(figsize=(9*0.6, 5.5*0.6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---------- 柱状图 ----------
    bars = ax.bar(
        x, bar_values,
        width=0.55,
        color=bar_color,
        edgecolor="none",
        label="最优比例 (%)",
        zorder=2,
    )

    # 在柱顶标数值（与折线点标分离，避免重叠）
    for i, (rect, v) in enumerate(zip(bars, bar_values)):
        # Top-1 因柱顶紧贴折线点，标在柱体右上角；其余居中放在柱顶上方
        if i == 0:
            ax.text(
                rect.get_x() + rect.get_width() - 0.04,
                rect.get_height() - 3.5,
                f"{v:.2f}",
                ha="right", va="top",
                fontsize=10, color="#333",
            )
        else:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 1.5,
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=10, color="#333",
            )

    # ---------- 折线图（共用 y 轴）----------
    ax.plot(
        x, line_values,
        color=line_color,
        linewidth=1.8,
        marker="o",
        markersize=6,
        markerfacecolor=line_color,
        markeredgecolor=line_color,
        label="累计覆盖率 (%)",
        zorder=3,
    )
    # 折线上方标数值
    for xi, yi in zip(x, line_values):
        ax.text(
            xi, yi + 2.2, f"{yi:.2f}",
            ha="center", va="bottom",
            fontsize=10, color="#222",
        )

    # ---------- 坐标轴样式 ----------
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("比例 (%)", fontsize=11)
    ax.set_xlabel("候选路径排序", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ranks, fontsize=10)

    # 仅保留水平网格
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)

    # 去掉上、右边框
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888")
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(axis="both", which="both", length=0, colors="#555")

    # ---------- 图例（顶部居中，按"柱-线"顺序：先最优比例，再累计覆盖率）----------
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("最优比例 (%)"), labels.index("累计覆盖率 (%)")]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=2.5,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=int(200), bbox_inches="tight", facecolor="white")
    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    plot_topk_coverage(
        ranks=("1", "2", "3", "4", "5"),
        bar_values=(67.90, 19.30, 7.30, 2.80, 2.80),
        line_values=(67.90, 87.20, 94.50, 97.30, 100.00),
        save_path="topk_coverage.png",
        show=True,
    )

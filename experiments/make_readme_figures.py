"""生成 README 用结果图（figures/），IEEE 学术风格。

约定（IEEE Transactions 惯例）：
- serif 字体（Times New Roman 优先）、无网格、仅左/下坐标轴
- 图内无标题（caption 在文档中）、图例无边框且避开数据
- 色盲友好高对比配色、300 dpi
- 数值标注置于图形元素上方留白处，避免与数据/误差线重叠

输入：data/results 下已完成的实验 CSV/JSON。
输出：figures/*.png（提交到仓库，供中英文 README 使用）。
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
VERIFY_DIR = ROOT / "data" / "results" / "verification_20260814_190641"
LLM_DIR = ROOT / "data" / "results" / "llm_agent_20260816_205230"
META_DIR = ROOT / "data" / "results" / "meta_20260816_190602"
OUT = ROOT / "figures"

# IEEE 风格色盲友好配色（Okabe-Ito）
C_RED = "#D55E00"
C_BLUE = "#0072B2"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_GREY = "#7F7F7F"
C_BLACK = "#000000"

METHOD_COLORS = {
    "evoagent": C_RED,
    "llm_evolve": C_RED,
    "llm_sew": C_PURPLE,
    "llm_fixed": C_ORANGE,
    "phase1": C_BLUE,
    "cma_es": C_BLUE,
    "random_search": C_GREY,
    "sa": C_GREEN,
    "ga": C_ORANGE,
    "bo": C_PURPLE,
    "nsga2": C_BLUE,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.titlesize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "stix",
        }
    )


def _legend(ax, loc: str = "best", ncol: int = 1) -> None:
    """IEEE 图例：无边框，透明背景。"""
    ax.legend(loc=loc, frameon=False, handlelength=2.2, borderaxespad=0.3, ncol=ncol)


def _legend_below(ax, ncol: int = 3) -> None:
    """图例置于轴外下方（横向），避免遮挡任何数据。"""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol,
              frameon=False, handlelength=2.0, columnspacing=1.4)


def single_objective_chart() -> None:
    """单目标：EvoAgent vs 最佳基线（相对提升率标注）。"""
    data = [
        ("semiconductor", 0.074566, 0.066693, "+11.8%"),
        ("rosenbrock", -19.888, -126.325, "+84.3%"),
        ("ackley", -0.791, -3.395, "+76.7%"),
        ("rastrigin", -4.032, -15.263, "+73.6%"),
    ]
    names = [d[0] for d in data]
    x = np.arange(len(names))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w / 2, [d[1] for d in data], w, label="EvoAgent", color=C_RED, edgecolor="none")
    ax.bar(x + w / 2, [d[2] for d in data], w, label="Best baseline (CMA-ES)", color=C_BLUE, edgecolor="none")
    for i, (_, ev, bs, improve) in enumerate(data):
        top = max(ev, bs)
        # 数值标注放在两组柱上方的留白处，符号与柱方向一致避免重叠
        offset = 0.045 * (abs(top) + 0.6)
        ax.text(x[i], top + offset, improve, ha="center", va="bottom",
                fontsize=8.5, color=C_BLACK)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean clean fitness (higher is better)")
    ax.set_ylim(bottom=ax.get_ylim()[0] - 0.5)
    _legend(ax, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "phase1_single_objective.png", bbox_inches="tight")
    plt.close(fig)


def multi_objective_chart() -> None:
    """多目标：超体积对比。"""
    data = [
        ("zdt1", 0.995074, 0.944246, "+5.4%"),
        ("semiconductor_2obj", 0.141663, 0.121431, "+16.7%"),
    ]
    names = [d[0] for d in data]
    x = np.arange(len(names))
    w = 0.3
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w / 2, [d[1] for d in data], w, label="EvoAgent", color=C_RED, edgecolor="none")
    ax.bar(x + w / 2, [d[2] for d in data], w, label="NSGA-II", color=C_BLUE, edgecolor="none")
    for i, (_, ev, bs, improve) in enumerate(data):
        ax.text(x[i], max(ev, bs) + 0.012, improve, ha="center", va="bottom",
                fontsize=8.5, color=C_BLACK)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Hypervolume (higher is better)")
    ax.set_xlim(-0.6, len(names) - 0.4)
    _legend_below(ax, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "phase1_multi_objective.png", bbox_inches="tight")
    plt.close(fig)


def phase2_chart() -> None:
    """阶段二/三：提示词进化 vs SEW 双模式 vs 固定提示词 vs 阶段一（真实 DeepSeek）。"""
    modes = ["llm_evolve", "llm_sew", "llm_fixed", "phase1"]
    labels = [
        "LLM evolve\n(evolved prompts)",
        "SEW dual-mode\n(structure + prompt)",
        "LLM fixed\n(fixed prompt)",
        "Phase 1\n(no LLM)",
    ]
    means = [0.071614, 0.068668, 0.067320, 0.070018]
    stds = [0.006002, 0.004511, 0.003173, 0.007152]
    colors = [C_RED, C_PURPLE, C_ORANGE, C_BLUE]
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.bar(x, means, yerr=stds, capsize=3.5, width=0.58,
                  color=colors, edgecolor="none", error_kw=dict(lw=0.9))
    # 领先标注：条形顶端上方留白，无箭头避免遮挡
    ax.text(0, means[0] + stds[0] + 0.004, "+2.3% vs phase1",
            ha="center", va="bottom", fontsize=8.5, color=C_BLACK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean clean fitness (higher is better)")
    ax.set_ylim(0.055, 0.092)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_llm_comparison.png", bbox_inches="tight")
    plt.close(fig)


def convergence_charts() -> None:
    """半导体收敛曲线（阶段一）与 LLM 收敛曲线（阶段二）。"""
    df = np.loadtxt(VERIFY_DIR / "single_semiconductor_curves.csv", delimiter=",", skiprows=1)
    x = df[:, 0]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, name in enumerate(["random_search", "sa", "ga", "cma_es", "bo", "evoagent"], start=1):
        ax.plot(x, df[:, i], label=name, color=METHOD_COLORS[name],
                linewidth=2.0 if name == "evoagent" else 1.2,
                linestyle="-" if name in ("evoagent", "cma_es", "bo") else "--")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best fitness (mean of 3 seeds)")
    _legend_below(ax, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "semiconductor_convergence.png", bbox_inches="tight")
    plt.close(fig)

    df2 = np.loadtxt(LLM_DIR / "llm_curves.csv", delimiter=",", skiprows=1)
    x2 = df2[:, 0]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, name in enumerate(["llm_evolve", "llm_sew", "llm_fixed", "phase1"], start=1):
        ax.plot(x2, df2[:, i], label=name, color=METHOD_COLORS[name],
                linewidth=2.0 if name == "llm_evolve" else 1.3,
                marker="o", markersize=3.5, markerfacecolor=METHOD_COLORS[name],
                markeredgewidth=0.4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (mean of 3 seeds)")
    _legend_below(ax, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "llm_convergence.png", bbox_inches="tight")
    plt.close(fig)


def meta_chart() -> None:
    """Meta 层：默认超参 vs BO 最优超参收敛对比。"""
    df = np.loadtxt(META_DIR / "meta_curves.csv", delimiter=",", skiprows=1) if (META_DIR / "meta_curves.csv").exists() else None
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(df[:, 0], df[:, 1], label="default", color=C_GREY, linewidth=1.3,
            marker="o", markersize=4, markerfacecolor=C_GREY)
    ax.plot(df[:, 0], df[:, 2], label="meta-optimized", color=C_RED, linewidth=2.0,
            marker="o", markersize=4, markerfacecolor=C_RED)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (mean of 3 seeds)")
    _legend_below(ax, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "meta_convergence.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    single_objective_chart()
    multi_objective_chart()
    phase2_chart()
    convergence_charts()
    meta_chart()
    print(f"figures saved to {OUT}")


if __name__ == "__main__":
    main()
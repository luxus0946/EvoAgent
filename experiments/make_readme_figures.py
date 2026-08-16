"""生成 README 用结果图（figures/）。

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
LLM_DIR = ROOT / "data" / "results" / "llm_agent_20260815_141731"
META_DIR = ROOT / "data" / "results" / "meta_20260816_190602"
OUT = ROOT / "figures"

METHOD_COLORS = {
    "evoagent": "#d62728",
    "llm_evolve": "#d62728",
    "llm_sew": "#8c564b",
    "llm_fixed": "#ff9896",
    "phase1": "#1f77b4",
    "cma_es": "#1f77b4",
    "random_search": "#7f7f7f",
    "sa": "#2ca02c",
    "ga": "#ff7f0e",
    "bo": "#9467bd",
    "nsga2": "#1f77b4",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def single_objective_chart() -> None:
    """单目标：EvoAgent vs 最佳基线（含领先幅度标注）。"""
    data = [
        ("semiconductor", 0.074566, 0.066693, "+11.8%"),
        ("rosenbrock", -19.888, -126.325, "+84.3%"),
        ("ackley", -0.791, -3.395, "+76.7%"),
        ("rastrigin", -4.032, -15.263, "+73.6%"),
    ]
    names = [d[0] for d in data]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars1 = ax.bar(x - w / 2, [d[1] for d in data], w, label="EvoAgent", color="#d62728")
    bars2 = ax.bar(x + w / 2, [d[2] for d in data], w, label="Best baseline (CMA-ES)", color="#1f77b4")
    for i, (_, _, _, improve) in enumerate(data):
        y = max(data[i][1], data[i][2]) + 8
        ax.annotate(improve, (x[i], y), ha="center", fontsize=10, fontweight="bold", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean clean fitness (higher is better)")
    ax.set_title("Phase 1: EvoAgent vs best baseline (3 seeds)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "phase1_single_objective.png")
    plt.close(fig)


def multi_objective_chart() -> None:
    """多目标：超体积对比。"""
    data = [
        ("zdt1", 0.995074, 0.944246, "+5.4%"),
        ("semiconductor_2obj", 0.141663, 0.121431, "+16.7%"),
    ]
    names = [d[0] for d in data]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w / 2, [d[1] for d in data], w, label="EvoAgent", color="#d62728")
    ax.bar(x + w / 2, [d[2] for d in data], w, label="NSGA-II", color="#1f77b4")
    for i, (_, _, _, improve) in enumerate(data):
        ax.annotate(improve, (x[i], max(data[i][1], data[i][2]) + 0.01), ha="center",
                    fontsize=10, fontweight="bold", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Hypervolume (higher is better)")
    ax.set_title("Phase 1: Pareto multi-objective (3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "phase1_multi_objective.png")
    plt.close(fig)


def phase2_chart() -> None:
    """阶段二/三：提示词进化 vs SEW 双模式 vs 固定提示词 vs 阶段一。"""
    modes = ["llm_evolve", "llm_sew", "llm_fixed", "phase1"]
    labels = [
        "LLM evolve\n(evolved prompts)",
        "SEW dual-mode\n(structure + prompt)",
        "LLM fixed\n(fixed prompt)",
        "Phase 1\n(no LLM)",
    ]
    means = [0.072797, 0.068730, 0.070688, 0.069053]
    stds = [0.004031, 0.004386, 0.003604, 0.004443]
    colors = ["#d62728", "#8c564b", "#ff9896", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(modes, means, yerr=stds, capsize=5, color=colors, alpha=0.9)
    ax.annotate("+3.0%", (0, 0.074), xytext=(0, 0.078), ha="center",
                fontsize=10, fontweight="bold", color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xticks(modes)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean clean fitness (higher is better)")
    ax.set_title("Phase 2/3: prompt evolution vs SEW vs baselines (mock LLM, 3 seeds)")
    ax.set_ylim(0.05, 0.09)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_llm_comparison.png")
    plt.close(fig)


def convergence_charts() -> None:
    """半导体收敛曲线（阶段一）与 LLM 收敛曲线（阶段二）。"""
    df = np.loadtxt(VERIFY_DIR / "single_semiconductor_curves.csv", delimiter=",", skiprows=1)
    x = df[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, name in enumerate(["random_search", "sa", "ga", "cma_es", "bo", "evoagent"], start=1):
        ax.plot(x, df[:, i], label=name, color=METHOD_COLORS[name], linewidth=1.8 if name == "evoagent" else 1.2)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best fitness (mean of 3 seeds)")
    ax.set_title("Semiconductor: convergence comparison (Phase 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "semiconductor_convergence.png")
    plt.close(fig)

    df2 = np.loadtxt(LLM_DIR / "llm_curves.csv", delimiter=",", skiprows=1)
    x2 = df2[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, name in enumerate(["llm_evolve", "llm_sew", "llm_fixed", "phase1"], start=1):
        ax.plot(x2, df2[:, i], label=name, color=METHOD_COLORS[name],
                linewidth=2.0 if name == "llm_evolve" else 1.4, marker="o", markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (mean of 3 seeds)")
    ax.set_title("Semiconductor: prompt evolution vs SEW convergence (Phase 2/3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "llm_convergence.png")
    plt.close(fig)


def meta_chart() -> None:
    """Meta 层：默认超参 vs BO 最优超参收敛对比。"""
    df = np.loadtxt(META_DIR / "meta_curves.csv", delimiter=",", skiprows=1) if (META_DIR / "meta_curves.csv").exists() else None
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(df[:, 0], df[:, 1], label="default", color="#7f7f7f", linewidth=1.6, marker="o", markersize=4)
    ax.plot(df[:, 0], df[:, 2], label="meta_optimized", color="#d62728", linewidth=2.0, marker="o", markersize=4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (mean of 3 seeds)")
    ax.set_title("Meta layer: BO-optimized hyperparameters vs default")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "meta_convergence.png")
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
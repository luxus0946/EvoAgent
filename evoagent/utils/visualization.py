"""可视化：收敛曲线与 Pareto 前沿图。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_convergence_curves(
    curves: dict[str, np.ndarray],
    title: str,
    output_path: str | Path,
    xlabel: str = "评估次数",
    ylabel: str = "最优适应度",
) -> None:
    """绘制多条收敛曲线对比图。

    Args:
        curves: 方法名 -> 最优值随评估次数变化的曲线
        title: 图标题
        output_path: 保存路径（PNG）
        xlabel: x 轴标签
        ylabel: y 轴标签
    """
    plt.figure(figsize=(9, 6))
    for name, curve in curves.items():
        x = np.arange(1, len(curve) + 1)
        plt.plot(x, curve, label=name, linewidth=1.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path)


def plot_pareto_front(
    points: np.ndarray,
    title: str,
    output_path: str | Path,
    labels: tuple[str, str] = ("目标1", "目标2"),
) -> None:
    """绘制二维 Pareto 前沿散点图。

    Args:
        points: 非支配点集，shape (n, 2)，最大化约定
        title: 图标题
        output_path: 保存路径（PNG）
        labels: 两个目标的坐标轴标签
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=30, alpha=0.8)
    if len(points) > 1:
        order = np.argsort(points[:, 0])
        plt.plot(points[order, 0], points[order, 1], "r--", alpha=0.5)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path)


def _save_fig(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

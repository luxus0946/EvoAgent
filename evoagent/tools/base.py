"""优化工具抽象基类与公共数据结构。

所有工具统一签名：在给定评估预算内最大化加权标量适应度。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from evoagent.environment.problem import OptimizationProblem

# 工具注册表：策略基因与工具池共享的名称集合
TOOL_NAMES = ["random_search", "sa", "ga", "cma_es", "bo", "ppo"]


@dataclass
class ToolResult:
    """工具执行结果。"""

    best_params: np.ndarray
    best_fitness: float
    history: list[float] = field(default_factory=list)
    n_evals: int = 0
    n_improvements: int = 0

    @property
    def last_improved_eval(self) -> int:
        """最后一次改进发生的评估序号（1 起），未改进时为 0。"""
        return self.n_improvements


class EarlyStopMonitor:
    """早停监视器：连续无改进达到阈值时通知工具提前终止。"""

    def __init__(self, patience_evals: int):
        """初始化。

        Args:
            patience_evals: 连续无改进容忍的评估次数，<=0 表示不启用
        """
        self.patience = patience_evals
        self._streak = 0

    @property
    def enabled(self) -> bool:
        """是否启用早停。"""
        return self.patience > 0

    def check(self, current: float, best: float) -> bool:
        """评估一次迭代后检查是否需要终止。

        Args:
            current: 当前适应度
            best: 历史最优适应度

        Returns:
            True 表示应终止
        """
        if not self.enabled:
            return False
        if current > best + 1e-12:
            self._streak = 0
        else:
            self._streak += 1
        return self._streak >= self.patience


class OptimizationTool(ABC):
    """优化工具抽象基类，所有工具必须继承此类。"""

    name: str = "base_tool"

    @abstractmethod
    def optimize(
        self,
        problem: OptimizationProblem,
        budget: int,
        weights: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
        early_stop: EarlyStopMonitor | None = None,
        rng: np.random.Generator | None = None,
    ) -> ToolResult:
        """在预算内执行优化（最大化加权标量适应度）。

        Args:
            problem: 优化问题
            budget: 最大评估次数
            weights: 目标权重，None 时使用均匀权重
            x_init: 初始参数（历史最优），None 时算法自行初始化
            early_stop: 早停监视器，None 表示不启用
            rng: 随机数生成器

        Returns:
            ToolResult：最优参数、最优适应度、收敛曲线、评估次数
        """
        raise NotImplementedError

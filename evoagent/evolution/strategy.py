"""策略执行器：将个体的策略基因映射为一次完整的优化任务执行。

两阶段策略：
1. 第一阶段：用 initial_tool 运行 switch_after_ratio * budget 次评估；
2. 第二阶段：从当前最优出发，用 second_tool 运行剩余预算（工具不同时）。
若 switch_after_ratio >= 0.95 或两工具相同，则单阶段运行。
stop_patience 控制早停：连续无改进达到预算比例时提前终止。
"""

import numpy as np

from evoagent.core.individual import StrategyGenome
from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, ToolResult
from evoagent.tools.factory import build_tool


class StrategyExecutor:
    """在给定问题上执行一个策略基因。"""

    def __init__(self, problem: OptimizationProblem, weights: np.ndarray | None = None):
        """初始化。

        Args:
            problem: 优化问题
            weights: 标量化权重（None 时使用问题默认均匀权重）
        """
        self.problem = problem
        self.weights = weights

    def run(
        self,
        genome: StrategyGenome,
        budget: int,
        rng: np.random.Generator | None = None,
    ) -> ToolResult:
        """执行策略。

        Args:
            genome: 策略基因
            budget: 总评估预算
            rng: 随机数生成器

        Returns:
            ToolResult：各阶段合并后的最优结果与收敛曲线
        """
        if rng is None:
            rng = np.random.default_rng()
        switch = genome.switch_after_ratio
        phase1_evals = int(budget * switch)
        phase2_evals = budget - phase1_evals
        same_tool = genome.initial_tool == genome.second_tool
        single_phase = same_tool or phase1_evals < 3 or phase2_evals < 3

        results: list[ToolResult] = []
        patience = (
            max(5, int(budget * genome.stop_patience))
            if genome.stop_patience > 0
            else 0
        )

        if single_phase:
            result = self._run_tool(
                genome.initial_tool,
                genome.tool_params,
                budget,
                None,
                patience,
                rng,
            )
            results.append(result)
        else:
            r1 = self._run_tool(
                genome.initial_tool,
                genome.tool_params,
                phase1_evals,
                None,
                patience,
                rng,
            )
            results.append(r1)
            r2 = self._run_tool(
                genome.second_tool,
                genome.tool_params,
                phase2_evals,
                r1.best_params,
                patience,
                rng,
            )
            results.append(r2)

        return self._merge(results)

    def _run_tool(
        self,
        tool_name: str,
        tool_params: dict[str, float],
        budget: int,
        x_init: np.ndarray | None,
        patience: int,
        rng: np.random.Generator,
    ) -> ToolResult:
        tool = build_tool(tool_name, tool_params)
        return tool.optimize(
            problem=self.problem,
            budget=budget,
            weights=self.weights,
            x_init=x_init,
            early_stop=EarlyStopMonitor(patience),
            rng=rng,
        )

    @staticmethod
    def _merge(results: list[ToolResult]) -> ToolResult:
        """合并多阶段结果：取全局最优，拼接收敛曲线。"""
        best = max(results, key=lambda r: r.best_fitness)
        history: list[float] = []
        for r in results:
            history.extend(r.history)
        # 曲线拼接时保持单调（全局最优）
        monotonic: list[float] = []
        for v in history:
            if not monotonic or v > monotonic[-1]:
                monotonic.append(v)
            else:
                monotonic.append(monotonic[-1])
        return ToolResult(
            best_params=best.best_params,
            best_fitness=best.best_fitness,
            history=monotonic,
            n_evals=sum(r.n_evals for r in results),
            n_improvements=sum(r.n_improvements for r in results),
        )

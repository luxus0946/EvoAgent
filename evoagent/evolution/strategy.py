"""Strategy executor: maps an individual's strategy genome to a complete optimization task run.

Two-phase strategy:
1. Phase 1: run initial_tool for switch_after_ratio * budget evaluations;
2. Phase 2: continue from the current best with second_tool for the remaining budget (when the tools differ).
If switch_after_ratio >= 0.95 or the two tools are identical, run as a single phase.
stop_patience controls early stopping: terminate early when consecutive non-improvements reach the budget ratio.
"""

import numpy as np

from evoagent.core.individual import StrategyGenome
from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, ToolResult
from evoagent.tools.factory import build_tool


class StrategyExecutor:
    """Execute a strategy genome on the given problem."""

    def __init__(self, problem: OptimizationProblem, weights: np.ndarray | None = None):
        """Initialize.

        Args:
            problem: optimization problem
            weights: scalarization weights (None uses the problem's default uniform weights)
        """
        self.problem = problem
        self.weights = weights

    def run(
        self,
        genome: StrategyGenome,
        budget: int,
        rng: np.random.Generator | None = None,
    ) -> ToolResult:
        """Execute the strategy.

        Args:
            genome: strategy genome
            budget: total evaluation budget
            rng: random number generator

        Returns:
            ToolResult: merged best result and convergence curve across phases
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
        """Merge multi-phase results: take the global best and concatenate the convergence curves."""
        best = max(results, key=lambda r: r.best_fitness)
        history: list[float] = []
        for r in results:
            history.extend(r.history)
        # Keep the concatenated curve monotonic (global best)
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

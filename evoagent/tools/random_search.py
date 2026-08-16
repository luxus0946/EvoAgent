"""Random search baseline tool."""

import numpy as np

from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, OptimizationTool, ToolResult


class RandomSearchTool(OptimizationTool):
    """Random search: global exploration baseline that samples the parameter space uniformly."""

    name = "random_search"

    def optimize(
        self,
        problem: OptimizationProblem,
        budget: int,
        weights: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
        early_stop: EarlyStopMonitor | None = None,
        rng: np.random.Generator | None = None,
    ) -> ToolResult:
        if rng is None:
            rng = np.random.default_rng()
        low, high = problem.bounds[:, 0], problem.bounds[:, 1]
        best_params = (
            x_init.copy() if x_init is not None else rng.uniform(low, high, problem.dim)
        )
        best_fitness = -np.inf
        history: list[float] = []
        n_evals = 0
        n_improvements = 0

        candidates = rng.uniform(low, high, size=(budget, problem.dim))
        if x_init is not None:
            candidates[0] = x_init
        for i in range(budget):
            x = candidates[i]
            fitness = problem.scalarize(x, weights)
            n_evals += 1
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = x.copy()
                n_improvements += 1
            history.append(best_fitness)
            if early_stop is not None and early_stop.check(fitness, best_fitness):
                break
        return ToolResult(
            best_params=best_params,
            best_fitness=best_fitness,
            history=history,
            n_evals=n_evals,
            n_improvements=n_improvements,
        )

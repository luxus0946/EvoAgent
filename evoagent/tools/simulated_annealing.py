"""Simulated annealing tool: escapes local optima by probabilistically accepting worse solutions."""

import numpy as np

from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, OptimizationTool, ToolResult


class SimulatedAnnealingTool(OptimizationTool):
    """Simulated annealing.

    Hyperparameters (evolvable via the strategy genome):
    - sa_t0: Initial temperature (relative to the fitness scale)
    - sa_alpha: Cooling coefficient (multiplicative decay per step)
    - sa_sigma: Neighborhood perturbation step (relative to the parameter range)
    """

    name = "sa"

    def __init__(
        self,
        t0: float = 0.05,
        alpha: float = 0.995,
        sigma: float = 0.1,
    ):
        self.t0 = t0
        self.alpha = alpha
        self.sigma = sigma

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
        span = high - low

        if x_init is not None:
            x_cur = x_init.copy()
            f_cur = problem.scalarize(x_cur, weights)
        else:
            x_cur = rng.uniform(low, high, problem.dim)
            f_cur = problem.scalarize(x_cur, weights)
        x_best, f_best = x_cur.copy(), f_cur

        temp = self.t0 * max(1.0, abs(f_cur))
        history: list[float] = []
        n_evals = 1
        n_improvements = 1
        history.append(f_best)
        for _ in range(1, budget):
            step = rng.normal(0.0, self.sigma, problem.dim) * span
            x_new = problem.validate(x_cur + step)
            f_new = problem.scalarize(x_new, weights)
            n_evals += 1
            delta = f_new - f_cur
            accept = delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-12))
            if accept:
                x_cur, f_cur = x_new, f_new
            if f_new > f_best:
                x_best, f_best = x_new.copy(), f_new
                n_improvements += 1
            temp *= self.alpha
            history.append(f_best)
            if early_stop is not None and early_stop.check(f_cur, f_best):
                break
        return ToolResult(
            best_params=x_best,
            best_fitness=f_best,
            history=history,
            n_evals=n_evals,
            n_improvements=n_improvements,
        )

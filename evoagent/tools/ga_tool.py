"""遗传算法工具：实数编码，锦标赛选择 + 均匀交叉 + 高斯变异，μ+λ 精英策略。"""

import numpy as np

from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, OptimizationTool, ToolResult


class GATool(OptimizationTool):
    """实数编码遗传算法。

    超参（可通过策略基因进化）：
    - mutation_rate: 高斯变异概率
    - crossover_rate: 均匀交叉概率
    - population_size: 种群大小（<=0 时按预算自适应）
    """

    name = "ga"

    def __init__(
        self,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        population_size: int = 0,
        tournament_size: int = 3,
    ):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.tournament_size = tournament_size

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

        pop_size = (
            self.population_size
            if self.population_size > 0
            else max(8, min(20, budget // 5))
        )
        pop_size = min(pop_size, budget // 2)

        pop = rng.uniform(low, high, size=(pop_size, problem.dim))
        if x_init is not None:
            pop[0] = x_init
        fitness = np.array([problem.scalarize(x, weights) for x in pop])
        n_evals = pop_size

        best_idx = int(np.argmax(fitness))
        best_params = pop[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        history = [best_fitness] * pop_size
        n_improvements = 1

        while n_evals + pop_size <= budget:
            offspring = self._reproduce(pop, fitness, pop_size, problem, low, high, span, rng)
            off_fitness = np.array([problem.scalarize(x, weights) for x in offspring])
            n_evals += pop_size

            combined = np.concatenate([pop, offspring], axis=0)
            comb_fit = np.concatenate([fitness, off_fitness])
            keep = np.argsort(-comb_fit)[:pop_size]
            pop = combined[keep]
            fitness = comb_fit[keep]

            if float(fitness[0]) > best_fitness:
                best_fitness = float(fitness[0])
                best_params = pop[0].copy()
                n_improvements += 1
            history.extend([best_fitness] * pop_size)

            if early_stop is not None and early_stop.check(float(fitness[0]), best_fitness):
                break

        return ToolResult(
            best_params=best_params,
            best_fitness=best_fitness,
            history=history,
            n_evals=n_evals,
            n_improvements=n_improvements,
        )

    def _reproduce(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        size: int,
        problem: OptimizationProblem,
        low: np.ndarray,
        high: np.ndarray,
        span: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """锦标赛选择父代 + 均匀交叉 + 高斯变异生成子代。"""
        offspring = np.empty((size, problem.dim))
        for j in range(size):
            p1 = self._tournament(pop, fitness, rng)
            p2 = self._tournament(pop, fitness, rng)
            if rng.random() < self.crossover_rate:
                mask = rng.random(problem.dim) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            if rng.random() < self.mutation_rate:
                child = child + rng.normal(0.0, 0.1, problem.dim) * span
                child = np.clip(child, low, high)
            offspring[j] = child
        return offspring

    def _tournament(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        k = min(self.tournament_size, len(pop))
        candidates = rng.choice(len(pop), k, replace=False)
        return pop[candidates[int(np.argmax(fitness[candidates]))]]

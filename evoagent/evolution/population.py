"""种群管理：初始化、评估、选择、繁殖与多样性统计。"""

import numpy as np

from evoagent.core.individual import AgentIndividual, random_individual
from evoagent.environment.fitness import pareto_rank_and_crowding
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.operators import (
    crossover_uniform,
    mutate_individual,
    tournament_selection,
)
from evoagent.evolution.strategy import StrategyExecutor


class Population:
    """一个岛上的 Agent 种群。"""

    def __init__(
        self,
        problem: OptimizationProblem,
        size: int,
        seed: int,
        mutation_rate: float = 0.15,
        selection_pressure: float = 0.3,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        multi_objective: bool = False,
        eval_budget_per_individual: int = 300,
        fitness_weights: np.ndarray | None = None,
    ):
        """初始化。

        Args:
            problem: 优化问题
            size: 种群大小
            seed: 随机种子
            mutation_rate: 变异率
            selection_pressure: 选择压力（父代保留比例）
            crossover_rate: 交叉率
            elite_ratio: 精英保留比例
            multi_objective: 是否多目标模式（Pareto 选择）
            eval_budget_per_individual: 每个个体执行策略的评估预算
            fitness_weights: 标量化权重（单目标模式），None 时使用均匀权重
        """
        self.problem = problem
        self.size = size
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.multi_objective = multi_objective
        self.eval_budget_per_individual = eval_budget_per_individual
        self.fitness_weights = fitness_weights
        self.rng = np.random.default_rng(seed)
        self.individuals = [
            random_individual(self.rng, problem.n_objectives) for _ in range(size)
        ]
        self.executor = StrategyExecutor(problem, weights=fitness_weights)
        self.generation = 0

    # ------------------------------------------------------------ 评估

    def evaluate_all(self) -> None:
        """评估种群中所有个体的策略（每代调用一次）。"""
        for ind in self.individuals:
            result = self.executor.run(
                ind.genome,
                budget=self._eval_budget(ind),
                rng=self._ind_rng(ind),
            )
            ind.best_params = result.best_params
            ind.n_evals = result.n_evals
            ind.n_improvements = result.n_improvements
            if self.multi_objective:
                ind.objectives = self.problem.objectives_clean(ind.best_params)
                ind.fitness = float(
                    self.problem.scalarize(ind.best_params, ind.genome.weights)
                )
            else:
                ind.objectives = self.problem.objectives_clean(ind.best_params)
                ind.fitness = result.best_fitness

    def _eval_budget(self, ind: AgentIndividual) -> int:
        """单个个体的评估预算。"""
        return self.eval_budget_per_individual

    def _ind_rng(self, ind: AgentIndividual) -> np.random.Generator:
        """每个个体使用种群 RNG 顺序派生的独立种子（保证可复现）。"""
        return np.random.default_rng(int(self.rng.integers(0, 2**31)))

    # ------------------------------------------------------------ 统计

    def best_individual(self) -> AgentIndividual:
        """返回适应度最高的个体。"""
        return max(self.individuals, key=lambda ind: ind.fitness)

    def mean_fitness(self) -> float:
        """种群平均适应度。"""
        return float(np.mean([ind.fitness for ind in self.individuals]))

    def diversity(self) -> float:
        """种群多样性：最优参数两两欧氏距离的均值。"""
        params = np.array(
            [ind.best_params for ind in self.individuals if ind.best_params is not None]
        )
        if len(params) < 2:
            return 0.0
        span = self.problem.bounds[:, 1] - self.problem.bounds[:, 0]
        span[span <= 0] = 1.0
        normalized = params / span
        diff = normalized[:, None, :] - normalized[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        mask = ~np.eye(len(normalized), dtype=bool)
        return float(np.mean(dist[mask]))

    # ------------------------------------------------------------ 进化

    def next_generation(self) -> None:
        """选择父代 -> 交叉/变异生成子代 -> 精英保留，推进一代。"""
        sorted_ind = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        n_elite = max(1, int(self.size * self.elite_ratio))
        n_parents = max(2, int(self.size * self.selection_pressure))
        n_children = self.size - n_elite

        if self.multi_objective:
            parents = self._pareto_tournament(n_parents)
        else:
            parents = tournament_selection(sorted_ind, n_parents, rng=self.rng)

        children: list[AgentIndividual] = []
        while len(children) < n_children:
            p1, p2 = parents[self.rng.integers(len(parents))], parents[
                self.rng.integers(len(parents))
            ]
            if self.rng.random() < self.crossover_rate:
                child = crossover_uniform(p1, p2, rng=self.rng)
            else:
                child = p1.clone()
            child = mutate_individual(child, self.mutation_rate, self.rng)
            children.append(child)

        self.individuals = sorted_ind[:n_elite] + children[:n_children]
        self.generation += 1

    def _pareto_tournament(self, k: int) -> list[AgentIndividual]:
        """多目标模式：按支配等级 + 拥挤度的二进制锦标赛选择父代。"""
        objs = np.array([ind.objectives for ind in self.individuals])
        ranks, crowding = pareto_rank_and_crowding(objs)
        for i, ind in enumerate(self.individuals):
            ind.pareto_rank = int(ranks[i])
            ind.crowding = float(crowding[i])
        selected: list[AgentIndividual] = []
        for _ in range(k):
            a, b = self.rng.choice(len(self.individuals), 2, replace=False)
            ia, ib = self.individuals[a], self.individuals[b]
            better = (
                ia
                if ia.pareto_rank < ib.pareto_rank
                or (ia.pareto_rank == ib.pareto_rank and ia.crowding >= ib.crowding)
                else ib
            )
            selected.append(better)
        return selected

    def sort_by_fitness(self) -> list[AgentIndividual]:
        """按适应度降序返回个体列表。"""
        return sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)

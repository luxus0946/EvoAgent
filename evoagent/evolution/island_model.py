"""多种群岛屿模型：探索 / 平衡 / 利用三种群 + 环状迁移。

- 探索岛：高变异率、低选择压力，保持多样性
- 平衡岛：中等变异率与选择压力
- 利用岛：低变异率、高选择压力，局部精调
"""

from dataclasses import dataclass

import numpy as np

from evoagent.evolution.population import Population

ISLAND_PROFILES: dict[str, dict[str, float]] = {
    "explore": {
        "mutation_rate": 0.30,
        "selection_pressure": 0.20,
        "archive_ratio": 0.4,
    },
    "balance": {
        "mutation_rate": 0.15,
        "selection_pressure": 0.30,
        "archive_ratio": 0.25,
    },
    "exploit": {
        "mutation_rate": 0.05,
        "selection_pressure": 0.40,
        "archive_ratio": 0.1,
    },
}

DEFAULT_ISLAND_NAMES = ["explore", "balance", "exploit"]


@dataclass
class IslandStats:
    """岛屿统计信息。"""

    name: str
    best_fitness: float
    mean_fitness: float
    diversity: float
    total_evals: int


class IslandModel:
    """岛屿模型种群管理。"""

    def __init__(
        self,
        problem,
        island_names: list[str] | None = None,
        population_size: int = 8,
        seed: int = 42,
        migration_interval: int = 3,
        migration_rate: float = 0.2,
        multi_objective: bool = False,
        eval_budget_per_individual: int = 300,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        fitness_weights: np.ndarray | None = None,
    ):
        """初始化。

        Args:
            problem: 优化问题
            island_names: 岛屿名称列表（决定岛屿数量与进化特征）
            population_size: 每个岛屿的种群大小
            seed: 随机种子
            migration_interval: 迁移间隔（代）
            migration_rate: 迁移比例
            multi_objective: 多目标模式
            eval_budget_per_individual: 个体策略评估预算
            crossover_rate: 交叉率
            elite_ratio: 精英比例
            fitness_weights: 标量化权重（单目标模式）
        """
        self.names = island_names or DEFAULT_ISLAND_NAMES
        self.population_size = population_size
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.generation = 0
        self.islands: list[Population] = []
        for i, name in enumerate(self.names):
            profile = ISLAND_PROFILES.get(name, ISLAND_PROFILES["balance"])
            island = Population(
                problem=problem,
                size=population_size,
                seed=seed + i * 1000,
                mutation_rate=profile["mutation_rate"],
                selection_pressure=profile["selection_pressure"],
                crossover_rate=crossover_rate,
                elite_ratio=elite_ratio,
                multi_objective=multi_objective,
                eval_budget_per_individual=eval_budget_per_individual,
                fitness_weights=fitness_weights,
                archive_ratio=profile.get("archive_ratio", 0.0),
            )
            island.name = name
            self.islands.append(island)

    @property
    def n_individuals(self) -> int:
        """全部个体数量。"""
        return len(self.names) * self.population_size

    def evaluate_all(self) -> int:
        """评估所有岛的个体，返回总评估次数。"""
        total = 0
        for island in self.islands:
            island.evaluate_all()
            total += sum(ind.n_evals for ind in island.individuals)
        return total

    def next_generation(self) -> None:
        """周期性环状迁移（基于上一代已评估的适应度），随后各岛内部进化。"""
        if self.generation > 0 and self.generation % self.migration_interval == 0:
            self._migrate()
        for island in self.islands:
            island.next_generation()
        self.generation += 1

    def _migrate(self) -> None:
        """环状迁移：每岛最优的 migration_rate 比例个体替换下一岛最差个体。

        先对各岛排序取快照，再统一执行替换，避免迁移中途影响后续排序。
        """
        n = len(self.islands)
        n_migrate = max(1, int(self.population_size * self.migration_rate))
        migrants = [
            island.sort_by_fitness()[:n_migrate] for island in self.islands
        ]
        worst = [
            island.sort_by_fitness()[-n_migrate:] for island in self.islands
        ]
        for i in range(n):
            target_idx = (i + 1) % n
            target = self.islands[target_idx]
            removed = worst[target_idx]
            target.individuals = [
                ind for ind in target.individuals if all(ind is not w for w in removed)
            ]
            for j in range(n_migrate):
                target.individuals.append(migrants[i][j].clone())

    def best_individual(self):
        """全局最优个体。"""
        bests = [island.best_individual() for island in self.islands]
        return max(bests, key=lambda ind: ind.fitness)

    def stats(self) -> list[IslandStats]:
        """各岛统计。"""
        return [
            IslandStats(
                name=self.names[i],
                best_fitness=island.best_individual().fitness,
                mean_fitness=island.mean_fitness(),
                diversity=island.diversity(),
                total_evals=sum(ind.n_evals for ind in island.individuals),
            )
            for i, island in enumerate(self.islands)
        ]

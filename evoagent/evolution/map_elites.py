"""MAP-Elites 特征档案（借鉴 OpenEvolve database.py 的特征坐标 + 精英网格）。

每个个体按特征坐标落入网格 cell，每格只保留精英（同格内更优者替换），
为父代采样提供"质量 x 多样性"的档案：
- 特征维度 1：策略利用度（工具组合的探索/利用倾向）
- 特征维度 2：最终适应度分位 bin
"""

import numpy as np

from evoagent.core.individual import AgentIndividual

# 工具倾向分数：0=纯探索, 1=纯利用
_TOOL_TENDENCY: dict[str, float] = {
    "random_search": 0.0,
    "sa": 0.25,
    "ga": 0.5,
    "bo": 0.75,
    "cma_es": 1.0,
}

DEFAULT_BINS = (5, 5)


def feature_coords(
    individual: AgentIndividual,
    bins: tuple[int, int] = DEFAULT_BINS,
) -> tuple[int, int]:
    """计算个体的 MAP-Elites 特征坐标。

    Args:
        individual: 已评估的个体
        bins: 两个维度的分箱数

    Returns:
        (利用度 bin, 适应度 bin)
    """
    genome = individual.genome
    tendency = (
        _TOOL_TENDENCY.get(genome.initial_tool, 0.5)
        + _TOOL_TENDENCY.get(genome.second_tool, 0.5)
    ) / 2.0
    tendency_bin = min(bins[0] - 1, int(tendency * bins[0]))
    fitness_bin = min(bins[1] - 1, int(float(individual.fitness) * bins[1]))
    return (tendency_bin, fitness_bin)


class MapElitesArchive:
    """MAP-Elites 精英档案：按特征网格保留每格最优个体。"""

    def __init__(self, bins: tuple[int, int] = DEFAULT_BINS):
        """初始化。

        Args:
            bins: 特征网格各维度分箱数
        """
        self.bins = bins
        self.grid: dict[tuple[int, int], AgentIndividual] = {}

    def add(self, individual: AgentIndividual) -> None:
        """尝试将个体放入网格（仅当该格为空或个体更优）。"""
        if individual.fitness is None:
            return
        coords = feature_coords(individual, self.bins)
        current = self.grid.get(coords)
        if current is None or individual.fitness > current.fitness:
            self.grid[coords] = individual.clone()

    def add_many(self, individuals: list[AgentIndividual]) -> None:
        """批量加入。"""
        for ind in individuals:
            self.add(ind)

    def sample_elite(self, rng: np.random.Generator) -> AgentIndividual:
        """随机抽取一个非空格的精英个体。"""
        if not self.grid:
            raise ValueError("档案为空")
        coords = rng.choice(list(self.grid.keys()))
        return self.grid[tuple(coords)]

    def size(self) -> int:
        """当前非空格数。"""
        return len(self.grid)

    def best(self) -> AgentIndividual:
        """档案中的全局最优。"""
        if not self.grid:
            raise ValueError("档案为空")
        return max(self.grid.values(), key=lambda ind: ind.fitness)
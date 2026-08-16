"""MAP-Elites feature archive (inspired by the feature coordinates and elite grid of OpenEvolve database.py).

Each individual falls into a grid cell by its feature coordinates; each cell keeps only its elite (a better individual in the same cell replaces the current one),
providing a "quality x diversity" archive for parent sampling:
- feature dimension 1: strategy exploitation level (exploration/exploitation tendency of the tool combination)
- feature dimension 2: final fitness quantile bin
"""

import numpy as np

from evoagent.core.individual import AgentIndividual

# Tool tendency score: 0 = pure exploration, 1 = pure exploitation
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
    """Compute the MAP-Elites feature coordinates of an individual.

    Args:
        individual: evaluated individual
        bins: number of bins per dimension

    Returns:
        (exploitation bin, fitness bin)
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
    """MAP-Elites elite archive: keeps the best individual per cell of the feature grid."""

    def __init__(self, bins: tuple[int, int] = DEFAULT_BINS):
        """Initialize.

        Args:
            bins: number of bins per feature grid dimension
        """
        self.bins = bins
        self.grid: dict[tuple[int, int], AgentIndividual] = {}

    def add(self, individual: AgentIndividual) -> None:
        """Try to insert an individual into the grid (only if the cell is empty or the individual is better)."""
        if individual.fitness is None:
            return
        coords = feature_coords(individual, self.bins)
        current = self.grid.get(coords)
        if current is None or individual.fitness > current.fitness:
            self.grid[coords] = individual.clone()

    def add_many(self, individuals: list[AgentIndividual]) -> None:
        """Insert multiple individuals."""
        for ind in individuals:
            self.add(ind)

    def sample_elite(self, rng: np.random.Generator) -> AgentIndividual:
        """Sample an elite individual from a random non-empty cell."""
        if not self.grid:
            raise ValueError("档案为空")
        coords = rng.choice(list(self.grid.keys()))
        return self.grid[tuple(coords)]

    def size(self) -> int:
        """Number of currently non-empty cells."""
        return len(self.grid)

    def best(self) -> AgentIndividual:
        """Global best in the archive."""
        if not self.grid:
            raise ValueError("档案为空")
        return max(self.grid.values(), key=lambda ind: ind.fitness)
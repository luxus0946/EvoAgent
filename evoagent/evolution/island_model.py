"""Multi-population island model: explore / balance / exploit populations + ring migration.

- explore island: high mutation rate, low selection pressure, maintains diversity
- balance island: moderate mutation rate and selection pressure
- exploit island: low mutation rate, high selection pressure, local fine-tuning
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
    """Island statistics."""

    name: str
    best_fitness: float
    mean_fitness: float
    diversity: float
    total_evals: int


class IslandModel:
    """Island-model population management."""

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
        """Initialize.

        Args:
            problem: optimization problem
            island_names: island name list (determines the number of islands and their evolution characteristics)
            population_size: population size per island
            seed: random seed
            migration_interval: migration interval (generations)
            migration_rate: migration ratio
            multi_objective: multi-objective mode
            eval_budget_per_individual: individual strategy evaluation budget
            crossover_rate: crossover rate
            elite_ratio: elite ratio
            fitness_weights: scalarization weights (single-objective mode)
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
        """Total number of individuals."""
        return len(self.names) * self.population_size

    def evaluate_all(self) -> int:
        """Evaluate individuals on all islands and return the total number of evaluations."""
        total = 0
        for island in self.islands:
            island.evaluate_all()
            total += sum(ind.n_evals for ind in island.individuals)
        return total

    def next_generation(self) -> None:
        """Periodic ring migration (based on fitness evaluated in the previous generation), then evolve each island internally."""
        if self.generation > 0 and self.generation % self.migration_interval == 0:
            self._migrate()
        for island in self.islands:
            island.next_generation()
        self.generation += 1

    def _migrate(self) -> None:
        """Ring migration: each island's best migration_rate-fraction of individuals replaces the next island's worst.

        Sort each island and take a snapshot first, then apply all replacements at once, so sorting during migration is not affected.
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
        """Global best individual."""
        bests = [island.best_individual() for island in self.islands]
        return max(bests, key=lambda ind: ind.fitness)

    def stats(self) -> list[IslandStats]:
        """Per-island statistics."""
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

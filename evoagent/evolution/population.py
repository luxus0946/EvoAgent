"""Population management: initialization, evaluation, selection, breeding, and diversity statistics."""

import numpy as np

from evoagent.core.individual import AgentIndividual, random_individual
from evoagent.environment.fitness import pareto_rank_and_crowding
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.operators import (
    crossover_uniform,
    mutate_individual,
    mutate_individual_eoh,
    tournament_selection,
)
from evoagent.evolution.strategy import StrategyExecutor


class Population:
    """Agent population on an island."""

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
        archive_ratio: float = 0.0,
        mutation_style: str = "eoh",
    ):
        """Initialize.

        Args:
            problem: optimization problem
            size: population size
            seed: random seed
            mutation_rate: mutation rate
            selection_pressure: selection pressure (parent retention ratio)
            crossover_rate: crossover rate
            elite_ratio: elite retention ratio
            multi_objective: whether to use multi-objective mode (Pareto selection)
            eval_budget_per_individual: evaluation budget for each individual's strategy execution
            fitness_weights: scalarization weights (single-objective mode), None uses uniform weights
            archive_ratio: MAP-Elites archive sampling ratio (three-way parent sampling, 0 disables)
            mutation_style: mutation operator style: eoh (EoH operator-based) / uniform (random field edits)
        """
        self.problem = problem
        self.size = size
        self.mutation_rate = mutation_rate
        self.mutation_style = mutation_style
        self.selection_pressure = selection_pressure
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.multi_objective = multi_objective
        self.eval_budget_per_individual = eval_budget_per_individual
        self.fitness_weights = fitness_weights
        self.archive_ratio = archive_ratio
        self.rng = np.random.default_rng(seed)
        self.individuals = [
            random_individual(self.rng, problem.n_objectives) for _ in range(size)
        ]
        self.executor = StrategyExecutor(problem, weights=fitness_weights)
        self.generation = 0
        self.map_elites = None
        if archive_ratio > 0:
            from evoagent.evolution.map_elites import MapElitesArchive

            self.map_elites = MapElitesArchive()

    # ------------------------------------------------------------ evaluation

    def evaluate_all(self) -> None:
        """Evaluate the strategy of all individuals in the population (called once per generation)."""
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
        if self.map_elites is not None:
            self.map_elites.add_many(self.individuals)

    def _eval_budget(self, ind: AgentIndividual) -> int:
        """Evaluation budget for a single individual."""
        return self.eval_budget_per_individual

    def _ind_rng(self, ind: AgentIndividual) -> np.random.Generator:
        """Derive an independent seed per individual sequentially from the population RNG (ensures reproducibility)."""
        return np.random.default_rng(int(self.rng.integers(0, 2**31)))

    # ------------------------------------------------------------ statistics

    def best_individual(self) -> AgentIndividual:
        """Return the individual with the highest fitness."""
        return max(self.individuals, key=lambda ind: ind.fitness)

    def mean_fitness(self) -> float:
        """Mean population fitness."""
        return float(np.mean([ind.fitness for ind in self.individuals]))

    def diversity(self) -> float:
        """Population diversity: mean pairwise Euclidean distance of best parameters."""
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

    # ------------------------------------------------------------ evolution

    def next_generation(self) -> None:
        """Select parents, generate offspring via crossover/mutation, retain elites, and advance one generation."""
        sorted_ind = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        n_elite = max(1, int(self.size * self.elite_ratio))
        n_parents = max(2, int(self.size * self.selection_pressure))
        n_children = self.size - n_elite

        if self.multi_objective:
            parents = self._pareto_tournament(n_parents)
        else:
            parents = tournament_selection(sorted_ind, n_parents, rng=self.rng)
        parents = self._three_way_mix(parents)

        children: list[AgentIndividual] = []
        while len(children) < n_children:
            p1, p2 = parents[self.rng.integers(len(parents))], parents[
                self.rng.integers(len(parents))
            ]
            if self.rng.random() < self.crossover_rate:
                child = crossover_uniform(p1, p2, rng=self.rng)
            else:
                child = p1.clone()
            if self.mutation_style == "eoh":
                child = mutate_individual_eoh(child, self.mutation_rate, self.rng)
            else:
                child = mutate_individual(child, self.mutation_rate, self.rng)
            children.append(child)

        self.individuals = sorted_ind[:n_elite] + children[:n_children]
        self.generation += 1

    def _three_way_mix(
        self, base: list[AgentIndividual]
    ) -> list[AgentIndividual]:
        """Three-way parent sampling (inspired by OpenEvolve database.py):
        tournament parents (exploitation) + MAP-Elites archive elites (exploration) + random individuals (diversity).
        """
        if (
            self.archive_ratio <= 0
            or self.map_elites is None
            or self.map_elites.size() == 0
        ):
            return base
        n = len(base)
        n_archive = int(n * self.archive_ratio)
        n_random = max(1, int(n * 0.15))
        n_archive = min(n_archive, max(0, n - n_random))
        if n_archive <= 0:
            return base
        archive_part = [
            self.map_elites.sample_elite(self.rng) for _ in range(n_archive)
        ]
        random_part = [
            self.individuals[int(self.rng.integers(self.size))]
            for _ in range(n_random)
        ]
        return base[: n - n_archive - n_random] + archive_part + random_part

    def _pareto_tournament(self, k: int) -> list[AgentIndividual]:
        """Multi-objective mode: binary tournament selection of parents by dominance rank and crowding distance."""
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
        """Return individuals sorted by fitness in descending order."""
        return sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)

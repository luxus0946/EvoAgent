"""NSGA-II multi-objective optimization baseline: non-dominated sorting + crowding distance + binary tournament selection."""

import numpy as np

from evoagent.environment.fitness import non_dominated_front, pareto_rank_and_crowding
from evoagent.environment.problem import OptimizationProblem


class NSGA2Result:
    """NSGA-II run result."""

    def __init__(
        self,
        front: np.ndarray,
        archive: np.ndarray,
        history: list[float],
        n_evals: int,
        archive_history: list[np.ndarray] | None = None,
        archive_x: np.ndarray | None = None,
    ):
        self.front = front
        self.archive = archive
        self.history = history
        self.n_evals = n_evals
        self.archive_history = archive_history or []
        self.archive_x = archive_x


class NSGA2Tool:
    """Real-coded NSGA-II (multi-objective, maximization convention)."""

    name = "nsga2"

    def __init__(
        self,
        population_size: int = 40,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.9,
        tournament_size: int = 2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

    def optimize(
        self,
        problem: OptimizationProblem,
        budget: int,
        rng: np.random.Generator | None = None,
    ) -> NSGA2Result:
        """Run multi-objective optimization within the budget.

        Args:
            problem: Optimization problem
            budget: Maximum number of evaluations
            rng: Random number generator

        Returns:
            NSGA2Result: final non-dominated front, global archive, hypervolume history, evaluation count
        """
        if rng is None:
            rng = np.random.default_rng()
        low, high = problem.bounds[:, 0], problem.bounds[:, 1]
        pop_size = min(self.population_size, budget // 2)
        n_obj = problem.n_objectives

        pop = rng.uniform(low, high, size=(pop_size, problem.dim))
        objs = np.array([problem.objectives(x) for x in pop])
        n_evals = pop_size
        archive_x = pop.copy()
        archive = objs.copy()
        archive_x, archive = self._update_archive(archive_x, archive, pop, objs)
        archive_history: list[np.ndarray] = [objs[non_dominated_front(objs)]]

        history: list[float] = []
        while n_evals + pop_size <= budget:
            ranks, crowding = pareto_rank_and_crowding(objs)
            parents = self._binary_tournament(pop, ranks, crowding, pop_size, rng)
            offspring = self._reproduce(parents, problem, low, high, rng)
            off_objs = np.array([problem.objectives(x) for x in offspring])
            n_evals += pop_size

            combined_pop = np.concatenate([pop, offspring], axis=0)
            combined_objs = np.concatenate([objs, off_objs], axis=0)
            ranks, crowding = pareto_rank_and_crowding(combined_objs)
            keep = self._environmental_selection(ranks, crowding, pop_size)
            pop = combined_pop[keep]
            objs = combined_objs[keep]
            archive_x, archive = self._update_archive(archive_x, archive, pop, objs)
            archive_history.append(objs[non_dominated_front(objs)])

            if hasattr(problem, "hv_ref"):
                history.append(self._hypervolume(objs, problem.hv_ref))
            else:
                history.append(len(non_dominated_front(objs)))

        front = objs[non_dominated_front(objs)]
        return NSGA2Result(
            front=front,
            archive=archive,
            history=history,
            n_evals=n_evals,
            archive_history=archive_history,
            archive_x=archive_x,
        )

    def _binary_tournament(
        self,
        pop: np.ndarray,
        ranks: np.ndarray,
        crowding: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        idx = np.empty(k, dtype=int)
        for i in range(k):
            a, b = rng.choice(len(pop), 2, replace=False)
            if ranks[a] < ranks[b] or (
                ranks[a] == ranks[b] and crowding[a] >= crowding[b]
            ):
                idx[i] = a
            else:
                idx[i] = b
        return pop[idx]

    def _reproduce(
        self,
        parents: np.ndarray,
        problem: OptimizationProblem,
        low: np.ndarray,
        high: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        size, d = parents.shape
        offspring = np.empty_like(parents)
        for j in range(0, size, 2):
            if j + 1 >= size:
                offspring[j] = parents[j].copy()
                continue
            p1, p2 = parents[j], parents[j + 1]
            if rng.random() < self.crossover_rate:
                mask = rng.random(d) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if rng.random() < self.mutation_rate:
                c1 = np.clip(c1 + rng.normal(0.0, 0.1, d) * (high - low), low, high)
            if rng.random() < self.mutation_rate:
                c2 = np.clip(c2 + rng.normal(0.0, 0.1, d) * (high - low), low, high)
            offspring[j], offspring[j + 1] = c1, c2
        return offspring

    def _environmental_selection(
        self, ranks: np.ndarray, crowding: np.ndarray, k: int
    ) -> np.ndarray:
        """Keep the top k individuals by dominance rank and crowding distance."""
        order = np.lexsort((-crowding, ranks))
        return order[:k]

    @staticmethod
    def _update_archive(
        archive_x: np.ndarray,
        archive_obj: np.ndarray,
        points_x: np.ndarray,
        points_obj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge new individuals into the non-dominated archive (maintaining both parameters and objectives)."""
        merged_x = np.vstack([archive_x, points_x])
        merged_obj = np.vstack([archive_obj, points_obj])
        keep = non_dominated_front(merged_obj)
        return merged_x[keep], merged_obj[keep]

    @staticmethod
    def _hypervolume(objs: np.ndarray, ref: np.ndarray) -> float:
        from evoagent.environment.fitness import hypervolume_2d

        return hypervolume_2d(objs, ref)

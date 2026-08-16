"""Meta layer: automatic hyperparameter configuration of the evolution algorithm (AAC / Meta-Optimization).

Design:
- the hyperparameter space (tunable knobs of the evolution framework) is encoded as a continuous/discrete vector
- outer loop: Bayesian optimization (reusing BayesianOptTool) searches the hyperparameter space
- inner loop: each candidate evaluation = one short evolution experiment with those hyperparameters, returning noise-free fitness
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from evoagent.config import EvolutionConfig
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.evolutionary_loop import run_evolution
from evoagent.tools.bayesian_opt import BayesianOptTool

logger = logging.getLogger("evoagent.meta")


@dataclass
class HyperparamSpec:
    """Definition of a single hyperparameter."""

    name: str
    low: float
    high: float
    is_int: bool = False


@dataclass
class MetaSearchSpace:
    """Hyperparameter search space of the evolution framework (defaults cover key knobs)."""

    specs: list[HyperparamSpec] = field(
        default_factory=lambda: [
            HyperparamSpec("population_size", 4, 16, is_int=True),
            HyperparamSpec("mutation_rate", 0.05, 0.4),
            HyperparamSpec("crossover_rate", 0.5, 0.95),
            HyperparamSpec("selection_pressure", 0.15, 0.5),
            HyperparamSpec("elite_ratio", 0.05, 0.25),
            HyperparamSpec("migration_interval", 2, 8, is_int=True),
            HyperparamSpec("migration_rate", 0.1, 0.4),
            HyperparamSpec("eval_budget_per_individual", 150, 450, is_int=True),
        ]
    )

    @property
    def names(self) -> list[str]:
        """List of hyperparameter names."""
        return [s.name for s in self.specs]

    @property
    def bounds(self) -> np.ndarray:
        """Hyperparameter bounds matrix (shape (n, 2))."""
        return np.array([[s.low, s.high] for s in self.specs])

    def to_config(self, x: np.ndarray) -> dict:
        """Map a hyperparameter vector to an EvolutionConfig keyword-argument dict (ints rounded, values clipped)."""
        values = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        return {
            spec.name: (
                int(round(float(values[i]))) if spec.is_int else float(values[i])
            )
            for i, spec in enumerate(self.specs)
        }


@dataclass
class MetaEvaluationConfig:
    """Inner evolution experiment config (run budget per candidate hyperparameter evaluation)."""

    problem: OptimizationProblem
    fitness_weights: np.ndarray | None = None
    n_islands: int = 3
    inner_generations: int = 4
    inner_seed: int = 7
    multi_objective: bool = False


class MetaProblem(OptimizationProblem):
    """Wrap "hyperparameter config -> inner evolution performance" as a black-box problem optimizable by BO.

    Each scalarize call runs one inner evolution experiment (deterministic: the inner seed is fixed).
    """

    name = "meta_hyperparams"

    def __init__(
        self,
        space: MetaSearchSpace,
        eval_config: MetaEvaluationConfig,
    ):
        """Initialize.

        Args:
            space: hyperparameter search space
            eval_config: inner evaluation config
        """
        self.space = space
        self.eval_config = eval_config
        self.dim = len(space.specs)
        self.bounds = space.bounds
        self.objective_names = ["inner_fitness"]
        self.minimize = np.array([False])
        self.calls = 0

    def scalarize(self, x: np.ndarray, weights: np.ndarray | None = None) -> float:
        """Evaluate one candidate hyperparameter config: run an inner evolution experiment and return noise-free fitness."""
        config = self.space.to_config(x)
        inner = EvolutionConfig(
            population_size=config["population_size"],
            max_generations=self.eval_config.inner_generations,
            n_islands=self.eval_config.n_islands,
            mutation_rate=config["mutation_rate"],
            crossover_rate=config["crossover_rate"],
            selection_pressure=config["selection_pressure"],
            elite_ratio=config["elite_ratio"],
            migration_interval=config["migration_interval"],
            migration_rate=config["migration_rate"],
            eval_budget_per_individual=config["eval_budget_per_individual"],
            multi_objective=self.eval_config.multi_objective,
            n_objectives=self.eval_config.problem.n_objectives,
            fitness_weights=self.eval_config.fitness_weights,
            random_seed=self.eval_config.inner_seed,
        )
        self.calls += 1
        result = run_evolution(self.eval_config.problem, inner)
        clean = self.eval_config.problem.scalarize_clean(
            result.best_params, self.eval_config.fitness_weights
        )
        logger.debug(
            "内层评估 #%d: %s -> %.6f",
            self.calls,
            config,
            clean,
        )
        return float(clean)

    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        """Placeholder implementation (inner evaluation goes through scalarize)."""
        return np.array([self.scalarize(x)])


class HyperparameterSearch:
    """Meta-layer hyperparameter search: Bayesian optimization outer loop + inner evolution evaluation."""

    def __init__(
        self,
        space: MetaSearchSpace,
        eval_config: MetaEvaluationConfig,
        n_init: int = 4,
        n_iterations: int = 12,
        seed: int = 42,
        weights: np.ndarray | None = None,
    ):
        """Initialize.

        Args:
            space: hyperparameter search space
            eval_config: inner evaluation config
            n_init: number of initial random exploration points
            n_iterations: number of BO iterations
            seed: outer random seed
            weights: scalarization weights of the inner problem
        """
        self.space = space
        self.eval_config = eval_config
        self.n_init = n_init
        self.n_iterations = n_iterations
        self.seed = seed
        self.weights = weights
        self.problem = MetaProblem(space, eval_config)
        self.rng = np.random.default_rng(seed)
        self.trajectory: list[tuple[np.ndarray, float]] = []

    def search(self) -> dict:
        """Run the hyperparameter search.

        Returns:
            best hyperparameter config (EvolutionConfig keyword dict)
        """
        low = self.problem.bounds[:, 0]
        high = self.problem.bounds[:, 1]
        x_hist: list[np.ndarray] = []
        f_hist: list[float] = []

        for _ in range(self.n_init):
            x = self.rng.uniform(low, high)
            f_hist.append(self.problem.scalarize(x))
            x_hist.append(x)
            self.trajectory.append((x.copy(), f_hist[-1]))

        bo = BayesianOptTool()
        x_arr = np.array(x_hist)
        f_arr = np.array(f_hist)
        for _ in range(self.n_iterations):
            gp = bo._fit_gp(x_arr, f_arr, self.rng)
            x_next = bo._maximize_ei(gp, self.problem, low, high, self.rng)
            f_next = self.problem.scalarize(x_next)
            x_arr = np.vstack([x_arr, x_next])
            f_arr = np.append(f_arr, f_next)
            self.trajectory.append((x_next.copy(), float(f_next)))
            logger.info(
                "BO 迭代: %s -> %.6f", self.space.to_config(x_next), f_next
            )

        best_idx = int(np.argmax(f_arr))
        best_x = x_arr[best_idx]
        logger.info(
            "Meta 搜索完成: 最优 %.6f @ %s",
            float(f_arr[best_idx]),
            self.space.to_config(best_x),
        )
        return self.space.to_config(best_x)

    def best_fitness(self) -> float:
        """Current best evaluated value."""
        return max(f for _, f in self.trajectory) if self.trajectory else 0.0
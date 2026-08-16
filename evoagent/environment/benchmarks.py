"""Standard benchmark functions: a classic test set for verifying algorithm generalization.

Includes: Rosenbrock (convex, valley-shaped), Ackley (multimodal, many local optima), Rastrigin (strongly multimodal),
ZDT1 (two-objective Pareto test problem). All are minimization problems.
"""

import numpy as np

from evoagent.environment.problem import OptimizationProblem


class _SingleObjectiveProblem(OptimizationProblem):
    """Common base class for single-objective minimization benchmarks."""

    objective_names = ["f"]
    minimize = np.array([True])
    noise_std = 0.0

    def _eval(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._eval(np.asarray(x, dtype=float))])


class RosenbrockProblem(_SingleObjectiveProblem):
    """Rosenbrock function: classic convex test (optimum x_i=1, f=0)."""

    name = "rosenbrock"
    dim = 10
    bounds = np.tile([-5.0, 10.0], (dim, 1))

    def _eval(self, x: np.ndarray) -> float:
        return float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
        )


class AckleyProblem(_SingleObjectiveProblem):
    """Ackley function: multimodal with many local optima (optimum x=0, f=0)."""

    name = "ackley"
    dim = 10
    bounds = np.tile([-32.0, 32.0], (dim, 1))

    def _eval(self, x: np.ndarray) -> float:
        d = len(x)
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x)))
        return float(term1 + term2 + 20.0 + np.e)


class RastriginProblem(_SingleObjectiveProblem):
    """Rastrigin function: strongly multimodal (optimum x=0, f=0)."""

    name = "rastrigin"
    dim = 10
    bounds = np.tile([-5.12, 5.12], (dim, 1))

    def _eval(self, x: np.ndarray) -> float:
        d = len(x)
        return float(10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


class ZDT1Problem(OptimizationProblem):
    """ZDT1 two-objective test problem (optimal Pareto front f2 = 1 - sqrt(f1))."""

    name = "zdt1"
    dim = 10
    bounds = np.tile([0.0, 1.0], (dim, 1))
    objective_names = ["f1", "f2"]
    minimize = np.array([True, True])
    noise_std = 0.0

    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = 1.0 + 9.0 * np.mean(x[1:])
        f1 = x[0]
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])


BENCHMARK_REGISTRY: dict[str, type[OptimizationProblem]] = {
    "rosenbrock": RosenbrockProblem,
    "ackley": AckleyProblem,
    "rastrigin": RastriginProblem,
    "zdt1": ZDT1Problem,
}

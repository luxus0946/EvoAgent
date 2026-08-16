"""Unified protocol for optimization problems: all test problems (semiconductor surrogate simulation + standard benchmarks) implement this interface.

Conventions:
- raw objective values are returned by ``evaluate`` / ``evaluate_clean``, with each objective's direction declared by ``minimize``;
- ``objectives`` returns objective vectors under the "maximization convention" (minimization objectives negated), for multi-objective sorting and metric computation;
- ``scalarize`` returns a weighted scalar fitness (higher is better).
"""

from abc import ABC, abstractmethod

import numpy as np


class OptimizationProblem(ABC):
    """Abstract base class for optimization problems."""

    name: str = "problem"
    dim: int = 0
    bounds: np.ndarray = np.zeros((1, 2))
    objective_names: list[str] = []
    minimize: np.ndarray = np.array([])
    noise_std: float = 0.0

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objective_names)

    def validate(self, x: np.ndarray) -> np.ndarray:
        """Clip parameters to the bounds.

        Args:
            x: parameter vector, shape (d,) or (n, d)

        Returns:
            clipped parameters
        """
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def scalarize(self, x: np.ndarray, weights: np.ndarray | None = None) -> float:
        """Weighted scalar fitness (maximization convention); higher is better.

        Args:
            x: parameter vector
            weights: objective weights (sum to 1), None uses uniform weights

        Returns:
            scalar fitness
        """
        obj = self.objectives(x)
        if weights is None:
            weights = np.ones(obj.shape[-1]) / obj.shape[-1]
        return float(np.sum(np.asarray(weights) * obj))

    def scalarize_clean(self, x: np.ndarray, weights: np.ndarray | None = None) -> float:
        """Noise-free weighted scalar fitness (for final metric reporting)."""
        obj = self.objectives_clean(x)
        if weights is None:
            weights = np.ones(obj.shape[-1]) / obj.shape[-1]
        return float(np.sum(np.asarray(weights) * obj))

    @abstractmethod
    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        """Noise-free raw objective evaluation.

        Args:
            x: parameter vector, shape (d,)

        Returns:
            raw objective vector, shape (n_objectives,)
        """
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Noisy raw objective evaluation (simulates real measurement error).

        Args:
            x: parameter vector, shape (d,)

        Returns:
            raw objective vector, shape (n_objectives,)
        """
        return self.evaluate_clean(x)

    def objectives(self, x: np.ndarray) -> np.ndarray:
        """Objective vector under the maximization convention: minimization objectives are negated.

        Args:
            x: parameter vector

        Returns:
            objective vector under the maximization convention
        """
        raw = self.evaluate(x)
        sign = np.where(self.minimize, -1.0, 1.0)
        return raw * sign

    def objectives_clean(self, x: np.ndarray) -> np.ndarray:
        """Noise-free objective vector under the maximization convention."""
        raw = self.evaluate_clean(x)
        sign = np.where(self.minimize, -1.0, 1.0)
        return raw * sign

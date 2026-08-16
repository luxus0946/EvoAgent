"""Abstract base classes and shared data structures for optimization tools.

All tools share a uniform signature: maximize the weighted scalar fitness within a given evaluation budget.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from evoagent.environment.problem import OptimizationProblem

# Tool registry: name set shared by strategy genomes and the tool pool
TOOL_NAMES = ["random_search", "sa", "ga", "cma_es", "bo", "ppo"]


@dataclass
class ToolResult:
    """Tool execution result."""

    best_params: np.ndarray
    best_fitness: float
    history: list[float] = field(default_factory=list)
    n_evals: int = 0
    n_improvements: int = 0

    @property
    def last_improved_eval(self) -> int:
        """Evaluation index (1-based) of the last improvement; 0 if never improved."""
        return self.n_improvements


class EarlyStopMonitor:
    """Early-stop monitor: notifies the tool to terminate early once consecutive non-improvement reaches the threshold."""

    def __init__(self, patience_evals: int):
        """Initialize.

        Args:
            patience_evals: Number of consecutive non-improving evaluations tolerated; <= 0 disables early stopping
        """
        self.patience = patience_evals
        self._streak = 0

    @property
    def enabled(self) -> bool:
        """Whether early stopping is enabled."""
        return self.patience > 0

    def check(self, current: float, best: float) -> bool:
        """Check whether termination is required after one iteration.

        Args:
            current: Current fitness
            best: Historical best fitness

        Returns:
            True if the run should terminate
        """
        if not self.enabled:
            return False
        if current > best + 1e-12:
            self._streak = 0
        else:
            self._streak += 1
        return self._streak >= self.patience


class OptimizationTool(ABC):
    """Abstract base class for optimization tools; all tools must inherit from it."""

    name: str = "base_tool"

    @abstractmethod
    def optimize(
        self,
        problem: OptimizationProblem,
        budget: int,
        weights: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
        early_stop: EarlyStopMonitor | None = None,
        rng: np.random.Generator | None = None,
    ) -> ToolResult:
        """Run optimization within the budget (maximize the weighted scalar fitness).

        Args:
            problem: Optimization problem
            budget: Maximum number of evaluations
            weights: Objective weights; uniform weights are used when None
            x_init: Initial parameters (historical best); the algorithm initializes itself when None
            early_stop: Early-stop monitor; disabled when None
            rng: Random number generator

        Returns:
            ToolResult: best parameters, best fitness, convergence history, evaluation count
        """
        raise NotImplementedError

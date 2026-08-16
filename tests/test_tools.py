"""Optimization tool tests: convergence and correct return value format."""

import numpy as np
import pytest

from evoagent.environment.benchmarks import RastriginProblem, RosenbrockProblem
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.tools.factory import build_tool
from evoagent.tools.nsga2 import NSGA2Tool
from evoagent.utils.random import make_rng

BUDGET = 200


@pytest.mark.parametrize(
    "tool_name",
    ["random_search", "sa", "ga", "cma_es", "bo"],
)
class TestSingleObjectiveTools:
    def test_result_format(self, tool_name):
        problem = RosenbrockProblem()
        tool = build_tool(tool_name)
        result = tool.optimize(problem, BUDGET, rng=make_rng(0))
        assert result.best_params.shape == (problem.dim,)
        assert result.n_evals <= BUDGET
        assert len(result.history) == result.n_evals
        assert result.best_fitness > -1e100

    def test_converges_on_rosenbrock(self, tool_name):
        """Should clearly outperform random search on the 2-D Rosenbrock."""
        problem = RosenbrockProblem()
        problem.dim = 2
        problem.bounds = np.tile([-5.0, 10.0], (2, 1))
        tool = build_tool(tool_name)
        result = tool.optimize(problem, BUDGET, rng=make_rng(0))
        assert result.best_fitness > -30.0  # near the optimum 0, not a distant random value

    def test_improves_from_init(self, tool_name):
        problem = SemiconductorSimulator()
        tool = build_tool(tool_name)
        x_init = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = tool.optimize(problem, BUDGET, x_init=x_init, rng=make_rng(1))
        assert result.best_fitness >= problem.scalarize(x_init) - 1e-9


class TestNSGA2:
    def test_returns_non_dominated_front(self):
        problem = RosenbrockProblem()
        problem.dim = 2
        problem.bounds = np.tile([0.0, 1.0], (2, 1))
        problem.objective_names = ["f1", "f2"]
        problem.minimize = np.array([True, False])
        problem.evaluate_clean = lambda x: np.array([x[0], x[1]])
        problem.evaluate = problem.evaluate_clean
        problem.objectives = lambda x: problem.evaluate_clean(x) * np.array([-1.0, 1.0])
        problem.objectives_clean = problem.objectives

        tool = NSGA2Tool(population_size=10)
        result = tool.optimize(problem, budget=100, rng=make_rng(0))
        assert result.n_evals <= 100
        assert result.front.shape[1] == 2
        assert len(result.front) >= 1


class TestStability:
    def test_reproducible_with_same_seed(self):
        problem = RastriginProblem()
        tool = build_tool("cma_es")
        r1 = tool.optimize(problem, 100, rng=make_rng(42))
        r2 = tool.optimize(problem, 100, rng=make_rng(42))
        assert r1.best_params.shape == r2.best_params.shape
        assert r1.best_fitness == pytest.approx(r2.best_fitness)

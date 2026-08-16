"""Meta 层超参搜索测试。"""

import numpy as np
import pytest

from evoagent.environment.benchmarks import RastriginProblem
from evoagent.meta.hyperparameter_search import (
    HyperparameterSearch,
    MetaEvaluationConfig,
    MetaProblem,
    MetaSearchSpace,
)


class TestMetaSearchSpace:
    def test_to_config_roundtrip_int(self):
        space = MetaSearchSpace()
        x = np.array([8.0, 0.15, 0.8, 0.3, 0.1, 3.0, 0.2, 300.0])
        cfg = space.to_config(x)
        assert cfg["population_size"] == 8
        assert cfg["migration_interval"] == 3
        assert cfg["eval_budget_per_individual"] == 300
        assert cfg["mutation_rate"] == pytest.approx(0.15)

    def test_to_config_clips_bounds(self):
        space = MetaSearchSpace()
        cfg = space.to_config(np.array([999.0, 0.0, 0.0, 0.0, 0.0, 99.0, 0.0, 0.0]))
        assert cfg["population_size"] == 16
        assert cfg["mutation_rate"] == pytest.approx(0.05)
        assert cfg["migration_interval"] == 8

    def test_names_and_bounds_aligned(self):
        space = MetaSearchSpace()
        assert len(space.names) == len(space.bounds) == space.bounds.shape[0]

    def test_int_rounding(self):
        space = MetaSearchSpace()
        cfg = space.to_config(np.array([4.6, 0.15, 0.8, 0.3, 0.1, 3.4, 0.2, 300.0]))
        assert cfg["population_size"] == 5


class TestMetaProblem:
    def _problem(self):
        return MetaProblem(
            MetaSearchSpace(),
            MetaEvaluationConfig(
                problem=RastriginProblem(),
                n_islands=1,
                inner_generations=2,
            ),
        )

    def test_bounds_match_space(self):
        p = self._problem()
        np.testing.assert_allclose(p.bounds, p.space.bounds)
        assert p.dim == len(p.space.specs)

    def test_scalarize_runs_inner_evolution(self):
        p = self._problem()
        x = np.array([6.0, 0.15, 0.8, 0.3, 0.1, 3.0, 0.2, 200.0])
        f1 = p.scalarize(x)
        f2 = p.scalarize(x)
        assert p.calls == 2
        assert f1 == pytest.approx(f2, abs=1e-9)
        assert isinstance(f1, float)


class TestHyperparameterSearch:
    def test_search_returns_valid_config(self):
        problem = RastriginProblem()
        search = HyperparameterSearch(
            space=MetaSearchSpace(),
            eval_config=MetaEvaluationConfig(
                problem=problem,
                n_islands=1,
                inner_generations=2,
            ),
            n_init=2,
            n_iterations=2,
            seed=1,
        )
        best = search.search()
        assert set(best.keys()) == set(search.space.names)
        assert 4 <= best["population_size"] <= 16
        assert len(search.trajectory) == 4
        assert search.best_fitness() >= search.trajectory[0][1]
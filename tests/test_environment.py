"""Environment layer tests: simulation environment, benchmark functions, fitness and multi-objective ranking."""

import numpy as np
import pytest

from evoagent.environment.benchmarks import (
    AckleyProblem,
    RastriginProblem,
    RosenbrockProblem,
    ZDT1Problem,
)
from evoagent.environment.fitness import (
    hypervolume_2d,
    non_dominated_front,
    pareto_rank_and_crowding,
    reference_point,
    weighted_fitness,
)
from evoagent.environment.simulator import (
    Semiconductor2Objective,
    SemiconductorSimulator,
    to_physical,
)


class TestSimulator:
    def test_output_range(self):
        sim = SemiconductorSimulator()
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = rng.uniform(0, 1, 8)
            y = sim.evaluate_clean(x)
            assert np.all(y >= 0.0) and np.all(y <= 1.0)
            assert y[0] > 0.0  # yield is always positive

    def test_noise_added(self):
        sim = SemiconductorSimulator()
        x = np.array([0.3, 0.7, 0.5, 0.4, 0.6, 0.2, 0.8, 0.5])
        clean = sim.evaluate_clean(x)
        noisy = sim.evaluate(x)
        assert not np.allclose(clean, noisy)

    def test_known_peak_region_high_yield(self):
        sim = SemiconductorSimulator()
        near_peak = np.array([0.3, 0.7, 0.5, 0.4, 0.6, 0.2, 0.8, 0.5])
        far = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        assert sim.evaluate_clean(near_peak)[0] > sim.evaluate_clean(far)[0]

    def test_validate_clips_bounds(self):
        sim = SemiconductorSimulator()
        x = np.array([-0.5, 1.5, 2.0, 0.0, -1.0, 0.5, 0.6, 0.7])
        clipped = sim.validate(x)
        assert np.all(clipped >= 0.0) and np.all(clipped <= 1.0)

    def test_to_physical_mapping(self):
        phys = to_physical(np.zeros(8))
        assert len(phys) == 8
        assert phys["曝光剂量"] == 50.0

    def test_two_objective_variant(self):
        sim2 = Semiconductor2Objective()
        assert sim2.n_objectives == 2
        x = np.zeros(8)
        y = sim2.evaluate_clean(x)
        assert y.shape == (2,)


class TestBenchmarks:
    @pytest.mark.parametrize(
        "cls,optimum",
        [
            (RosenbrockProblem, np.ones(10)),
            (AckleyProblem, np.zeros(10)),
            (RastriginProblem, np.zeros(10)),
        ],
    )
    def test_known_optimum(self, cls, optimum):
        problem = cls()
        f_opt = problem.evaluate_clean(optimum)[0]
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = rng.uniform(*problem.bounds.T)
            assert problem.evaluate_clean(x)[0] >= f_opt - 1e-9

    def test_zdt1_optimum_front(self):
        problem = ZDT1Problem()
        # On the optimal front, f2 = 1 - sqrt(f1)
        x = np.array([0.25] + [0.0] * 9)
        f1, f2 = problem.evaluate_clean(x)
        assert f1 == pytest.approx(0.25)
        assert f2 == pytest.approx(1.0 - np.sqrt(0.25), abs=1e-6)


class TestFitness:
    def test_weighted_fitness_direction(self):
        assert weighted_fitness(np.array([1.0, -0.5]), np.array([1.0, 1.0])) == pytest.approx(0.25)

    def test_non_dominated_front(self):
        points = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.8], [0.5, 0.4]])
        front = non_dominated_front(points)
        assert len(front) == 3  # only (0.5, 0.4) is dominated

    def test_pareto_rank(self):
        points = np.array([[1.0, 1.0], [0.5, 0.5], [0.2, 0.2]])
        ranks, _ = pareto_rank_and_crowding(points)
        assert list(ranks) == [0, 1, 2]

    def test_hypervolume_2d_exact(self):
        # Hypervolume of point (1,1) relative to origin (0,0) = 1
        assert hypervolume_2d(np.array([[1.0, 1.0]]), np.array([0.0, 0.0])) == pytest.approx(1.0)
        # Two non-dominated points (1,0.5),(0.5,1) -> 1*0.5 + (1-0.5)*(1-0.5) = 0.75
        hv = hypervolume_2d(np.array([[1.0, 0.5], [0.5, 1.0]]), np.array([0.0, 0.0]))
        assert hv == pytest.approx(0.75)

    def test_reference_point_worse_than_all(self):
        points = np.array([[0.8, 0.6], [0.5, 0.9], [0.7, 0.7]])
        ref = reference_point(points)
        assert np.all(ref < np.min(points, axis=0))

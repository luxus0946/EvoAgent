"""Agent 个体与策略基因测试。"""

import numpy as np

from evoagent.core.individual import random_genome, random_individual
from evoagent.evolution.operators import (
    crossover_arithmetic,
    crossover_uniform,
    mutate_genome,
    mutate_individual,
)
from evoagent.tools.base import TOOL_NAMES


class TestGenome:
    def test_random_genome_fields_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            genome = random_genome(rng)
            assert genome.initial_tool in TOOL_NAMES
            assert genome.second_tool in TOOL_NAMES
            assert 0.05 <= genome.switch_after_ratio <= 0.95
            assert 0.05 <= genome.stop_patience <= 0.5
            assert len(genome.tool_params) >= 5

    def test_random_genome_multi_objective_weights(self):
        rng = np.random.default_rng(1)
        genome = random_genome(rng, n_objectives=3)
        assert genome.weights is not None
        assert genome.weights.sum() == pytest.approx(1.0)
        assert np.all(genome.weights >= 0)

    def test_clone_independent(self):
        rng = np.random.default_rng(2)
        genome = random_genome(rng)
        clone = genome.clone()
        clone.tool_params["cma_sigma"] = 0.99
        clone.switch_after_ratio = 0.0
        assert genome.tool_params["cma_sigma"] != 0.99
        assert genome.switch_after_ratio != 0.0


class TestOperators:
    def test_mutation_preserves_structure(self):
        rng = np.random.default_rng(3)
        genome = random_genome(rng)
        mutant = mutate_genome(genome.clone(), rate=1.0, rng=rng)
        assert mutant.initial_tool in TOOL_NAMES
        assert 0.05 <= mutant.switch_after_ratio <= 0.95
        for key, value in mutant.tool_params.items():
            assert np.isfinite(value)

    def test_uniform_crossover_children_valid(self):
        rng = np.random.default_rng(4)
        p1, p2 = random_individual(rng), random_individual(rng)
        for _ in range(10):
            child = crossover_uniform(p1, p2, probability=0.5, rng=rng)
            assert child.genome.initial_tool in TOOL_NAMES
            assert child.genome.second_tool in TOOL_NAMES
            assert 0.05 <= child.genome.switch_after_ratio <= 0.95

    def test_arithmetic_crossover_continuous(self):
        rng = np.random.default_rng(5)
        p1, p2 = random_individual(rng), random_individual(rng)
        child = crossover_arithmetic(p1, p2, alpha=0.5)
        expected = 0.5 * (
            p1.genome.switch_after_ratio + p2.genome.switch_after_ratio
        )
        assert child.genome.switch_after_ratio == pytest.approx(expected)

    def test_mutate_individual_returns_clone(self):
        rng = np.random.default_rng(6)
        ind = random_individual(rng)
        mutated = mutate_individual(ind, rate=0.0, rng=rng)
        assert mutated.agent_id == ind.agent_id
        mutated.genome.switch_after_ratio = 0.123
        assert ind.genome.switch_after_ratio != 0.123


import pytest

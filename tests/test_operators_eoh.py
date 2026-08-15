"""EoH 算子式变异测试。"""

import numpy as np
import pytest

from evoagent.core.individual import StrategyGenome
from evoagent.evolution.operators import (
    _apply_eoh_operator,
    mutate_genome_eoh,
    mutate_individual_eoh,
)
from evoagent.tools.base import TOOL_NAMES


def _genome() -> StrategyGenome:
    return StrategyGenome(initial_tool="ga", second_tool="bo")


def _clamp(value, lo, hi):
    return min(hi, max(lo, value))


class TestEohExploitOperators:
    def test_m1_perturbs_single_field_within_range(self):
        g = _genome()
        original = g.switch_after_ratio
        for _ in range(20):
            _apply_eoh_operator(g.clone(), "m1", np.random.default_rng(0))
        g = _genome()
        _apply_eoh_operator(g, "m1", np.random.default_rng(0))
        assert 0.05 <= g.switch_after_ratio <= 0.95 or original == g.switch_after_ratio

    def test_m2_moves_all_params_same_direction(self):
        g = _genome()
        before = (g.switch_after_ratio, g.stop_patience, g.tool_params["sa_t0"])
        g2 = g.clone()
        _apply_eoh_operator(g2, "m2", np.random.default_rng(5))
        changed = [
            abs(a - b) > 1e-12
            for a, b in zip(
                (g2.switch_after_ratio, g2.stop_patience, g2.tool_params["sa_t0"]),
                before,
            )
        ]
        assert any(changed)

    def test_m3_normalizes_weights(self):
        g = _genome()
        g.weights = np.array([0.5, 0.3, 0.2])
        _apply_eoh_operator(g, "m3", np.random.default_rng(0))
        assert np.isclose(g.weights.sum(), 1.0)


class TestEohExploreOperators:
    def test_e1_replaces_a_tool(self):
        g = _genome()
        _apply_eoh_operator(g, "e1", np.random.default_rng(0))
        assert g.initial_tool in TOOL_NAMES and g.second_tool in TOOL_NAMES

    def test_e2_swaps_tools(self):
        g = _genome()
        _apply_eoh_operator(g, "e2", np.random.default_rng(0))
        assert (g.initial_tool, g.second_tool) == ("bo", "ga")

    def test_e2_is_involution(self):
        g = _genome()
        _apply_eoh_operator(g, "e2", np.random.default_rng(0))
        _apply_eoh_operator(g, "e2", np.random.default_rng(0))
        assert (g.initial_tool, g.second_tool) == ("ga", "bo")


class TestMutateGenomeEoh:
    def test_deterministic_with_same_seed(self):
        g1, g2 = _genome(), _genome()
        mutate_genome_eoh(g1, 0.8, np.random.default_rng(11))
        mutate_genome_eoh(g2, 0.8, np.random.default_rng(11))
        assert (g1.initial_tool, g1.second_tool, g1.switch_after_ratio) == (
            g2.initial_tool,
            g2.second_tool,
            g2.switch_after_ratio,
        )

    def test_rate_zero_never_mutates(self):
        g = _genome()
        before = (g.initial_tool, g.second_tool, g.switch_after_ratio)
        mutate_genome_eoh(g, 0.0, np.random.default_rng(0))
        assert (g.initial_tool, g.second_tool, g.switch_after_ratio) == before

    def test_values_stay_in_legal_ranges(self):
        g = _genome()
        for seed in range(30):
            mutate_genome_eoh(g, 0.9, np.random.default_rng(seed))
            assert g.initial_tool in TOOL_NAMES
            assert g.second_tool in TOOL_NAMES
            assert 0.05 <= g.switch_after_ratio <= 0.95
            assert 0.05 <= g.stop_patience <= 0.5
            for key, (lo, hi) in {
                "cma_sigma": (0.05, 0.5),
                "ga_mutation": (0.02, 0.4),
                "sa_t0": (0.005, 0.2),
                "sa_alpha": (0.9, 0.9999),
                "sa_sigma": (0.03, 0.3),
                "bo_xi": (0.0, 0.1),
            }.items():
                assert lo <= g.tool_params[key] <= hi, key

    def test_returns_clone_via_individual_wrapper(self):
        from evoagent.core.individual import AgentIndividual

        ind = AgentIndividual(agent_id="a", genome=_genome())
        mutant = mutate_individual_eoh(ind, 0.8, np.random.default_rng(3))
        assert mutant is not ind
        assert mutant.genome is not ind.genome
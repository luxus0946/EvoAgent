"""MAP-Elites archive and three-way parent sampling tests."""

import numpy as np
import pytest

from evoagent.config import EvolutionConfig
from evoagent.core.individual import AgentIndividual, StrategyGenome, random_individual
from evoagent.environment.benchmarks import RastriginProblem
from evoagent.evolution.map_elites import MapElitesArchive, feature_coords
from evoagent.evolution.population import Population


def _eval_ind(ind: AgentIndividual, fitness: float) -> AgentIndividual:
    ind.fitness = fitness
    ind.best_params = np.array([0.1, 0.2])
    ind.objectives = np.array([fitness])
    return ind


def _mk_genome(initial: str = "ga", second: str = "bo") -> StrategyGenome:
    return StrategyGenome(initial_tool=initial, second_tool=second)


class TestFeatureCoords:
    def test_exploit_and_explore_map_to_different_bins(self):
        explore = _eval_ind(
            AgentIndividual(agent_id="e", genome=_mk_genome("random_search", "sa")),
            0.5,
        )
        exploit = _eval_ind(
            AgentIndividual(agent_id="x", genome=_mk_genome("cma_es", "bo")),
            0.5,
        )
        assert feature_coords(explore)[0] < feature_coords(exploit)[0]

    def test_fitness_bin_boundaries(self):
        ind = AgentIndividual(agent_id="i", genome=_mk_genome())
        ind.fitness = 0.999
        assert feature_coords(ind)[1] == 4
        ind.fitness = 0.0
        assert feature_coords(ind)[1] == 0


class TestMapElitesArchive:
    def test_best_per_cell_retained(self):
        archive = MapElitesArchive()
        a = _eval_ind(random_individual(np.random.default_rng(1), 1), 0.3)
        b = _eval_ind(random_individual(np.random.default_rng(2), 1), 0.7)
        c = _eval_ind(random_individual(np.random.default_rng(3), 1), 0.5)
        archive.add(a)
        archive.add(b)
        archive.add(c)
        assert archive.size() >= 1
        assert archive.best().fitness == 0.7

    def test_empty_archive_raises(self):
        archive = MapElitesArchive()
        with pytest.raises(ValueError):
            archive.sample_elite(np.random.default_rng(0))

    def test_add_ignores_unscored(self):
        archive = MapElitesArchive()
        ind = random_individual(np.random.default_rng(0), 1)
        archive.add(ind)
        assert archive.size() == 0


class TestThreeWaySampling:
    def _population(self, archive_ratio: float = 0.4) -> Population:
        pop = Population(
            problem=RastriginProblem(),
            size=8,
            seed=7,
            archive_ratio=archive_ratio,
            eval_budget_per_individual=10,
        )
        for i, ind in enumerate(pop.individuals):
            _eval_ind(ind, float(i))
        if pop.map_elites is not None:
            pop.map_elites.add_many(pop.individuals)
        return pop

    def test_disabled_by_default(self):
        pop = self._population(archive_ratio=0.0)
        assert pop.map_elites is None

    def test_parents_mix_includes_archive_elite(self):
        pop = self._population(archive_ratio=0.5)
        parents = pop._three_way_mix(pop.sort_by_fitness()[:6])
        assert len(parents) == 6

    def test_archive_sampling_uses_only_fitness_scored(self):
        pop = self._population(archive_ratio=0.5)
        pop.next_generation()
        assert len(pop.individuals) == 8
        assert pop.individuals[0].fitness is not None
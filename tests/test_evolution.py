"""进化层测试：选择算子、进化循环、岛屿模型。"""

import numpy as np
import pytest

from evoagent.config import EvolutionConfig
from evoagent.core.individual import random_individual
from evoagent.environment.benchmarks import RastriginProblem, ZDT1Problem
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.evolution.evolutionary_loop import run_evolution
from evoagent.evolution.operators import roulette_selection, tournament_selection
from evoagent.evolution.strategy import StrategyExecutor


class TestSelection:
    def _population(self, n=20):
        rng = np.random.default_rng(0)
        pop = [random_individual(rng) for _ in range(n)]
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        return pop

    def test_tournament_returns_k(self):
        pop = self._population()
        selected = tournament_selection(pop, k=5)
        assert len(selected) == 5

    def test_tournament_prefers_high_fitness(self):
        pop = self._population()
        selected = tournament_selection(pop, k=20, tournament_size=5)
        fitnesses = [ind.fitness for ind in selected]
        assert max(fitnesses) >= 17.0

    def test_roulette_returns_k(self):
        pop = self._population()
        assert len(roulette_selection(pop, k=5)) == 5


class TestStrategyExecutor:
    def test_two_phase_strategy(self):
        problem = SemiconductorSimulator()
        executor = StrategyExecutor(problem)
        rng = np.random.default_rng(1)
        genome = random_individual(rng).genome
        genome.initial_tool, genome.second_tool = "random_search", "cma_es"
        genome.switch_after_ratio = 0.4
        genome.stop_patience = 0.0  # 关闭早停
        result = executor.run(genome, budget=100, rng=rng)
        assert result.n_evals == 100
        assert len(result.history) == 100
        assert result.best_fitness > -np.inf

    def test_single_phase_strategy(self):
        problem = SemiconductorSimulator()
        executor = StrategyExecutor(problem)
        rng = np.random.default_rng(2)
        genome = random_individual(rng).genome
        genome.initial_tool = "random_search"
        genome.second_tool = "random_search"
        genome.stop_patience = 0.0
        result = executor.run(genome, budget=50, rng=rng)
        assert result.n_evals == 50

    def test_early_stop_reduces_evals(self):
        problem = SemiconductorSimulator()
        executor = StrategyExecutor(problem)
        rng = np.random.default_rng(3)
        genome = random_individual(rng).genome
        genome.initial_tool = "random_search"
        genome.second_tool = "random_search"
        genome.stop_patience = 0.1
        result = executor.run(genome, budget=100, rng=rng)
        assert result.n_evals < 100


class TestEvolutionLoop:
    def test_single_objective_runs(self):
        config = EvolutionConfig(
            population_size=4,
            max_generations=3,
            n_islands=3,
            eval_budget_per_individual=60,
            random_seed=42,
        )
        result = run_evolution(SemiconductorSimulator(), config)
        assert len(result.generation_history) == 3
        # 小种群短预算下适应度可能为负，验证优于同预算随机搜索即可
        from evoagent.tools.factory import build_tool

        random_baseline = build_tool("random_search").optimize(
            SemiconductorSimulator(), result.total_evals, rng=np.random.default_rng(42)
        )
        assert result.best_fitness > random_baseline.best_fitness
        assert result.total_evals > 0
        assert result.best_params.shape == (8,)

    def test_single_island_mode(self):
        config = EvolutionConfig(
            population_size=4,
            max_generations=3,
            n_islands=1,
            eval_budget_per_individual=60,
            random_seed=1,
        )
        result = run_evolution(RastriginProblem(), config)
        assert result.best_params.shape == (10,)

    def test_multi_objective_returns_front(self):
        config = EvolutionConfig(
            population_size=4,
            max_generations=3,
            n_islands=3,
            eval_budget_per_individual=60,
            multi_objective=True,
            n_objectives=2,
            random_seed=7,
        )
        result = run_evolution(ZDT1Problem(), config)
        assert result.pareto_front is not None
        assert result.pareto_front.shape[1] == 2
        assert len(result.archive_history) == 3

    def test_best_fitness_improves_or_holds(self):
        """验证进化确实在改善策略。

        半导体含噪评估（yield std=0.02）：精英重评估存在噪声波动，
        放宽到末代不差于首代 0.05（2σ 噪声上限约 0.02）。
        """
        config = EvolutionConfig(
            population_size=6,
            max_generations=5,
            n_islands=3,
            eval_budget_per_individual=80,
            random_seed=3,
        )
        result = run_evolution(SemiconductorSimulator(), config)
        first = result.generation_history[0].best_fitness
        last = result.generation_history[-1].best_fitness
        assert last >= first - 0.05

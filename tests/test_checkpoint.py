"""Checkpoint and resume-from-checkpoint tests."""

import numpy as np
import pytest

from evoagent.config import EvolutionConfig
from evoagent.environment.benchmarks import RastriginProblem, ZDT1Problem
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.evolution.checkpoint import (
    FORMAT_VERSION,
    load_checkpoint,
    restore_model,
    save_checkpoint,
)
from evoagent.evolution.evolutionary_loop import run_evolution


def _config(seed: int = 42, gens: int = 4, mo: bool = False) -> EvolutionConfig:
    return EvolutionConfig(
        population_size=4,
        max_generations=gens,
        n_islands=3,
        eval_budget_per_individual=60,
        random_seed=seed,
        multi_objective=mo,
        n_objectives=2 if mo else 3,
        fitness_weights=np.array([0.5, 0.3, 0.2]) if not mo else None,
    )


class TestCheckpointRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        problem = SemiconductorSimulator()
        config = _config()
        ckpt_path = tmp_path / "ckpt.json"
        run_evolution(problem, config, checkpoint_path=str(ckpt_path))
        assert ckpt_path.exists()
        ckpt = load_checkpoint(ckpt_path)
        assert ckpt.config["population_size"] == 4
        assert ckpt.generation == config.max_generations - 1
        assert len(ckpt.islands) == 3
        assert all(len(isl["individuals"]) == 4 for isl in ckpt.islands)
        assert len(ckpt.history) == config.max_generations

    def test_restore_matches_interrupted_state(self, tmp_path):
        problem = SemiconductorSimulator()
        config = _config(gens=2)
        ckpt_path = tmp_path / "ckpt.json"
        run_evolution(problem, config, checkpoint_path=str(ckpt_path))
        ckpt = load_checkpoint(ckpt_path)
        model, _ = restore_model(ckpt, problem, config.multi_objective)
        assert model.generation == config.max_generations - 1
        assert len(ckpt.history) == config.max_generations
        best = model.best_individual()
        assert best.fitness is not None
        assert best.best_params is not None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nope.json")

    def test_multi_objective_roundtrip(self, tmp_path):
        problem = ZDT1Problem()
        config = _config(mo=True, gens=2)
        ckpt_path = tmp_path / "ckpt_mo.json"
        run_evolution(problem, config, checkpoint_path=str(ckpt_path))
        ckpt = load_checkpoint(ckpt_path)
        model, state = restore_model(ckpt, problem, True)
        assert len(state["archive"]) > 0


class TestResume:
    def test_resume_equals_uninterrupted(self, tmp_path):
        """Resume-from-checkpoint produces identical results to a single full run (RNG state restored)."""
        problem = SemiconductorSimulator()
        full_config = _config(gens=4)
        full = run_evolution(problem, full_config)

        ckpt_path = tmp_path / "ckpt.json"
        part_config = _config(gens=2)
        run_evolution(problem, part_config, checkpoint_path=str(ckpt_path))

        resumed = run_evolution(
            problem, full_config, checkpoint_path=str(ckpt_path), resume=True
        )
        assert len(resumed.generation_history) == 4
        assert resumed.best_fitness == pytest.approx(full.best_fitness, abs=1e-12)
        assert np.allclose(resumed.best_params, full.best_params)
        assert resumed.total_evals == full.total_evals
        np.testing.assert_allclose(
            [g.best_fitness for g in resumed.generation_history],
            [g.best_fitness for g in full.generation_history],
        )

    def test_resume_without_checkpoint_starts_fresh(self, tmp_path):
        problem = RastriginProblem()
        config = _config(gens=3)
        result = run_evolution(
            problem, config, checkpoint_path=str(tmp_path / "none.json"), resume=True
        )
        assert len(result.generation_history) == 3

    def test_checkpoint_written_each_generation(self, tmp_path):
        problem = RastriginProblem()
        config = _config(gens=3)
        ckpt_path = tmp_path / "ckpt.json"
        run_evolution(problem, config, checkpoint_path=str(ckpt_path))
        ckpt = load_checkpoint(ckpt_path)
        assert ckpt.history[-1]["generation"] == 2
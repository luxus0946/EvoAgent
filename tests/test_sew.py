"""SEW dual-mode (structure + prompt) tests."""

import numpy as np
import pytest

from evoagent.agent.llm import MockLLMClient
from evoagent.agent.workflow import AgentWorkflow
from evoagent.environment.benchmarks import RastriginProblem
from evoagent.evolution.llm_population import LlmPopulation
from evoagent.tools.base import TOOL_NAMES


def _workflow() -> AgentWorkflow:
    return AgentWorkflow(
        problem=RastriginProblem(), budget=20, llm=MockLLMClient()
    )


class TestWorkflowStructureMode:
    def test_structure_mode_produces_valid_individual(self):
        wf = _workflow()
        ind = wf.run(None, seed=1, mode="structure")
        assert ind.genome.initial_tool in TOOL_NAMES
        assert ind.fitness is not None
        assert ind.mode == "structure"

    def test_prompt_mode_defaults_to_neutral(self):
        wf = _workflow()
        ind = wf.run(None, seed=1)
        assert ind.fitness is not None


class TestSewPopulation:
    def _pop(self, sew_ratio: float = 0.5) -> LlmPopulation:
        return LlmPopulation(
            problem=RastriginProblem(),
            size=8,
            seed=3,
            workflow=_workflow(),
            sew_ratio=sew_ratio,
        )

    def test_init_mixes_modes(self):
        pop = self._pop(sew_ratio=0.5)
        modes = [ind.mode for ind in pop.individuals]
        assert modes.count("structure") == 4
        assert modes.count("prompt") == 4
        structure_inds = [i for i in pop.individuals if i.mode == "structure"]
        assert all(i.genome is not None and i.genome_prompt is None for i in structure_inds)
        prompt_inds = [i for i in pop.individuals if i.mode == "prompt"]
        assert all(i.genome is None and i.genome_prompt is not None for i in prompt_inds)

    def test_zero_ratio_is_pure_prompt(self):
        pop = self._pop(sew_ratio=0.0)
        assert all(ind.mode == "prompt" for ind in pop.individuals)

    def test_evaluate_and_evolve_keeps_dual_modes(self):
        pop = self._pop(sew_ratio=0.5)
        pop.evaluate_all()
        assert all(ind.fitness is not None for ind in pop.individuals)
        assert len(pop.best_history) == 1
        pop.next_generation()
        assert len(pop.individuals) == 8
        modes = [ind.mode for ind in pop.individuals]
        assert "structure" in modes and "prompt" in modes

    def test_fixed_prompt_forces_pure_prompt(self):
        pop = LlmPopulation(
            problem=RastriginProblem(),
            size=6,
            seed=3,
            workflow=_workflow(),
            fixed_prompt=True,
            sew_ratio=0.5,
        )
        assert all(ind.mode == "prompt" for ind in pop.individuals)
"""Phase 2 tests: prompt genome, LLM client, strategy generation, knowledge base, agent workflow, LLM population."""

import numpy as np
import pytest

from evoagent.agent.knowledge_base import KnowledgeBase
from evoagent.agent.llm import LLMError, MockLLMClient, _parse_json
from evoagent.agent.strategy_generator import (
    parse_strategy_json,
    parse_strategy_with_fallback,
)
from evoagent.agent.workflow import AgentWorkflow
from evoagent.config import LLMConfig
from evoagent.core.genome_prompt import (
    ROLE_OPTIONS,
    THINKING_STYLE_OPTIONS,
    TOOL_PREFERENCE_OPTIONS,
    EvolvablePrompt,
    crossover_prompt,
    mutate_prompt,
    random_prompt,
)
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.evolution.llm_population import LlmPopulation


class TestGenomePrompt:
    def test_random_prompt_fields_valid(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = random_prompt(rng)
            assert p.role in ROLE_OPTIONS
            assert p.thinking_style in THINKING_STYLE_OPTIONS
            assert p.tool_preference in TOOL_PREFERENCE_OPTIONS
            assert 0.05 <= p.stopping_criteria <= 0.5
            assert 5 <= p.max_iterations <= 50
            assert 0.0 <= p.exploration_bias <= 1.0

    def test_mutate_keeps_ranges(self):
        rng = np.random.default_rng(1)
        p = random_prompt(rng)
        for _ in range(200):
            m = mutate_prompt(p.clone(), rate=0.9, rng=rng)
            assert 0.0 <= m.exploration_bias <= 1.0
            assert 0.05 <= m.stopping_criteria <= 0.5
            assert m.max_iterations >= 5

    def test_mutate_changes_something(self):
        rng = np.random.default_rng(2)
        p = EvolvablePrompt()
        m = mutate_prompt(p.clone(), rate=1.0, rng=rng)
        assert m != p

    def test_crossover_merges_fields(self):
        rng = np.random.default_rng(3)
        p1 = EvolvablePrompt(role="analyst", exploration_bias=0.9)
        p2 = EvolvablePrompt(role="strategist", exploration_bias=0.1)
        child = crossover_prompt(p1, p2, probability=1.0, rng=rng)
        assert child.role in {"analyst", "strategist"}
        assert 0.1 <= child.exploration_bias <= 0.9

    def test_clone_independent(self):
        p = EvolvablePrompt()
        c = p.clone()
        c.exploration_bias = 0.99
        assert p.exploration_bias == 0.5


class TestMockLLM:
    def test_parses_prompt_fields(self):
        client = MockLLMClient()
        user = (
            "问题：半导体工艺参数优化\n"
            "工具偏好: cma_es_first\n"
            "探索偏置: 0.2\n"
            "思维风格: step_by_step\n"
            "收敛阈值: 0.4"
        )
        data = client.chat_json("sys", user)
        assert data["initial_tool"] == "cma_es"
        assert data["second_tool"] == "bo"
        assert data["switch_after_ratio"] < 0.4
        assert data["stop_patience"] > 0.3

    def test_diversify_high_exploration(self):
        client = MockLLMClient()
        user = "工具偏好: diversify_first\n探索偏置: 0.9\n思维风格: tree_of_thought\n收敛阈值: 0.3"
        data = client.chat_json("sys", user)
        assert data["initial_tool"] == "random_search"
        assert data["switch_after_ratio"] >= 0.6

    def test_deterministic_with_seed(self):
        client = MockLLMClient()
        user = "工具偏好: cma_es_first\n探索偏置: 0.5\n思维风格: chain_of_thought\n收敛阈值: 0.3"
        a = client.chat_json("sys", user, seed=7)
        b = client.chat_json("sys", user, seed=7)
        assert a == b

    def test_parse_json_tolerates_code_fence(self):
        raw = '```json\n{"initial_tool": "cma_es"}\n```'
        assert _parse_json(raw) == {"initial_tool": "cma_es"}

    def test_parse_json_raises_without_json(self):
        with pytest.raises(LLMError):
            _parse_json("很抱歉，我无法回答。")


class TestStrategyGenerator:
    def test_valid_parse(self):
        data = {
            "initial_tool": "ga",
            "second_tool": "cma_es",
            "switch_after_ratio": 0.6,
            "stop_patience": 0.2,
            "tool_params": {"cma_sigma": 0.9, "ga_mutation": 0.3},
        }
        g = parse_strategy_json(data)
        assert g is not None
        assert g.initial_tool == "ga"
        assert g.tool_params["cma_sigma"] == pytest.approx(0.5)  # out-of-range values are clamped

    def test_invalid_tool_returns_none(self):
        assert parse_strategy_json({"initial_tool": "magic"}) is None

    def test_fallback_on_invalid(self):
        rng = np.random.default_rng(0)
        g = parse_strategy_with_fallback({"initial_tool": "magic"}, rng)
        assert g.initial_tool in {"cma_es", "bo", "ga", "ppo", "random_search", "sa"}


class TestKnowledgeBase:
    def test_retrieve_relevant(self):
        kb = KnowledgeBase()
        hits = kb.retrieve("cma 精调 连续 黑盒")
        assert len(hits) > 0
        assert "CMA-ES" in hits[0]

    def test_retrieve_empty_query(self):
        assert KnowledgeBase().retrieve("") == []


class TestWorkflow:
    def _workflow(self, seed: int = 0) -> AgentWorkflow:
        return AgentWorkflow(
            SemiconductorSimulator(), budget=50, llm=MockLLMClient()
        )

    def test_run_returns_evaluated_individual(self):
        wf = self._workflow()
        ind = wf.run(random_prompt(np.random.default_rng(0)), seed=1)
        assert ind.fitness is not None
        assert ind.genome is not None
        assert ind.best_params is not None
        assert wf.call_count == 1

    def test_run_deterministic(self):
        prompt = EvolvablePrompt()
        a = self._workflow().run(prompt, seed=42)
        b = self._workflow().run(prompt, seed=42)
        assert a.fitness == b.fitness

    def test_run_differs_by_prompt(self):
        wf = self._workflow()
        low = EvolvablePrompt(tool_preference="diversify_first", exploration_bias=0.9)
        high = EvolvablePrompt(tool_preference="cma_es_first", exploration_bias=0.1)
        assert wf.run(low, seed=5).fitness != wf.run(high, seed=5).fitness

    def test_llm_failure_falls_back(self):
        class BrokenClient(MockLLMClient):
            def chat_json(self, system, user, seed=None):
                raise RuntimeError("api down")

        wf = AgentWorkflow(SemiconductorSimulator(), budget=20, llm=BrokenClient())
        ind = wf.run(EvolvablePrompt(), seed=1)
        assert ind.fitness is not None
        assert wf.call_count == 0


class TestLlmPopulation:
    def _pop(self, fixed: bool = False, size: int = 4, gens: int = 3):
        from evoagent.agent.knowledge_base import KnowledgeBase

        wf = AgentWorkflow(
            SemiconductorSimulator(),
            budget=30,
            llm=MockLLMClient(),
            knowledge_base=KnowledgeBase(),
        )
        return LlmPopulation(
            SemiconductorSimulator(),
            size=size,
            seed=0,
            workflow=wf,
            fixed_prompt=fixed,
        ), gens

    def test_evolution_runs_and_best_history(self):
        pop, gens = self._pop()
        for _ in range(gens):
            pop.evaluate_all()
            pop.next_generation()
        pop.evaluate_all()
        assert len(pop.best_history) == gens + 1
        assert pop.best_individual().fitness is not None
        assert pop.individuals[0].genome is not None

    def test_fixed_prompt_mode_keeps_default(self):
        pop, gens = self._pop(fixed=True)
        for _ in range(gens):
            pop.evaluate_all()
            pop.next_generation()
        pop.evaluate_all()
        assert pop.individuals[0].genome_prompt.tool_preference == "cma_es_first"
        assert pop.best_individual().fitness is not None

    def test_elite_retained(self):
        pop, gens = self._pop()
        pop.evaluate_all()
        best_before = pop.best_individual().clone()
        pop.next_generation()
        bests = [ind.agent_id for ind in pop.individuals]
        assert best_before.agent_id in bests or best_before.fitness in [
            ind.fitness for ind in pop.individuals
        ]


class TestLLMConfig:
    def test_disabled_without_key(self):
        assert not LLMConfig(api_key="").enabled

    def test_enabled_with_key(self):
        assert LLMConfig(api_key="sk-test-123").enabled

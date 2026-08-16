"""LangGraph 编排工作流测试：状态图构建、SEW 双模式、LLM 失败重试/兜底、与确定性工作流等价。"""

import numpy as np
import pytest

from evoagent.agent.graph_workflow import GraphWorkflow, MAX_LLM_RETRIES, build_workflow_graph
from evoagent.agent.llm import MockLLMClient
from evoagent.agent.workflow import AgentWorkflow
from evoagent.core.genome_prompt import EvolvablePrompt, default_prompt
from evoagent.core.individual import AgentIndividual
from evoagent.environment.benchmarks import RastriginProblem
from evoagent.environment.simulator import SemiconductorSimulator


class TestGraphWorkflow:
    def _wf(self, llm: MockLLMClient | None = None) -> GraphWorkflow:
        return GraphWorkflow(
            RastriginProblem(), budget=20, llm=llm or MockLLMClient()
        )

    def test_graph_compiles_and_runs(self):
        wf = self._wf()
        ind = wf.run(default_prompt(), seed=1)
        assert isinstance(ind, AgentIndividual)
        assert ind.fitness is not None
        assert ind.genome is not None
        assert ind.mode == "prompt"
        assert wf.call_count == 1

    def test_structure_mode(self):
        wf = self._wf()
        ind = wf.run(prompt=None, seed=1, mode="structure")
        assert ind.mode == "structure"
        assert ind.fitness is not None

    def test_deterministic(self):
        prompt = EvolvablePrompt()
        a = self._wf().run(prompt, seed=42)
        b = self._wf().run(prompt, seed=42)
        assert a.fitness == b.fitness

    def test_interface_parity_with_agent_workflow(self):
        """同一模拟 LLM + 同种子：图编排与过程式工作流结果一致。"""
        llm = MockLLMClient()
        prompt = default_prompt()
        graph = GraphWorkflow(RastriginProblem(), budget=20, llm=llm)
        proc = AgentWorkflow(RastriginProblem(), budget=20, llm=llm)
        gi = graph.run(prompt, seed=7)
        pi = proc.run(prompt, seed=7)
        assert gi.fitness == pi.fitness

    def test_semiconductor_workflow(self):
        wf = GraphWorkflow(SemiconductorSimulator(), budget=20, llm=MockLLMClient())
        ind = wf.run(default_prompt(), seed=3)
        assert ind.fitness is not None
        assert ind.best_params is not None


class TestGraphFailureHandling:
    class _FlakyClient(MockLLMClient):
        """前 fail_times 次失败，之后成功。"""

        def __init__(self, fail_times: int):
            super().__init__()
            self.fail_times = fail_times
            self.calls = 0

        def chat_json(self, system, user, seed=None):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError("api down")
            return super().chat_json(system, user, seed=seed)

    def test_retries_then_succeeds(self):
        llm = self._FlakyClient(fail_times=2)
        wf = GraphWorkflow(RastriginProblem(), budget=20, llm=llm)
        ind = wf.run(default_prompt(), seed=1)
        assert ind.fitness is not None
        assert llm.calls == 3  # 1 次初始 + 2 次重试

    def test_retries_then_fallback(self):
        llm = self._FlakyClient(fail_times=999)
        wf = GraphWorkflow(RastriginProblem(), budget=20, llm=llm)
        ind = wf.run(default_prompt(), seed=1)
        assert ind.fitness is not None  # 随机策略兜底仍返回评估结果
        assert llm.calls == 1 + MAX_LLM_RETRIES

    def test_success_no_retry(self):
        llm = self._FlakyClient(fail_times=0)
        wf = GraphWorkflow(RastriginProblem(), budget=20, llm=llm)
        wf.run(default_prompt(), seed=1)
        assert llm.calls == 1

    def test_graph_has_expected_nodes(self):
        app = build_workflow_graph(RastriginProblem(), budget=10, llm=MockLLMClient())
        nodes = set(app.get_graph().nodes.keys())
        assert {"retrieve", "build_prompt", "llm_call", "fallback", "parse", "execute"} <= nodes
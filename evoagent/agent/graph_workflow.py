"""LangGraph 编排版 Agent 工作流：将提示词 → LLM → 策略 → 执行 建模为状态图。

与 AgentWorkflow 的差异：工作流是显式的有向状态图（langgraph）——
- 节点：知识检索 → 提示词构建 → LLM 调用 → 策略解析 → 策略执行
- 条件边：LLM 失败自动重试（最多 2 次），仍失败则路由到随机策略兜底
- 状态显式化：检索结果、提示词、LLM 原始输出、解析结果均可中途检查/替换

面试亮点：展示主流 LLM 编排框架（langgraph 状态图）与确定性工作流的等价性，
以及"图路由"相对"过程式 if/else"的可视化与可插拔优势。
"""

import logging
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from evoagent.agent.knowledge_base import KnowledgeBase
from evoagent.agent.llm import LLMClient, build_llm_client, timed_chat
from evoagent.agent.prompts import build_system_prompt, build_user_prompt
from evoagent.agent.strategy_generator import parse_strategy_with_fallback
from evoagent.core.genome_prompt import EvolvablePrompt, default_prompt
from evoagent.core.individual import AgentIndividual
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.strategy import StrategyExecutor
from evoagent.utils.random import get_global_rng

logger = logging.getLogger("evoagent.agent")

MAX_LLM_RETRIES = 2


class GraphState(TypedDict, total=False):
    """图状态：工作流各阶段产物的显式载体。"""

    prompt: EvolvablePrompt | None
    seed: int | None
    mode: str
    knowledge: list[str]
    system_prompt: str
    user_prompt: str
    llm_data: dict
    error: str | None
    llm_retries: int
    genome: Any
    individual: AgentIndividual


def build_workflow_graph(
    problem: OptimizationProblem,
    budget: int = 300,
    llm: LLMClient | None = None,
    knowledge_base: KnowledgeBase | None = None,
    executor: StrategyExecutor | None = None,
    weights: np.ndarray | None = None,
) -> Any:
    """构建 LangGraph 状态图（节点 + 条件边 + 路由）。"""
    llm = llm or build_llm_client()
    knowledge_base = knowledge_base or KnowledgeBase()
    executor = executor or StrategyExecutor(problem, weights=weights)

    def retrieve(state: GraphState) -> dict[str, Any]:
        prompt = state["prompt"]
        if state["mode"] == "structure":
            return {}
        return {
            "knowledge": knowledge_base.retrieve(
                f"{problem.name} {prompt.tool_preference} {prompt.thinking_style}"
            )
        }

    def build_prompt(state: GraphState) -> dict[str, Any]:
        if state["mode"] == "structure":
            return {
                "system_prompt": build_system_prompt(None),
                "user_prompt": build_user_prompt(None, problem),
            }
        prompt = state["prompt"] or default_prompt()
        return {
            "system_prompt": build_system_prompt(prompt),
            "user_prompt": build_user_prompt(prompt, problem, state.get("knowledge")),
        }

    def llm_call(state: GraphState) -> dict[str, Any]:
        retries = state.get("llm_retries", 0)
        try:
            data = timed_chat(llm, state["system_prompt"], state["user_prompt"])
            return {"llm_data": data, "error": None, "llm_retries": retries}
        except Exception as e:  # noqa: BLE001 - LLM 失败由路由处理
            logger.warning("LLM 调用失败（%s），第 %d 次重试", e, retries + 1)
            return {"error": str(e), "llm_retries": retries + 1}

    def route_after_llm(state: GraphState) -> str:
        if state.get("error") is None:
            return "parse"
        if state.get("llm_retries", 0) <= MAX_LLM_RETRIES:
            return "llm_call"
        return "fallback"

    def fallback(state: GraphState) -> dict[str, Any]:
        logger.warning("LLM 连续失败，回退随机策略")
        return {"llm_data": {}}

    def parse(state: GraphState) -> dict[str, Any]:
        rng = get_global_rng() if state.get("seed") is None else np.random.default_rng(state["seed"])
        genome = parse_strategy_with_fallback(state.get("llm_data") or {}, rng)
        return {"genome": genome}

    def execute(state: GraphState) -> dict[str, Any]:
        rng = get_global_rng() if state.get("seed") is None else np.random.default_rng(state["seed"])
        result = executor.run(state["genome"], budget=budget, rng=rng)
        individual = AgentIndividual(agent_id="graph-wf", genome=state["genome"])
        individual.fitness = result.best_fitness
        individual.objectives = problem.objectives_clean(result.best_params)
        individual.best_params = result.best_params
        individual.n_evals = result.n_evals
        individual.n_improvements = result.n_improvements
        individual.mode = state["mode"]
        individual.result = result
        return {"individual": individual}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("llm_call", llm_call)
    graph.add_node("fallback", fallback)
    graph.add_node("parse", parse)
    graph.add_node("execute", execute)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "llm_call")
    graph.add_conditional_edges(
        "llm_call",
        route_after_llm,
        {"parse": "parse", "llm_call": "llm_call", "fallback": "fallback"},
    )
    graph.add_edge("fallback", "parse")
    graph.add_edge("parse", "execute")
    graph.add_edge("execute", END)
    return graph.compile()


class GraphWorkflow:
    """LangGraph 编排版 Agent 工作流（接口与 AgentWorkflow 对齐）。"""

    def __init__(
        self,
        problem: OptimizationProblem,
        budget: int = 300,
        llm: LLMClient | None = None,
        knowledge_base: KnowledgeBase | None = None,
        executor: StrategyExecutor | None = None,
        weights: np.ndarray | None = None,
    ):
        self.problem = problem
        self.budget = budget
        self.llm = llm or build_llm_client()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.executor = executor or StrategyExecutor(problem, weights=weights)
        self.weights = weights
        self.call_count = 0
        self.app = build_workflow_graph(
            problem,
            budget=budget,
            llm=self.llm,
            knowledge_base=self.knowledge_base,
            executor=self.executor,
            weights=weights,
        )

    def run(
        self,
        prompt: EvolvablePrompt | None = None,
        seed: int | None = None,
        mode: str = "prompt",
    ) -> AgentIndividual:
        """执行一次完整图工作流（LLM 失败自动重试 → 随机兜底）。"""
        result = self.app.invoke(
            {
                "prompt": prompt,
                "seed": seed,
                "mode": mode,
                "knowledge": [],
                "error": None,
                "llm_retries": 0,
            }
        )
        self.call_count += 1
        individual = result["individual"]
        individual.agent_id = f"graph-{self.call_count}"
        return individual

    def reset_call_count(self) -> None:
        """重置调用计数。"""
        self.call_count = 0
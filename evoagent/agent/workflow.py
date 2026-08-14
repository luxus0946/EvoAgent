"""Agent 工作流：检索 → 提示词 → LLM → 策略 → 执行。

单次调用流程：
1. 从知识库检索与问题相关的算法指南
2. 将可进化提示词基因渲染为系统/用户提示
3. 调用 LLM 获得策略 JSON
4. 解析校验为 StrategyGenome（非法回退随机）
5. 确定性执行策略并返回评估结果
"""

import logging

import numpy as np

from evoagent.agent.knowledge_base import KnowledgeBase
from evoagent.agent.llm import LLMClient, build_llm_client, timed_chat
from evoagent.agent.prompts import build_system_prompt, build_user_prompt
from evoagent.agent.strategy_generator import parse_strategy_with_fallback
from evoagent.core.genome_prompt import EvolvablePrompt
from evoagent.core.individual import AgentIndividual
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.strategy import StrategyExecutor
from evoagent.utils.random import get_global_rng, set_global_seed

logger = logging.getLogger("evoagent.agent")


class AgentWorkflow:
    """LLM Agent 工作流：提示词基因 → 策略 → 评估结果。"""

    def __init__(
        self,
        problem: OptimizationProblem,
        budget: int = 300,
        llm: LLMClient | None = None,
        knowledge_base: KnowledgeBase | None = None,
        executor: StrategyExecutor | None = None,
        weights: np.ndarray | None = None,
    ):
        """初始化。

        Args:
            problem: 优化问题
            budget: 单策略评估预算
            llm: LLM 客户端（默认自动构建，无 Key 回退模拟）
            knowledge_base: 知识库
            executor: 策略执行器
            weights: 标量化权重（半导体多目标场景）
        """
        self.problem = problem
        self.budget = budget
        self.llm = llm or build_llm_client()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.executor = executor or StrategyExecutor(problem, weights=weights)
        self.weights = weights
        self.call_count = 0

    def run(self, prompt: EvolvablePrompt, seed: int | None = None) -> AgentIndividual:
        """执行一次完整 Agent 流程。

        Args:
            prompt: 可进化提示词基因
            seed: 随机种子（None 使用全局 RNG）

        Returns:
            带评估结果的个体（genome 为 LLM 生成的策略）
        """
        rng = get_global_rng() if seed is None else np.random.default_rng(seed)
        system = build_system_prompt(prompt)
        knowledge = self.knowledge_base.retrieve(
            f"{self.problem.name} {prompt.tool_preference} {prompt.thinking_style}"
        )
        user = build_user_prompt(prompt, self.problem, knowledge)
        try:
            data = timed_chat(self.llm, system, user)
            self.call_count += 1
        except Exception as e:  # noqa: BLE001 - LLM 失败需兜底
            logger.warning("LLM 调用失败（%s），回退随机策略", e)
            data = {}
        genome = parse_strategy_with_fallback(data, rng)
        result = self.executor.run(genome, budget=self.budget, rng=rng)
        individual = AgentIndividual(agent_id=f"wf-{self.call_count}", genome=genome)
        individual.fitness = result.best_fitness
        individual.objectives = self.problem.objectives_clean(result.best_params)
        individual.best_params = result.best_params
        individual.n_evals = result.n_evals
        individual.n_improvements = result.n_improvements
        individual.result = result
        return individual

    def reset_call_count(self) -> None:
        """重置调用计数。"""
        self.call_count = 0

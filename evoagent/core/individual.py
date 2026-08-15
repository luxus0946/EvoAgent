"""Agent 个体与可进化策略基因（算法验证版）。

个体 = 一个可进化的优化策略：选择初始工具、切换时机、早停耐心与工具超参。
个体在仿真环境上执行该策略得到适应度，种群进化驱动策略的自我改进。
"""

import uuid
from dataclasses import dataclass, field

import numpy as np

from evoagent.environment.fitness import normalize_weights
from evoagent.tools.base import TOOL_NAMES

# 连续基因字段的参数范围（变异时保证落在界内）
_TOOL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "cma_sigma": (0.05, 0.5),
    "ga_mutation": (0.02, 0.4),
    "sa_t0": (0.005, 0.2),
    "sa_alpha": (0.9, 0.9999),
    "sa_sigma": (0.03, 0.3),
    "bo_xi": (0.0, 0.1),
}


@dataclass
class StrategyGenome:
    """可进化策略基因。"""

    initial_tool: str = "cma_es"
    second_tool: str = "bo"
    switch_after_ratio: float = 0.5
    stop_patience: float = 0.3
    tool_params: dict[str, float] = field(
        default_factory=lambda: {
            "cma_sigma": 0.25,
            "ga_mutation": 0.15,
            "sa_t0": 0.05,
            "sa_alpha": 0.995,
            "sa_sigma": 0.1,
            "bo_xi": 0.01,
        }
    )
    weights: np.ndarray | None = None

    def clone(self) -> "StrategyGenome":
        """深拷贝基因。"""
        return StrategyGenome(
            initial_tool=self.initial_tool,
            second_tool=self.second_tool,
            switch_after_ratio=self.switch_after_ratio,
            stop_patience=self.stop_patience,
            tool_params=dict(self.tool_params),
            weights=None if self.weights is None else self.weights.copy(),
        )


@dataclass
class AgentIndividual:
    """Agent 个体。"""

    agent_id: str
    genome: StrategyGenome
    fitness: float | None = None
    objectives: np.ndarray | None = None
    pareto_rank: int | None = None
    crowding: float | None = None
    best_params: np.ndarray | None = None
    n_evals: int = 0
    n_improvements: int = 0
    genome_prompt: object | None = None
    mode: str = "prompt"

    def clone(self) -> "AgentIndividual":
        """深拷贝个体（含评估结果字段，供迁移等场景保留适应度）。"""
        clone_prompt = (
            None if self.genome_prompt is None else self.genome_prompt.clone()
        )
        return AgentIndividual(
            agent_id=self.agent_id,
            genome=self.genome.clone(),
            fitness=self.fitness,
            objectives=None if self.objectives is None else self.objectives.copy(),
            pareto_rank=self.pareto_rank,
            crowding=self.crowding,
            best_params=None if self.best_params is None else self.best_params.copy(),
            n_evals=self.n_evals,
            n_improvements=self.n_improvements,
            genome_prompt=clone_prompt,
            mode=self.mode,
        )


def random_genome(rng: np.random.Generator, n_objectives: int = 1) -> StrategyGenome:
    """随机生成一个策略基因。

    Args:
        rng: 随机数生成器
        n_objectives: 目标数量（>1 时随机生成权重向量）

    Returns:
        随机策略基因
    """
    tools = rng.choice(TOOL_NAMES, size=2, replace=True)
    tool_params = {
        key: float(rng.uniform(lo, hi))
        for key, (lo, hi) in _TOOL_PARAM_RANGES.items()
    }
    weights = None
    if n_objectives > 1:
        w = rng.dirichlet(np.ones(n_objectives))
        weights = normalize_weights(w)
    return StrategyGenome(
        initial_tool=str(tools[0]),
        second_tool=str(tools[1]),
        switch_after_ratio=float(rng.uniform(0.05, 0.95)),
        stop_patience=float(rng.uniform(0.05, 0.5)),
        tool_params=tool_params,
        weights=weights,
    )


def random_individual(
    rng: np.random.Generator, n_objectives: int = 1
) -> AgentIndividual:
    """随机生成一个 Agent 个体。"""
    return AgentIndividual(
        agent_id=str(uuid.uuid4())[:8],
        genome=random_genome(rng, n_objectives),
    )


def make_rng_derived(agent_id: str, seed: int) -> np.random.Generator:
    """从个体 ID 派生独立 RNG（保证策略评估可复现）。"""
    return np.random.default_rng(int(seed) + sum(ord(c) for c in agent_id))

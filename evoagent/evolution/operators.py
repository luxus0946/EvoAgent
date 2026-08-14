"""进化算子：选择 / 交叉 / 变异（作用于 Agent 个体的策略基因）。"""

import numpy as np

from evoagent.core.individual import AgentIndividual, StrategyGenome
from evoagent.tools.base import TOOL_NAMES

_TOOL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "cma_sigma": (0.05, 0.5),
    "ga_mutation": (0.02, 0.4),
    "sa_t0": (0.005, 0.2),
    "sa_alpha": (0.9, 0.9999),
    "sa_sigma": (0.03, 0.3),
    "bo_xi": (0.0, 0.1),
}

_CONTINUOUS_FIELDS = ["switch_after_ratio", "stop_patience"]


# ---------------------------------------------------------------- 选择算子


def tournament_selection(
    population: list[AgentIndividual],
    k: int,
    tournament_size: int = 3,
    rng: np.random.Generator | None = None,
) -> list[AgentIndividual]:
    """锦标赛选择：从种群中选 k 个适应度高的个体。

    Args:
        population: 已评估适应度的种群
        k: 选择数量
        tournament_size: 锦标赛规模
        rng: 随机数生成器

    Returns:
        选中的个体列表（可重复）
    """
    if rng is None:
        rng = np.random.default_rng()
    selected: list[AgentIndividual] = []
    for _ in range(k):
        candidates = rng.choice(len(population), tournament_size, replace=False)
        best = max(candidates, key=lambda i: population[i].fitness)
        selected.append(population[int(best)])
    return selected


def roulette_selection(
    population: list[AgentIndividual],
    k: int,
    rng: np.random.Generator | None = None,
) -> list[AgentIndividual]:
    """轮盘赌选择：适应度正比概率选择。"""
    if rng is None:
        rng = np.random.default_rng()
    fitness = np.array([ind.fitness for ind in population], dtype=float)
    prob = fitness - fitness.min() + 1e-12
    if prob.sum() <= 0:
        prob = np.ones_like(prob)
    prob /= prob.sum()
    idx = rng.choice(len(population), size=k, p=prob)
    return [population[i] for i in idx]


def rank_selection(
    population: list[AgentIndividual],
    k: int,
    rng: np.random.Generator | None = None,
) -> list[AgentIndividual]:
    """排名选择：按适应度排名分配线性概率。"""
    if rng is None:
        rng = np.random.default_rng()
    order = np.argsort([ind.fitness for ind in population])
    n = len(population)
    prob = np.arange(1, n + 1, dtype=float)
    prob /= prob.sum()
    idx = rng.choice(order, size=k, p=prob)
    return [population[i] for i in idx]


# ---------------------------------------------------------------- 交叉算子


def crossover_uniform(
    parent1: AgentIndividual,
    parent2: AgentIndividual,
    probability: float = 0.5,
    rng: np.random.Generator | None = None,
) -> AgentIndividual:
    """均匀交叉：基因字段按概率互换，连续参数算术混合。

    Args:
        parent1: 父代 1
        parent2: 父代 2
        probability: 离散字段交换概率
        rng: 随机数生成器

    Returns:
        子代个体
    """
    if rng is None:
        rng = np.random.default_rng()
    g1, g2 = parent1.genome, parent2.genome
    child = AgentIndividual(agent_id=parent1.agent_id, genome=g1.clone())

    if rng.random() < probability:
        child.genome.initial_tool = g2.initial_tool
    if rng.random() < probability:
        child.genome.second_tool = g2.second_tool
    for f in _CONTINUOUS_FIELDS:
        if rng.random() < 0.5:
            setattr(child.genome, f, float(getattr(g2, f)))
        else:
            setattr(child.genome, f, (getattr(g1, f) + getattr(g2, f)) / 2.0)
    for key in child.genome.tool_params:
        if rng.random() < 0.5:
            child.genome.tool_params[key] = float(
                0.5 * (g1.tool_params[key] + g2.tool_params[key])
            )
        elif rng.random() < 0.5:
            child.genome.tool_params[key] = g2.tool_params[key]
    if g1.weights is not None and g2.weights is not None:
        child.genome.weights = normalize_weights_after_mix(g1.weights, g2.weights, rng)
    return child


def crossover_arithmetic(
    parent1: AgentIndividual,
    parent2: AgentIndividual,
    alpha: float = 0.5,
) -> AgentIndividual:
    """算术交叉：连续字段取线性组合。"""
    g1, g2 = parent1.genome, parent2.genome
    child = AgentIndividual(agent_id=parent1.agent_id, genome=g1.clone())
    for f in _CONTINUOUS_FIELDS:
        setattr(
            child.genome,
            f,
            float(alpha * getattr(g1, f) + (1 - alpha) * getattr(g2, f)),
        )
    for key in child.genome.tool_params:
        child.genome.tool_params[key] = float(
            alpha * g1.tool_params[key] + (1 - alpha) * g2.tool_params[key]
        )
    return child


def normalize_weights_after_mix(
    w1: np.ndarray, w2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """混合两个父代权重向量并归一化。"""
    if rng.random() < 0.5:
        mixed = 0.5 * (w1 + w2)
    else:
        mixed = w1 if rng.random() < 0.5 else w2
    total = mixed.sum()
    return mixed / total if total > 0 else np.full_like(mixed, 1.0 / len(mixed))


# ---------------------------------------------------------------- 变异算子


def mutate_genome(
    genome: StrategyGenome,
    rate: float,
    rng: np.random.Generator | None = None,
) -> StrategyGenome:
    """高斯/离散混合变异：以 rate 概率变异每个基因字段。

    Args:
        genome: 待变异基因
        rate: 变异率
        rng: 随机数生成器

    Returns:
        变异后的基因（原对象，调用方负责克隆）
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < rate:
        genome.initial_tool = str(rng.choice([t for t in TOOL_NAMES if t != genome.initial_tool]))
    if rng.random() < rate:
        genome.second_tool = str(rng.choice([t for t in TOOL_NAMES if t != genome.second_tool]))
    for f in _CONTINUOUS_FIELDS:
        if rng.random() < rate:
            lo, hi = _field_range(f)
            value = float(getattr(genome, f))
            setattr(genome, f, float(np.clip(value + rng.normal(0.0, 0.1), lo, hi)))
    for key, (lo, hi) in _TOOL_PARAM_RANGES.items():
        if rng.random() < rate:
            value = genome.tool_params[key]
            genome.tool_params[key] = float(np.clip(value + rng.normal(0.0, 0.1 * (hi - lo)), lo, hi))
    if genome.weights is not None and rng.random() < rate:
        perturb = rng.normal(0.0, 0.1, len(genome.weights))
        genome.weights = normalize_weights_after_mix(genome.weights + perturb, genome.weights, rng)
    return genome


def mutate_individual(
    individual: AgentIndividual,
    rate: float,
    rng: np.random.Generator | None = None,
) -> AgentIndividual:
    """对个体基因执行变异并返回克隆。"""
    if rng is None:
        rng = np.random.default_rng()
    mutant = individual.clone()
    mutate_genome(mutant.genome, rate, rng)
    return mutant


def _field_range(field: str) -> tuple[float, float]:
    return (0.05, 0.95) if field == "switch_after_ratio" else (0.05, 0.5)

"""可进化提示词基因（阶段二：LLM Agent 层）。

对应设计文档 3.2.1：角色、思维风格、工具偏好、收敛阈值、最大迭代、探索偏置。
进化算子（交叉/变异）作用于这些字段，种群进化驱动提示词的自我改进。
"""

import numpy as np
from dataclasses import dataclass

# 离散字段的可选值
ROLE_OPTIONS = ["expert_optimizer", "analyst", "strategist"]
THINKING_STYLE_OPTIONS = ["step_by_step", "chain_of_thought", "tree_of_thought"]
TOOL_PREFERENCE_OPTIONS = ["cma_es_first", "bo_first", "ga_first", "diversify_first"]

# 连续字段范围
_STOPPING_RANGE = (0.05, 0.5)
_EXPLORATION_RANGE = (0.0, 1.0)
_MAX_ITERATIONS_RANGE = (5, 50)


@dataclass
class EvolvablePrompt:
    """可进化提示词基因。"""

    role: str = "expert_optimizer"
    thinking_style: str = "chain_of_thought"
    tool_preference: str = "cma_es_first"
    stopping_criteria: float = 0.3
    max_iterations: int = 20
    exploration_bias: float = 0.5

    def clone(self) -> "EvolvablePrompt":
        """深拷贝提示词基因。"""
        return EvolvablePrompt(
            role=self.role,
            thinking_style=self.thinking_style,
            tool_preference=self.tool_preference,
            stopping_criteria=self.stopping_criteria,
            max_iterations=self.max_iterations,
            exploration_bias=self.exploration_bias,
        )


def random_prompt(rng: np.random.Generator) -> EvolvablePrompt:
    """随机生成一个提示词基因。

    Args:
        rng: 随机数生成器

    Returns:
        随机提示词基因
    """
    return EvolvablePrompt(
        role=str(rng.choice(ROLE_OPTIONS)),
        thinking_style=str(rng.choice(THINKING_STYLE_OPTIONS)),
        tool_preference=str(rng.choice(TOOL_PREFERENCE_OPTIONS)),
        stopping_criteria=float(rng.uniform(*_STOPPING_RANGE)),
        max_iterations=int(rng.integers(*_MAX_ITERATIONS_RANGE)),
        exploration_bias=float(rng.uniform(*_EXPLORATION_RANGE)),
    )


def default_prompt() -> EvolvablePrompt:
    """固定默认提示词（基线用）。"""
    return EvolvablePrompt()


def _pick_other(rng: np.random.Generator, options: list[str], current: str) -> str:
    return str(rng.choice([o for o in options if o != current]))


def mutate_prompt(
    prompt: EvolvablePrompt,
    rate: float,
    rng: np.random.Generator | None = None,
) -> EvolvablePrompt:
    """提示词基因变异（原地修改，调用方负责克隆）。

    Args:
        prompt: 提示词基因
        rate: 变异率
        rng: 随机数生成器

    Returns:
        变异后的提示词基因
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < rate:
        prompt.role = _pick_other(rng, ROLE_OPTIONS, prompt.role)
    if rng.random() < rate:
        prompt.thinking_style = _pick_other(
            rng, THINKING_STYLE_OPTIONS, prompt.thinking_style
        )
    if rng.random() < rate:
        prompt.tool_preference = _pick_other(
            rng, TOOL_PREFERENCE_OPTIONS, prompt.tool_preference
        )
    if rng.random() < rate:
        prompt.stopping_criteria = float(
            np.clip(
                prompt.stopping_criteria + rng.normal(0.0, 0.1),
                *_STOPPING_RANGE,
            )
        )
    if rng.random() < rate:
        prompt.max_iterations = int(
            np.clip(
                int(prompt.max_iterations * rng.normal(1.0, 0.1)),
                *_MAX_ITERATIONS_RANGE,
            )
        )
    if rng.random() < rate:
        prompt.exploration_bias = float(
            np.clip(
                prompt.exploration_bias + rng.normal(0.0, 0.15),
                *_EXPLORATION_RANGE,
            )
        )
    return prompt


def crossover_prompt(
    p1: EvolvablePrompt,
    p2: EvolvablePrompt,
    probability: float = 0.5,
    rng: np.random.Generator | None = None,
) -> EvolvablePrompt:
    """提示词均匀交叉。

    Args:
        p1: 父代 1
        p2: 父代 2
        probability: 离散字段交换概率
        rng: 随机数生成器

    Returns:
        子代提示词基因
    """
    if rng is None:
        rng = np.random.default_rng()
    child = p1.clone()
    if rng.random() < probability:
        child.role = p2.role
    if rng.random() < probability:
        child.thinking_style = p2.thinking_style
    if rng.random() < probability:
        child.tool_preference = p2.tool_preference
    if rng.random() < 0.5:
        child.stopping_criteria = 0.5 * (p1.stopping_criteria + p2.stopping_criteria)
    if rng.random() < 0.5:
        child.max_iterations = (p1.max_iterations + p2.max_iterations) // 2
    if rng.random() < 0.5:
        child.exploration_bias = 0.5 * (p1.exploration_bias + p2.exploration_bias)
    return child

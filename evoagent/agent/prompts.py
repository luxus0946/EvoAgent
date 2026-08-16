"""Prompt templates: render evolvable prompt genomes into system/user prompts."""

from evoagent.core.genome_prompt import EvolvablePrompt
from evoagent.environment.problem import OptimizationProblem
from evoagent.environment.simulator import PARAM_NAMES
from evoagent.tools.base import TOOL_NAMES

_ROLE_TEXT = {
    "expert_optimizer": "你是资深工艺优化专家，深谙进化计算与全局优化方法。",
    "analyst": "你是严谨的优化分析师，擅长基于数据与收敛状态做理性决策。",
    "strategist": "你是战略规划师，擅长在探索与利用之间权衡取舍。",
}

_THINKING_TEXT = {
    "step_by_step": "请分步推理：先分析问题结构，再规划阶段，最后输出策略。",
    "chain_of_thought": "请链式思考：逐步推导每个选择背后的理由。",
    "tree_of_thought": "请像树状搜索一样考虑多种候选策略路径，再择优输出。",
}

_PREFERENCE_TEXT = {
    "cma_es_first": "优先使用 CMA-ES 作为主优化工具。",
    "bo_first": "优先使用贝叶斯优化作为主优化工具。",
    "ga_first": "优先使用遗传算法作为主优化工具。",
    "ppo_first": "优先使用 PPO 强化学习作为主优化工具，让策略网络自适应学习更新方向。",
    "diversify_first": "优先从多样化探索（随机/模拟退火）开始。",
}


def build_system_prompt(prompt: EvolvablePrompt | None) -> str:
    """Build the system prompt (role + output format constraints).

    Args:
        prompt: Evolvable prompt genome (None selects a neutral role for the SEW structure mode)

    Returns:
        System prompt string
    """
    if prompt is None:
        role_text = "你是资深优化专家，熟悉各类全局优化算法。"
        thinking_text = "请先分析问题，再输出策略。"
        preference_text = ""
    else:
        role_text = _ROLE_TEXT[prompt.role]
        thinking_text = _THINKING_TEXT[prompt.thinking_style]
        preference_text = f"{_PREFERENCE_TEXT[prompt.tool_preference]}\n"
    return (
        f"{role_text}\n"
        f"{thinking_text}\n"
        f"{preference_text}"
        "你必须在 JSON 代码块中输出一个完整的优化策略，格式如下：\n"
        "```json\n"
        '{"initial_tool": "cma_es", "second_tool": "bo", '
        '"switch_after_ratio": 0.5, "stop_patience": 0.3, '
        '"tool_params": {"cma_sigma": 0.25, "ga_mutation": 0.15, '
        '"sa_t0": 0.05, "sa_alpha": 0.995, "sa_sigma": 0.1, "bo_xi": 0.01, '
        '"ppo_lr": 0.01, "ppo_clip": 0.2, "ppo_gamma": 0.99}}\n'
        "```\n"
        f"可选工具: {TOOL_NAMES}\n"
        "除 JSON 外不要输出任何其他内容。"
    )


def build_user_prompt(
    prompt: EvolvablePrompt | None,
    problem: OptimizationProblem,
    knowledge: list[str] | None = None,
) -> str:
    """Build the user prompt (problem description + knowledge-base results + preference constraints).

    Args:
        prompt: Evolvable prompt genome (None omits the genome constraint section for the SEW structure mode)
        problem: Optimization problem
        knowledge: Knowledge-base retrieval results

    Returns:
        User prompt string
    """
    param_desc = "\n".join(
        f"- 参数 {i}: {PARAM_NAMES[i]}（归一化范围 [0, 1]）"
        for i in range(problem.dim)
        if i < len(PARAM_NAMES)
    )
    lines = [
        f"问题：{_describe_problem(problem)}",
        f"参数维度：{problem.dim}",
        param_desc,
        f"目标：{problem.objective_names}（最小化目标: {problem.minimize.tolist()}）",
        "",
        "知识库参考：",
    ]
    if knowledge:
        lines += [f"- {k}" for k in knowledge]
    else:
        lines.append("- （无）")
    if prompt is not None:
        lines += [
            "",
            "你的策略约束（必须遵守）：",
            f"工具偏好: {prompt.tool_preference}（主工具选择倾向）",
            f"思维风格: {prompt.thinking_style}（思考方式）",
            f"探索偏置: {prompt.exploration_bias:.2f}（0=纯利用，1=纯探索）",
            f"收敛阈值: {prompt.stopping_criteria:.2f}（连续无改进超过该比例时提前终止）",
            f"最大迭代轮数: {prompt.max_iterations}",
        ]
    return "\n".join(lines)


def _describe_problem(problem: OptimizationProblem) -> str:
    if problem.name == "semiconductor":
        return "半导体工艺参数优化：8 维工艺参数，多峰非线性良率函数，含测量噪声，目标为最大化良率、最小化成本与周期"
    if problem.name == "semiconductor_2obj":
        return "半导体工艺参数优化（双目标）：最大化良率、最小化成本"
    if problem.name == "zdt1":
        return "ZDT1 多目标测试问题：两目标凸 Pareto 前沿"
    return f"{problem.name} 标准基准测试函数"

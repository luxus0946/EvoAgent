"""Tool factory: builds optimization tool instances by name and hyperparameters."""

from evoagent.tools.base import TOOL_NAMES
from evoagent.tools.bayesian_opt import BayesianOptTool
from evoagent.tools.cma_es_tool import CMAESTool
from evoagent.tools.ga_tool import GATool
from evoagent.tools.ppo_tool import PPOTool
from evoagent.tools.random_search import RandomSearchTool
from evoagent.tools.simulated_annealing import SimulatedAnnealingTool


def build_tool(name: str, params: dict[str, float] | None = None) -> object:
    """Build a tool by name and hyperparameters.

    Args:
        name: Tool name (one of TOOL_NAMES)
        params: Tool hyperparameter dictionary

    Returns:
        Tool instance

    Raises:
        ValueError: Unknown tool name
    """
    params = params or {}
    if name == "random_search":
        return RandomSearchTool()
    if name == "sa":
        return SimulatedAnnealingTool(
            t0=params.get("sa_t0", 0.05),
            alpha=params.get("sa_alpha", 0.995),
            sigma=params.get("sa_sigma", 0.1),
        )
    if name == "ga":
        return GATool(mutation_rate=params.get("ga_mutation", 0.15))
    if name == "cma_es":
        return CMAESTool(sigma0=params.get("cma_sigma", 0.25))
    if name == "bo":
        return BayesianOptTool(xi=params.get("bo_xi", 0.01))
    if name == "ppo":
        return PPOTool(
            lr=params.get("ppo_lr", 0.01),
            clip_eps=params.get("ppo_clip", 0.2),
            gamma=params.get("ppo_gamma", 0.99),
        )
    raise ValueError(f"Unknown tool: {name}. Available: {TOOL_NAMES}")

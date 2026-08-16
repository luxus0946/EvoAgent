"""Strategy generator: parses and validates the LLM output JSON into an executable StrategyGenome."""

import logging

import numpy as np

from evoagent.core.individual import StrategyGenome
from evoagent.tools.base import TOOL_NAMES

logger = logging.getLogger("evoagent.agent")

# Valid ranges for tool parameters
_TOOL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "cma_sigma": (0.05, 0.5),
    "ga_mutation": (0.02, 0.4),
    "sa_t0": (0.005, 0.2),
    "sa_alpha": (0.9, 0.9999),
    "sa_sigma": (0.03, 0.3),
    "bo_xi": (0.0, 0.1),
    "ppo_lr": (0.001, 0.05),
    "ppo_clip": (0.05, 0.4),
    "ppo_gamma": (0.9, 0.999),
}

_DEFAULT_PARAMS: dict[str, float] = {
    "cma_sigma": 0.25,
    "ga_mutation": 0.15,
    "sa_t0": 0.05,
    "sa_alpha": 0.995,
    "sa_sigma": 0.1,
    "bo_xi": 0.01,
    "ppo_lr": 0.01,
    "ppo_clip": 0.2,
    "ppo_gamma": 0.99,
}


def parse_strategy_json(data: dict) -> StrategyGenome | None:
    """Parse the LLM output into a strategy genome; return None if invalid.

    Args:
        data: JSON dictionary returned by the LLM

    Returns:
        A valid StrategyGenome, or None if invalid
    """
    try:
        initial = str(data.get("initial_tool", "cma_es"))
        second = str(data.get("second_tool", "bo"))
        if initial not in TOOL_NAMES or second not in TOOL_NAMES:
            logger.warning("LLM 输出非法工具名: %s -> %s", initial, second)
            return None
        switch = _clamp_float(data.get("switch_after_ratio", 0.5), 0.05, 0.95)
        patience = _clamp_float(data.get("stop_patience", 0.3), 0.0, 0.5)
        params = dict(_DEFAULT_PARAMS)
        raw_params = data.get("tool_params")
        if isinstance(raw_params, dict):
            for key, (lo, hi) in _TOOL_PARAM_RANGES.items():
                if key in raw_params:
                    params[key] = float(np.clip(float(raw_params[key]), lo, hi))
        return StrategyGenome(
            initial_tool=initial,
            second_tool=second,
            switch_after_ratio=switch,
            stop_patience=patience,
            tool_params=params,
        )
    except (TypeError, ValueError) as e:
        logger.warning("LLM 策略解析失败: %s", e)
        return None


def parse_strategy_with_fallback(
    data: dict,
    rng: np.random.Generator,
) -> StrategyGenome:
    """Parse the LLM output; fall back to a random strategy if invalid.

    Args:
        data: JSON dictionary returned by the LLM
        rng: Random number generator (for the fallback)

    Returns:
        Strategy genome (from LLM output or random fallback)
    """
    genome = parse_strategy_json(data)
    if genome is None:
        from evoagent.core.individual import random_genome

        genome = random_genome(rng)
        logger.warning("回退为随机策略")
    return genome


def _clamp_float(value, lo: float, hi: float) -> float:
    return float(np.clip(float(value), lo, hi))

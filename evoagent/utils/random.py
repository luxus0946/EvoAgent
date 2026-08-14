"""随机种子管理：保证实验可复现。"""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """设置全局随机种子。

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """创建独立的随机数生成器。

    Args:
        seed: 随机种子

    Returns:
        独立 RNG 实例
    """
    return np.random.default_rng(seed)


_GLOBAL_RNG = np.random.default_rng()


def get_global_rng() -> np.random.Generator:
    """返回全局 RNG（Agent 工作流等无显式种子场景使用）。"""
    return _GLOBAL_RNG


def set_global_seed(seed: int) -> None:
    """重置全局 RNG 种子。"""
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)

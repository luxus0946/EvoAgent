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

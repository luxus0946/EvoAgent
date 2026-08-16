"""Random seed management: ensures reproducible experiments."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set the global random seed.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Create an independent random number generator.

    Args:
        seed: Random seed

    Returns:
        Independent RNG instance
    """
    return np.random.default_rng(seed)


_GLOBAL_RNG = np.random.default_rng()


def get_global_rng() -> np.random.Generator:
    """Return the global RNG (used by the Agent workflow and other contexts without an explicit seed)."""
    return _GLOBAL_RNG


def set_global_seed(seed: int) -> None:
    """Reset the global RNG seed."""
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)

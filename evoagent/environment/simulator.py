"""Semiconductor process surrogate simulation environment.

Simulates the mapping from lithography/etching process parameters to multi-objective metrics:
- input: 8-dimensional normalized process parameters x in [0,1]^8
- output: (yield, cost, cycle_time), all normalized to [0,1]

Design characteristics:
- multimodal: two high-yield peak regions (corresponding to different process windows)
- parameter coupling: a sin coupling term between the peaks
- conflicting objectives: high-yield regions partially conflict with low-cost/short-cycle regions, forming a Pareto trade-off
- noise: Gaussian noise is added to every evaluation to simulate real measurement error

Physical meaning mapping of process parameters (normalized -> physical range):
| index | parameter | physical range |
|------|------|---------|
| 0 | exposure dose | 50 ~ 300 mJ/cm^2 |
| 1 | focal distance | -50 ~ 50 um |
| 2 | temperature | 200 ~ 400 C |
| 3 | pressure | 0.1 ~ 5 Torr |
| 4 | power | 100 ~ 1000 W |
| 5 | processing time | 10 ~ 120 s |
| 6 | gas flow rate | 10 ~ 100 sccm |
| 7 | etching time | 5 ~ 60 s |
"""

import numpy as np

from evoagent.environment.problem import OptimizationProblem

PARAM_NAMES = [
    "曝光剂量",
    "焦距",
    "温度",
    "气压",
    "功率",
    "处理时间",
    "气体流量",
    "蚀刻时间",
]

_PARAM_RANGES_PHYSICAL = [
    (50.0, 300.0),
    (-50.0, 50.0),
    (200.0, 400.0),
    (0.1, 5.0),
    (100.0, 1000.0),
    (10.0, 120.0),
    (10.0, 100.0),
    (5.0, 60.0),
]

# Centers of the two high-yield process windows
_YIELD_PEAK_A = np.array([0.30, 0.70, 0.50, 0.40, 0.60, 0.20, 0.80, 0.50])
_YIELD_PEAK_B = np.array([0.80, 0.20, 0.40, 0.60, 0.30, 0.70, 0.20, 0.60])
_PEAK_WIDTH = 0.40

# Cost-related dimensions: dose, power, flow (energy/material consumption)
_COST_DIMS = np.array([0, 4, 6])
# Cycle-time-related dimensions: processing time, etching time
_CYCLE_DIMS = np.array([5, 7])


def to_physical(x: np.ndarray) -> dict[str, float]:
    """Convert normalized parameters to physical units (for reporting display).

    Args:
        x: normalized parameter vector, shape (8,)

    Returns:
        mapping of parameter name -> physical value
    """
    return {
        name: float(np.clip(x[i], 0, 1) * (hi - lo) + lo)
        for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, _PARAM_RANGES_PHYSICAL))
    }


class SemiconductorSimulator(OptimizationProblem):
    """Semiconductor process surrogate simulation problem (3 objectives: yield/cost/cycle time)."""

    name = "semiconductor"
    dim = 8
    bounds = np.tile([0.0, 1.0], (8, 1))
    objective_names = ["yield", "cost", "cycle_time"]
    minimize = np.array([False, True, True])
    noise_std = 0.02

    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        def _peak(center: np.ndarray, width: float) -> float:
            return float(np.exp(-0.5 * np.sum(((x - center) / width) ** 2)))

        coupling = 1.0 + 0.15 * np.sin(2.0 * np.pi * (x[0] + x[1]))
        yield_rate = 0.10 + 0.52 * _peak(_YIELD_PEAK_A, _PEAK_WIDTH) + 0.34 * _peak(
            _YIELD_PEAK_B, _PEAK_WIDTH
        )
        yield_rate = float(np.clip(yield_rate * coupling, 0.05, 0.98))

        cost = float(np.clip(0.15 + 0.70 * np.mean(x[_COST_DIMS]), 0.05, 0.95))
        cycle = float(np.clip(0.20 + 0.80 * np.mean(x[_CYCLE_DIMS]), 0.05, 0.95))
        return np.array([yield_rate, cost, cycle])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Noisy evaluation: yield noise 0.02, cost/cycle noise 0.01."""
        clean = self.evaluate_clean(x)
        rng = np.random.default_rng(int(np.sum(np.asarray(x) * 1e6)) % (2**32))
        noise = rng.normal(0.0, [self.noise_std, 0.01, 0.01], size=3)
        return np.clip(clean + noise, 0.0, 1.0)


class Semiconductor2Objective(OptimizationProblem):
    """Two-objective variant of the semiconductor surrogate: yield (maximize) vs cost (minimize).

    Used for multi-objective Pareto validation (cycle time is folded into cost to form a 2D trade-off).
    """

    name = "semiconductor_2obj"
    dim = 8
    bounds = np.tile([0.0, 1.0], (8, 1))
    objective_names = ["yield", "cost"]
    minimize = np.array([False, True])
    noise_std = 0.02

    def __init__(self) -> None:
        self._inner = SemiconductorSimulator()

    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        raw = self._inner.evaluate_clean(x)
        return np.array([raw[0], raw[1]])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raw = self._inner.evaluate(x)
        return np.array([raw[0], raw[1]])

"""半导体工艺代理仿真环境。

模拟光刻/刻蚀工艺参数到多目标指标的映射：
- 输入：8 维归一化工艺参数 x ∈ [0,1]^8
- 输出：(yield 良率, cost 成本, cycle_time 周期)，均归一化到 [0,1]

设计特性：
- 多峰：两个高良率峰值区域（对应不同工艺窗口）
- 参数耦合：峰间存在 sin 耦合项
- 冲突目标：良率峰值区与低成本/短周期区域部分冲突，构成 Pareto 折中
- 噪声：每次评估叠加高斯噪声，模拟真实测量误差

工艺参数物理含义映射表（归一化 -> 物理范围）：
| 索引 | 参数 | 物理范围 |
|------|------|---------|
| 0 | 曝光剂量 | 50 ~ 300 mJ/cm^2 |
| 1 | 焦距 | -50 ~ 50 um |
| 2 | 温度 | 200 ~ 400 C |
| 3 | 气压 | 0.1 ~ 5 Torr |
| 4 | 功率 | 100 ~ 1000 W |
| 5 | 处理时间 | 10 ~ 120 s |
| 6 | 气体流量 | 10 ~ 100 sccm |
| 7 | 蚀刻时间 | 5 ~ 60 s |
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

# 两个高良率工艺窗口中心
_YIELD_PEAK_A = np.array([0.30, 0.70, 0.50, 0.40, 0.60, 0.20, 0.80, 0.50])
_YIELD_PEAK_B = np.array([0.80, 0.20, 0.40, 0.60, 0.30, 0.70, 0.20, 0.60])
_PEAK_WIDTH = 0.40

# 成本相关维度：剂量、功率、流量（能耗/物料）
_COST_DIMS = np.array([0, 4, 6])
# 周期相关维度：处理时间、蚀刻时间
_CYCLE_DIMS = np.array([5, 7])


def to_physical(x: np.ndarray) -> dict[str, float]:
    """将归一化参数转换为物理量纲（用于报告展示）。

    Args:
        x: 归一化参数向量，shape (8,)

    Returns:
        参数名 -> 物理值 的映射
    """
    return {
        name: float(np.clip(x[i], 0, 1) * (hi - lo) + lo)
        for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, _PARAM_RANGES_PHYSICAL))
    }


class SemiconductorSimulator(OptimizationProblem):
    """半导体工艺代理仿真问题（3 目标：良率/成本/周期）。"""

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
        """含噪声评估：yield 噪声 0.02，cost/cycle 噪声 0.01。"""
        clean = self.evaluate_clean(x)
        rng = np.random.default_rng(int(np.sum(np.asarray(x) * 1e6)) % (2**32))
        noise = rng.normal(0.0, [self.noise_std, 0.01, 0.01], size=3)
        return np.clip(clean + noise, 0.0, 1.0)


class Semiconductor2Objective(OptimizationProblem):
    """半导体代理仿真双目标变体：良率(最大化) vs 成本(最小化)。

    用于多目标 Pareto 验证（周期目标折叠进成本以构成二维折中）。
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

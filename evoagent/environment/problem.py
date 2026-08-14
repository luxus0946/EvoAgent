"""优化问题统一协议：所有测试问题（半导体代理仿真 + 标准 Benchmark）实现此接口。

约定：
- 原始目标值由 ``evaluate`` / ``evaluate_clean`` 返回，各目标方向由 ``minimize`` 声明；
- ``objectives`` 返回"最大化约定"下的目标向量（最小化目标取负），供多目标排序与指标计算；
- ``scalarize`` 返回加权标量适应度（越大越好）。
"""

from abc import ABC, abstractmethod

import numpy as np


class OptimizationProblem(ABC):
    """优化问题抽象基类。"""

    name: str = "problem"
    dim: int = 0
    bounds: np.ndarray = np.zeros((1, 2))
    objective_names: list[str] = []
    minimize: np.ndarray = np.array([])
    noise_std: float = 0.0

    @property
    def n_objectives(self) -> int:
        """目标数量。"""
        return len(self.objective_names)

    def validate(self, x: np.ndarray) -> np.ndarray:
        """将参数裁剪到边界内。

        Args:
            x: 参数向量，shape (d,) 或 (n, d)

        Returns:
            裁剪后的参数
        """
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def scalarize(self, x: np.ndarray, weights: np.ndarray | None = None) -> float:
        """加权标量适应度（最大化约定），越大越好。

        Args:
            x: 参数向量
            weights: 目标权重（和为 1），None 时使用均匀权重

        Returns:
            标量适应度
        """
        obj = self.objectives(x)
        if weights is None:
            weights = np.ones(obj.shape[-1]) / obj.shape[-1]
        return float(np.sum(np.asarray(weights) * obj))

    def scalarize_clean(self, x: np.ndarray, weights: np.ndarray | None = None) -> float:
        """无噪声加权标量适应度（用于最终指标报告）。"""
        obj = self.objectives_clean(x)
        if weights is None:
            weights = np.ones(obj.shape[-1]) / obj.shape[-1]
        return float(np.sum(np.asarray(weights) * obj))

    @abstractmethod
    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        """无噪声原始目标评估。

        Args:
            x: 参数向量，shape (d,)

        Returns:
            原始目标向量，shape (n_objectives,)
        """
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """含噪声原始目标评估（模拟真实测量误差）。

        Args:
            x: 参数向量，shape (d,)

        Returns:
            原始目标向量，shape (n_objectives,)
        """
        return self.evaluate_clean(x)

    def objectives(self, x: np.ndarray) -> np.ndarray:
        """最大化约定下的目标向量：最小化目标取负。

        Args:
            x: 参数向量

        Returns:
            最大化约定目标向量
        """
        raw = self.evaluate(x)
        sign = np.where(self.minimize, -1.0, 1.0)
        return raw * sign

    def objectives_clean(self, x: np.ndarray) -> np.ndarray:
        """最大化约定下的无噪声目标向量。"""
        raw = self.evaluate_clean(x)
        sign = np.where(self.minimize, -1.0, 1.0)
        return raw * sign

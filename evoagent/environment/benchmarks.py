"""标准基准测试函数：验证算法泛化能力的经典测试集。

包含：Rosenbrock（凸、沟谷）、Ackley（多峰、大量局部最优）、Rastrigin（强多峰）、
ZDT1（两目标 Pareto 测试问题）。全部为最小化问题。
"""

import numpy as np

from evoagent.environment.problem import OptimizationProblem


class _SingleObjectiveProblem(OptimizationProblem):
    """单目标最小化基准问题的公共基类。"""

    objective_names = ["f"]
    minimize = np.array([True])
    noise_std = 0.0

    def _eval(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._eval(np.asarray(x, dtype=float))])


class RosenbrockProblem(_SingleObjectiveProblem):
    """Rosenbrock 函数：经典凸测试（最优 x_i=1, f=0）。"""

    name = "rosenbrock"
    dim = 10
    bounds = np.tile([-5.0, 10.0], (dim, 1))

    def _eval(self, x: np.ndarray) -> float:
        return float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
        )


class AckleyProblem(_SingleObjectiveProblem):
    """Ackley 函数：多峰、大量局部最优（最优 x=0, f=0）。"""

    name = "ackley"
    dim = 10
    bounds = np.tile([-32.0, 32.0], (dim, 1))

    def _eval(self, x: np.ndarray) -> float:
        d = len(x)
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x)))
        return float(term1 + term2 + 20.0 + np.e)


class RastriginProblem(_SingleObjectiveProblem):
    """Rastrigin 函数：强多峰（最优 x=0, f=0）。"""

    name = "rastrigin"
    dim = 10
    bounds = np.tile([-5.12, 5.12], (dim, 1))

    def _eval(self, x: np.ndarray) -> float:
        d = len(x)
        return float(10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


class ZDT1Problem(OptimizationProblem):
    """ZDT1 两目标测试问题（最优 Pareto 前沿 f2 = 1 - sqrt(f1)）。"""

    name = "zdt1"
    dim = 10
    bounds = np.tile([0.0, 1.0], (dim, 1))
    objective_names = ["f1", "f2"]
    minimize = np.array([True, True])
    noise_std = 0.0

    def evaluate_clean(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = 1.0 + 9.0 * np.mean(x[1:])
        f1 = x[0]
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])


BENCHMARK_REGISTRY: dict[str, type[OptimizationProblem]] = {
    "rosenbrock": RosenbrockProblem,
    "ackley": AckleyProblem,
    "rastrigin": RastriginProblem,
    "zdt1": ZDT1Problem,
}

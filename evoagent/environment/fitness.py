"""适应度评估：加权标量法 + NSGA-II 风格 Pareto 非支配排序 + 超体积指标。"""

import numpy as np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """将权重归一化为和为 1。

    Args:
        weights: 权重向量

    Returns:
        归一化权重
    """
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(np.abs(w)))
    return w / total if total > 0 else np.full_like(w, 1.0 / len(w))


def weighted_fitness(objectives: np.ndarray, weights: np.ndarray) -> float:
    """加权标量适应度（最大化约定，越大越好）。

    Args:
        objectives: 目标向量（最大化约定）
        weights: 权重向量

    Returns:
        标量适应度
    """
    return float(np.sum(normalize_weights(weights) * objectives))


def non_dominated_front(objectives: np.ndarray) -> np.ndarray:
    """计算非支配前沿（最大化约定）。

    Args:
        objectives: 目标矩阵，shape (n, m)，最大化约定

    Returns:
        非支配点的索引数组
    """
    n = len(objectives)
    if n == 0:
        return np.array([], dtype=int)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j or dominated[i]:
                continue
            if _dominates(objectives[j], objectives[i]):
                dominated[i] = True
                break
    return np.nonzero(~dominated)[0]


def pareto_rank_and_crowding(
    objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """NSGA-II 非支配排序 + 拥挤度距离。

    Args:
        objectives: 目标矩阵，shape (n, m)，最大化约定

    Returns:
        (rank, crowding)：每个个体的支配等级与拥挤度
    """
    n = len(objectives)
    ranks = np.full(n, -1, dtype=int)
    crowding = np.zeros(n)
    remaining = np.arange(n)
    rank = 0
    while len(remaining) > 0:
        front = non_dominated_front(objectives[remaining])
        idx = remaining[front]
        ranks[idx] = rank
        if len(idx) > 1:
            crowding[idx] = _crowding_distance(objectives[idx])
        remaining = np.delete(remaining, front)
        rank += 1
    return ranks, crowding


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """二维精确超体积（最大化约定）。

    Args:
        points: 目标点集，shape (n, 2)
        ref: 参考点（劣于所有点），shape (2,)

    Returns:
        超体积值
    """
    if len(points) == 0:
        return 0.0
    front = points[non_dominated_front(points)]
    order = np.argsort(-front[:, 0])
    sorted_pts = front[order]
    hv = 0.0
    prev_y = ref[1]
    for p in sorted_pts:
        if p[1] <= prev_y:
            continue
        hv += (p[0] - ref[0]) * (p[1] - prev_y)
        prev_y = p[1]
    return float(hv)


def reference_point(objectives: np.ndarray, margin_ratio: float = 0.1) -> np.ndarray:
    """从数据计算超体积参考点（各维度最差值再向外扩展）。

    Args:
        objectives: 目标矩阵（最大化约定）
        margin_ratio: 外扩比例

    Returns:
        参考点向量
    """
    worst = np.min(objectives, axis=0)
    best = np.max(objectives, axis=0)
    span = best - worst + 1e-9
    return worst - margin_ratio * span


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a 是否 Pareto 支配 b（最大化约定）。"""
    return bool(np.all(a >= b) and np.any(a > b))


def _crowding_distance(front: np.ndarray) -> np.ndarray:
    """计算给定前沿点集的拥挤度距离。"""
    n, m = front.shape
    distances = np.zeros(n)
    for dim in range(m):
        order = np.argsort(front[:, dim])
        sorted_vals = front[order, dim]
        span = sorted_vals[-1] - sorted_vals[0]
        if span <= 0:
            continue
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        for k in range(1, n - 1):
            if np.isinf(distances[order[k]]):
                continue
            distances[order[k]] += (sorted_vals[k + 1] - sorted_vals[k - 1]) / span
    return distances

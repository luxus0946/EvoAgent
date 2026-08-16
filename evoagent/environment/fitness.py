"""Fitness evaluation: weighted scalarization + NSGA-II-style Pareto non-dominated sorting + hypervolume metric."""

import numpy as np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights so they sum to 1.

    Args:
        weights: weight vector

    Returns:
        normalized weights
    """
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(np.abs(w)))
    return w / total if total > 0 else np.full_like(w, 1.0 / len(w))


def weighted_fitness(objectives: np.ndarray, weights: np.ndarray) -> float:
    """Weighted scalar fitness (maximization convention; higher is better).

    Args:
        objectives: objective vector (maximization convention)
        weights: weight vector

    Returns:
        scalar fitness
    """
    return float(np.sum(normalize_weights(weights) * objectives))


def non_dominated_front(objectives: np.ndarray) -> np.ndarray:
    """Compute the non-dominated front (maximization convention).

    Args:
        objectives: objective matrix, shape (n, m), maximization convention

    Returns:
        indices of the non-dominated points
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
    """NSGA-II non-dominated sorting + crowding distance.

    Args:
        objectives: objective matrix, shape (n, m), maximization convention

    Returns:
        (rank, crowding): dominance rank and crowding distance of each individual
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
    """Exact 2D hypervolume (maximization convention).

    Args:
        points: objective points, shape (n, 2)
        ref: reference point (worse than all points), shape (2,)

    Returns:
        hypervolume value
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
    """Compute the hypervolume reference point from the data (worst value per dimension, extended outward).

    Args:
        objectives: objective matrix (maximization convention)
        margin_ratio: outward margin ratio

    Returns:
        reference point vector
    """
    worst = np.min(objectives, axis=0)
    best = np.max(objectives, axis=0)
    span = best - worst + 1e-9
    return worst - margin_ratio * span


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether a Pareto-dominates b (maximization convention)."""
    return bool(np.all(a >= b) and np.any(a > b))


def _crowding_distance(front: np.ndarray) -> np.ndarray:
    """Compute the crowding distance of the given front points."""
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

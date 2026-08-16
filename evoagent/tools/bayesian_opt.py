"""Bayesian optimization tool: Gaussian process surrogate model + EI acquisition function (self-implemented in numpy)."""

import math

import numpy as np

from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, OptimizationTool, ToolResult


class _GaussianProcess:
    """Isotropic RBF-kernel Gaussian process regression (used as the surrogate model)."""

    def __init__(self, lengthscale: float = 0.5, sigma_f: float = 1.0, noise: float = 1e-4):
        self.lengthscale = lengthscale
        self.sigma_f = sigma_f
        self.noise = noise
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.l = np.empty(0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP (standardized targets, Cholesky decomposition)."""
        self.x_train = x
        self.y_train = y
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) or 1.0
        y_norm = (y - self.y_mean) / self.y_std
        k = self._kernel(x, x)
        k += np.eye(len(x)) * (self.noise + 1e-8)
        try:
            self.l = np.linalg.cholesky(k)
            self.alpha = np.linalg.solve(self.l.T, np.linalg.solve(self.l, y_norm))
        except np.linalg.LinAlgError:
            k += np.eye(len(x)) * 1e-6
            self.l = np.linalg.cholesky(k)
            self.alpha = np.linalg.solve(self.l.T, np.linalg.solve(self.l, y_norm))

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation (original scale)."""
        k_s = self._kernel(x, self.x_train)
        mean = k_s @ self.alpha * self.y_std + self.y_mean
        v = np.linalg.solve(self.l, k_s.T)
        var = self.sigma_f**2 - np.sum(v**2, axis=0)
        std = np.sqrt(np.clip(var, 1e-12, None))
        return mean, std

    def log_marginal_likelihood(
        self, lengthscale: float, sigma_f: float, noise: float
    ) -> float:
        """Log marginal likelihood for the given hyperparameters (used for hyperparameter optimization)."""
        k = self._kernel(self.x_train, lengthscale=lengthscale, sigma_f=sigma_f)
        k += np.eye(len(k)) * (noise + 1e-8)
        try:
            l = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return -np.inf
        y_norm = self._y_norm
        alpha = np.linalg.solve(l.T, np.linalg.solve(l, y_norm))
        return float(
            -0.5 * y_norm @ alpha
            - np.sum(np.log(np.diag(l)))
            - 0.5 * len(k) * np.log(2 * np.pi)
        )

    def _kernel(
        self,
        a: np.ndarray,
        b: np.ndarray | None = None,
        lengthscale: float | None = None,
        sigma_f: float | None = None,
    ) -> np.ndarray:
        ls = self.lengthscale if lengthscale is None else lengthscale
        sf = self.sigma_f if sigma_f is None else sigma_f
        if b is None:
            b = self.x_train
        sq = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2 * a @ b.T
        return sf**2 * np.exp(-0.5 * sq / ls**2)


class BayesianOptTool(OptimizationTool):
    """Bayesian optimization: GP regression + EI acquisition, highly efficient for small samples.

    Hyperparameters (evolvable via the strategy genome):
    - bo_xi: EI exploration coefficient
    - bo_max_train: GP training set cap (beyond it, keep the most recent points plus a uniform subsample)
    """

    name = "bo"

    def __init__(self, xi: float = 0.01, max_train: int = 80, n_candidates: int = 400):
        self.xi = xi
        self.max_train = max_train
        self.n_candidates = n_candidates

    def optimize(
        self,
        problem: OptimizationProblem,
        budget: int,
        weights: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
        early_stop: EarlyStopMonitor | None = None,
        rng: np.random.Generator | None = None,
    ) -> ToolResult:
        if rng is None:
            rng = np.random.default_rng()
        low, high = problem.bounds[:, 0], problem.bounds[:, 1]
        n_init = min(6, max(3, budget // 20))

        x_hist: list[np.ndarray] = []
        f_hist: list[float] = []
        if x_init is not None:
            x_hist.append(x_init.copy())
            f_hist.append(problem.scalarize(x_init, weights))
            n_init = max(0, n_init - 1)
        for _ in range(n_init):
            x = rng.uniform(low, high, problem.dim)
            x_hist.append(x)
            f_hist.append(problem.scalarize(x, weights))

        x_arr = np.array(x_hist)
        f_arr = np.array(f_hist)
        best_idx = int(np.argmax(f_arr))
        best_params = x_arr[best_idx].copy()
        best_fitness = float(f_arr[best_idx])
        history = list(f_arr)
        n_evals = len(f_arr)
        n_improvements = 1

        while n_evals < budget:
            train_x, train_y = self._training_subset(x_arr, f_arr, rng)
            gp = self._fit_gp(train_x, train_y, rng)
            x_next = self._maximize_ei(gp, problem, low, high, rng)
            f_next = problem.scalarize(x_next, weights)
            n_evals += 1
            x_arr = np.vstack([x_arr, x_next])
            f_arr = np.append(f_arr, f_next)

            if f_next > best_fitness:
                best_fitness = float(f_next)
                best_params = x_next.copy()
                n_improvements += 1
            history.append(best_fitness)

            if early_stop is not None and early_stop.check(f_next, best_fitness):
                break

        return ToolResult(
            best_params=best_params,
            best_fitness=best_fitness,
            history=history,
            n_evals=n_evals,
            n_improvements=n_improvements,
        )

    def _training_subset(
        self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cap the GP training set size: keep the most recent half plus a uniformly sampled half when over the limit."""
        if len(x) <= self.max_train:
            return x, y
        recent = self.max_train // 2
        keep = list(range(len(x) - recent, len(x)))
        rest = list(range(len(x) - recent))
        sampled = rng.choice(rest, self.max_train - recent, replace=False)
        keep = sorted(keep + list(sampled))
        return x[keep], y[keep]

    def _fit_gp(
        self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> _GaussianProcess:
        """Randomly search GP hyperparameters (lengthscale / signal variance / noise)."""
        gp = _GaussianProcess()
        gp.x_train = x
        gp._y_norm = (y - np.mean(y)) / (np.std(y) or 1.0)
        best_ll, best_h = -np.inf, (0.5, 1.0, 1e-3)
        for _ in range(6):
            ls = float(rng.uniform(0.1, 2.0))
            sf = float(rng.uniform(0.5, 3.0))
            noise = float(rng.uniform(1e-4, 0.1))
            ll = gp.log_marginal_likelihood(ls, sf, noise)
            if ll > best_ll:
                best_ll, best_h = ll, (ls, sf, noise)
        gp.lengthscale, gp.sigma_f, gp.noise = best_h
        gp.fit(x, y)
        return gp

    def _maximize_ei(
        self,
        gp: _GaussianProcess,
        problem: OptimizationProblem,
        low: np.ndarray,
        high: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Maximize the EI acquisition function via random candidates plus local refinement."""
        candidates = rng.uniform(low, high, size=(self.n_candidates, problem.dim))
        best = float(np.max(gp.y_train))
        mean, std = gp.predict(candidates)
        z = (mean - best - self.xi) / std
        ei = std * (z * _normal_cdf(z) + _normal_pdf(z))
        ei[std <= 1e-9] = 0.0
        top = np.argsort(-ei)[:10]
        for _ in range(10):
            refine = candidates[top] + rng.normal(0.0, 0.02, size=(len(top), problem.dim))
            refine = np.clip(refine, low, high)
            mean_r, std_r = gp.predict(refine)
            z_r = (mean_r - best - self.xi) / std_r
            ei_r = std_r * (z_r * _normal_cdf(z_r) + _normal_pdf(z_r))
            if np.max(ei_r) > np.max(ei[top]):
                merged = np.vstack([candidates, refine])
                merged_ei = np.concatenate([ei, ei_r])
                top = np.argsort(-merged_ei)[:10]
                candidates, ei = merged, merged_ei
        return candidates[int(np.argmax(ei))]


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    erf = np.frompyfunc(math.erf, 1, 1)
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0))).astype(float)


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

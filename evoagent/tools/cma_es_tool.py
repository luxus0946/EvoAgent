"""CMA-ES tool: covariance matrix adaptation evolution strategy (simplified Hansen implementation, self-implemented in numpy)."""

import numpy as np

from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, OptimizationTool, ToolResult


class CMAESTool(OptimizationTool):
    """CMA-ES (mu, lambda) evolution strategy for black-box optimization in continuous spaces.

    Hyperparameters (evolvable via the strategy genome):
    - sigma0: Initial step size (relative to the parameter range)
    """

    name = "cma_es"

    def __init__(self, sigma0: float = 0.25):
        self.sigma0 = sigma0

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
        span = high - low
        d = problem.dim

        lam = 4 + int(3 * np.log(d))
        mu = max(1, lam // 2)
        if x_init is not None:
            x_mean = x_init.copy()
        else:
            x_mean = (low + high) / 2.0
        sigma = self.sigma0 * span.mean()

        weights_mu = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_mu /= weights_mu.sum()
        mu_eff = 1.0 / np.sum(weights_mu**2)
        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        c_cov = 4.0 / (d + 4.0 + 2.0 * mu_eff / d) / (1.0 + 2.0 * mu_eff / (d + 6.0))
        c_mu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) ** 2 + mu_eff)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0) + c_sigma
        chi_n = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))

        identity = np.eye(d)
        cov = identity.copy()
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        eigenvals = np.ones(d)
        eigenvecs = identity.copy()
        inv_sqrt_cov = identity.copy()

        best_params = x_mean.copy()
        best_fitness = -np.inf
        history: list[float] = []
        n_evals = 0
        n_improvements = 0

        while n_evals + lam <= budget:
            z = rng.normal(0.0, 1.0, size=(lam, d))
            x_pop = x_mean + sigma * (z @ (eigenvecs * np.sqrt(eigenvals)).T)
            x_pop = np.clip(x_pop, low, high)
            fitness = np.array([problem.scalarize(x, weights) for x in x_pop])
            n_evals += lam

            order = np.argsort(-fitness)
            x_sort = x_pop[order]
            f_sort = fitness[order]

            if f_sort[0] > best_fitness:
                best_fitness = float(f_sort[0])
                best_params = x_sort[0].copy()
                n_improvements += 1

            y = (x_sort[:mu] - x_mean) / sigma
            x_mean = x_mean + sigma * (weights_mu @ y)

            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2.0 - c_sigma) * mu_eff
            ) * (y.T @ weights_mu)
            h_sigma = (
                1.0
                if np.linalg.norm(p_sigma) / np.sqrt(
                    1.0 - (1.0 - c_sigma) ** (2.0 * n_evals / lam)
                )
                < (1.4 + 2.0 / (d + 1.0)) * chi_n
                else 0.0
            )
            p_c = (1.0 - c_cov) * p_c + h_sigma * np.sqrt(
                c_cov * (2.0 - c_cov) * mu_eff
            ) * (y.T @ weights_mu)
            cov = (
                (1.0 - c_cov - c_mu) * cov
                + c_cov
                * (
                    np.outer(p_c, p_c)
                    + (1.0 - h_sigma) * c_cov * (2.0 - c_cov) * cov
                )
                + c_mu
                * ((weights_mu * y.T) @ y)
            )
            sigma *= np.exp(
                (c_sigma / d_sigma)
                * (np.linalg.norm(p_sigma) / chi_n - 1.0)
            )

            eigenvals, eigenvecs = np.linalg.eigh(cov)
            eigenvals = np.clip(eigenvals, 1e-16, None)
            inv_sqrt_cov = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T

            history.extend([best_fitness] * lam)
            if early_stop is not None and early_stop.check(f_sort[0], best_fitness):
                break

        return ToolResult(
            best_params=best_params,
            best_fitness=best_fitness,
            history=history,
            n_evals=n_evals,
            n_improvements=n_improvements,
        )

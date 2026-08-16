"""PPO tool tests: RL optimizer correctness + effect vs baseline + analytic gradient check."""

import numpy as np
import pytest

from evoagent.environment.benchmarks import AckleyProblem, RosenbrockProblem
from evoagent.tools.base import TOOL_NAMES
from evoagent.tools.factory import build_tool
from evoagent.tools.ppo_tool import (
    _PPONetworks,
    _mlp_forward,
    _policy_gradients,
    PPOTool,
)

SEEDS = [1, 7, 42]


class TestPPOToolBasics:
    def test_budget_respected(self):
        r = PPOTool().optimize(AckleyProblem(), budget=200, rng=np.random.default_rng(0))
        assert r.n_evals <= 200
        assert len(r.history) == r.n_evals

    def test_returns_tool_result_fields(self):
        r = PPOTool().optimize(AckleyProblem(), budget=100, rng=np.random.default_rng(0))
        assert r.best_params.shape == (AckleyProblem().dim,)
        assert r.best_fitness is not None
        assert r.n_improvements >= 0

    def test_factory_builds_ppo(self):
        assert "ppo" in TOOL_NAMES
        tool = build_tool("ppo", {"ppo_lr": 0.02, "ppo_clip": 0.3, "ppo_gamma": 0.98})
        assert isinstance(tool, PPOTool)
        assert tool.lr == 0.02 and tool.clip_eps == 0.3 and tool.gamma == 0.98

    def test_factory_unknown_tool_still_raises(self):
        with pytest.raises(ValueError):
            build_tool("nope")


class TestPPOEffectiveness:
    """RL learning evidence: PPO should clearly beat random search on Ackley (budget 300)."""

    def test_ppo_beats_random_search(self):
        p = AckleyProblem()
        rs = [RandomSearch().optimize(p, budget=300, rng=np.random.default_rng(s)).best_fitness for s in SEEDS]
        ppo = [PPOTool().optimize(p, budget=300, rng=np.random.default_rng(s)).best_fitness for s in SEEDS]
        assert np.mean(rs) - np.mean(ppo) > 0.5

    def test_ppo_deterministic(self):
        p = AckleyProblem()
        a = PPOTool().optimize(p, budget=300, rng=np.random.default_rng(5))
        b = PPOTool().optimize(p, budget=300, rng=np.random.default_rng(5))
        assert a.best_fitness == b.best_fitness
        assert np.allclose(a.best_params, b.best_params)


class TestPPOGradients:
    """Analytic gradients vs numeric gradients (PPO clipped objective)."""

    @staticmethod
    def _make(net: _PPONetworks, s: np.ndarray, a: np.ndarray, logp_old: np.ndarray, adv: np.ndarray):
        h, mean = _mlp_forward(s, net.w1, net.b1, net.w2, net.b2)
        return _policy_gradients(s, a, adv, logp_old, mean, h, net, 0.2)

    @staticmethod
    def _loss(net: _PPONetworks, s: np.ndarray, a: np.ndarray, logp_old: np.ndarray, adv: np.ndarray) -> float:
        """PPO objective L = min(r*A, clip(r)*A), taking the correct branch per adv sign."""
        _, m = _mlp_forward(s, net.w1, net.b1, net.w2, net.b2)
        lp = -0.5 * np.sum(((a - m) / np.exp(net.log_std)) ** 2, axis=1) - np.sum(net.log_std)
        ratio = np.exp(lp - logp_old)
        clipped = np.clip(ratio, 0.8, 1.2)
        chosen = np.where(adv > 0, np.minimum(ratio, clipped), np.maximum(ratio, clipped))
        return float(np.mean(adv * chosen))

    def test_gradients_match_numeric(self):
        rng = np.random.default_rng(0)
        net = _PPONetworks(3, 2, seed=1)
        s = rng.normal(size=(4, 3))
        h, mean = _mlp_forward(s, net.w1, net.b1, net.w2, net.b2)
        a = mean + np.exp(net.log_std) * rng.normal(size=(4, 2))
        adv = rng.normal(size=4)
        logp_old = np.full(4, -1.5)
        gw1, gb1, gw2, gb2, gls = self._make(net, s, a, logp_old, adv)

        eps = 1e-6
        for name, grad, shape in [
            ("w1", gw1, net.w1.shape),
            ("b1", gb1, net.b1.shape),
            ("w2", gw2, net.w2.shape),
            ("b2", gb2, net.b2.shape),
            ("log_std", gls, net.log_std.shape),
        ]:
            # Gradients corrected by 2x numeric gradient norm (small magnitude after sample averaging)
            grad_flat = grad.ravel()
            num_flat = np.zeros_like(grad_flat)
            for i in range(grad_flat.size):
                param = getattr(net, name).ravel()
                orig = param[i]
                param[i] = orig + eps
                lp = self._loss(net, s, a, logp_old, adv)
                param[i] = orig - eps
                lm = self._loss(net, s, a, logp_old, adv)
                param[i] = orig
                num_flat[i] = (lp - lm) / (2 * eps)
            scale = max(np.linalg.norm(grad_flat), 1e-8)
            assert np.linalg.norm(grad_flat - num_flat) / scale < 0.05, name


def RandomSearch():
    from evoagent.tools.random_search import RandomSearchTool

    return RandomSearchTool()
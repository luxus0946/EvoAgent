"""PPO tool: policy-gradient reinforcement learning optimizer (hand-written in numpy, no external dependencies).

Role in the framework: the RL member of the optimization tool pool -- the policy network
learns "how to update the current solution", and PPO (Proximal Policy Optimization)
stabilizes updates with a clipped objective plus GAE.

Design:
- State: current solution x (d-dim) + current best fitness (1-dim), d+1 dims total
- Action: update step delta (d-dim) sampled from a Gaussian policy (MLP outputs the mean, log_std is learned)
- Reward: relative fitness increment r = f(x') - f(x) (a dense reward meaning "improve at every step")
- Each rollout collects one trajectory; PPO updates (GAE + clip + value network) run for multiple epochs
- Networks are 2-layer MLPs (tanh) with hand-written forward and backward passes (analytic gradients)
"""

import numpy as np

from evoagent.environment.problem import OptimizationProblem
from evoagent.tools.base import EarlyStopMonitor, OptimizationTool, ToolResult


def _mlp_forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-layer MLP forward pass: x -> tanh(xW1+b1) -> W2 + b2. Returns (hidden, output)."""
    h = np.tanh(x @ w1 + b1)
    out = h @ w2 + b2
    return h, out


class _PPONetworks:
    """PPO policy network (Gaussian mean) and value network with hand-written backpropagation."""

    def __init__(self, state_dim: int, act_dim: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.3, (state_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 0.3, (hidden, act_dim))
        self.b2 = np.zeros(act_dim)
        self.log_std = np.full(act_dim, -1.0)
        self.vw1 = rng.normal(0, 0.3, (state_dim, hidden))
        self.vb1 = np.zeros(hidden)
        self.vw2 = rng.normal(0, 0.3, (hidden, 1))
        self.vb2 = np.zeros(1)

    def act(self, s: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample an action and its log-probability (constant term omitted, consistent with the PPO objective)."""
        _, mean = _mlp_forward(s, self.w1, self.b1, self.w2, self.b2)
        std = np.exp(self.log_std)
        a = mean + std * rng.normal(size=mean.shape)
        logp = -0.5 * np.sum((a - mean) ** 2 / std**2 + 2 * self.log_std)
        return a, float(logp)

    def value(self, s: np.ndarray) -> float:
        """Value estimate (scalar)."""
        _, out = _mlp_forward(s, self.vw1, self.vb1, self.vw2, self.vb2)
        return float(out[0])

    def policy_and_value(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch forward pass: returns (mean, value, hidden activations)."""
        h, mean = _mlp_forward(states, self.w1, self.b1, self.w2, self.b2)
        v = _mlp_forward(states, self.vw1, self.vb1, self.vw2, self.vb2)[1]
        return mean, v, h


def _policy_gradients(
    states: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    logp_old: np.ndarray,
    mean: np.ndarray,
    hidden: np.ndarray,
    net: _PPONetworks,
    clip_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic gradients of the PPO clipped objective L = min(r*A, clip(r)*A).

    Returns:
        (gw1, gb1, gw2, gb2, g_logstd): gradients of each policy network parameter
    """
    std = np.exp(net.log_std)
    inv_std = 1.0 / std
    dz = (actions - mean) * inv_std  # (n, d)
    # log pi(a|s) = -0.5*sum(dz^2) - sum(log_std) (consistent with act, no constant term)
    logp_new = -0.5 * np.sum(dz**2, axis=1) - np.sum(net.log_std)
    ratio = np.exp(logp_new - np.array(logp_old))
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    # Min branch: take the unclipped term when adv*ratio <= adv*clipped
    mask = (advantages * ratio <= advantages * clipped).astype(float)
    coef = mask * advantages * ratio  # (n,)
    n = len(states)
    # d logp / d mean = dz * inv_std; d logp / d log_std = dz^2 - 1 (averaged over trajectory length)
    g_mean = coef[:, None] * (dz * inv_std) / n
    g_logstd = np.sum(coef[:, None] * (dz**2 - 1.0), axis=0) / n
    # Backpropagate through the two-layer MLP
    dh2 = g_mean @ net.w2.T
    dh = dh2 * (1 - hidden**2)
    gw2 = hidden.T @ g_mean
    gb2 = np.sum(g_mean, axis=0)
    gw1 = states.T @ dh
    gb1 = np.sum(dh, axis=0)
    return gw1, gb1, gw2, gb2, g_logstd


class PPOTool(OptimizationTool):
    """PPO reinforcement learning optimizer: the policy network learns parameter update directions.

    Hyperparameters (evolvable via the strategy genome):
    - ppo_lr: Learning rate
    - ppo_clip: PPO clipping coefficient
    - ppo_gamma: GAE discount factor
    """

    name = "ppo"

    def __init__(
        self,
        lr: float = 0.01,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epochs: int = 4,
        hidden: int = 16,
        step_scale: float = 0.1,
    ):
        self.lr = lr
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.hidden = hidden
        self.step_scale = step_scale

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

        x = (x_init.copy() if x_init is not None else rng.uniform(low, high)).astype(float)
        f = float(problem.scalarize(x, weights))
        best_params, best_fitness = x.copy(), f
        net = _PPONetworks(state_dim=d + 1, act_dim=d, hidden=self.hidden, seed=int(rng.integers(2**31)))
        history = [best_fitness]
        n_evals = 1
        n_improvements = 1

        def _state(v: np.ndarray, fv: float) -> np.ndarray:
            """Normalized state: parameters in [-1, 1], fitness compressed via tanh."""
            x_norm = (v - low) / span * 2.0 - 1.0
            return np.concatenate([x_norm, [float(np.tanh(fv / 10.0))]])

        while n_evals < budget:
            # ---- rollout: collect one trajectory (normalized action space, clipped rewards to prevent explosion) ----
            states, actions, rewards, values, logps, dones = [], [], [], [], [], []
            s = _state(x, f)
            for _ in range(min(64, budget - n_evals)):
                a, logp = net.act(s, rng)
                x_new = np.clip(x + a * span * self.step_scale, low, high)
                f_new = float(problem.scalarize(x_new, weights))
                n_evals += 1
                # Relative improvement reward: sign * min(1, |df| / (|f|+1)).
                # f spans multiple orders of magnitude (e.g. millions for Rosenbrock),
                # so the relative form does not saturate and preserves both the
                # improvement direction and magnitude
                df = f_new - f
                sign = 1.0 if df > 0 else (-1.0 if df < 0 else 0.0)
                rewards.append(sign * min(1.0, abs(df) / (abs(f) + 1.0)))
                states.append(s)
                actions.append(a)
                logps.append(logp)
                values.append(net.value(s))
                dones.append(n_evals >= budget)
                x, f = x_new, f_new
                if f > best_fitness:
                    best_fitness, best_params = f, x.copy()
                    n_improvements += 1
                history.append(best_fitness)
                s = _state(x, f)
                if early_stop is not None and early_stop.check(f, best_fitness):
                    break
                if n_evals >= budget:
                    break

            states_a = np.array(states)
            actions_a = np.array(actions)
            rewards_a = np.array(rewards)
            logps_a = np.array(logps)

            # ---- GAE ----
            values_a = np.array(values)
            next_v = net.value(s)
            adv = np.zeros(len(rewards))
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards_a[t] + self.gamma * (0.0 if dones[t] else next_v) - values_a[t]
                gae = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else gae)
                adv[t] = gae
                next_v = values_a[t]
            returns = adv + values_a
            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ---- PPO update (multiple epochs, full batch) ----
            mean, v_pred, hidden = net.policy_and_value(states_a)
            v_target = returns.reshape(-1, 1)
            for _ in range(self.epochs):
                gw1, gb1, gw2, gb2, g_logstd = _policy_gradients(
                    states_a, actions_a, adv_norm, logps_a,
                    mean, hidden, net, self.clip_eps,
                )
                net.w1 += self.lr * gw1
                net.b1 += self.lr * gb1
                net.w2 += self.lr * gw2
                net.b2 += self.lr * gb2
                net.log_std += self.lr * g_logstd
                # Value network (MSE, single gradient step)
                dv2 = 2 * (v_pred - v_target) / len(states_a)
                hh = np.tanh(states_a @ net.vw1 + net.vb1)
                dh_v = dv2 @ net.vw2.T
                dh_v2 = dh_v * (1 - hh**2)
                net.vw2 -= self.lr * hh.T @ dv2
                net.vb2 -= self.lr * np.sum(dv2, axis=0)
                net.vw1 -= self.lr * states_a.T @ dh_v2
                net.vb1 -= self.lr * np.sum(dh_v2, axis=0)
                mean, v_pred, hidden = net.policy_and_value(states_a)

        return ToolResult(
            best_params=best_params,
            best_fitness=best_fitness,
            history=history,
            n_evals=n_evals,
            n_improvements=n_improvements,
        )
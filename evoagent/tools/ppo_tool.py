"""PPO 工具：策略梯度强化学习优化器（numpy 手写实现，无外部依赖）。

面试亮点：优化工具池中的 RL 成员——策略网络学习"如何更新当前解"，
PPO（Proximal Policy Optimization）用裁剪目标 + GAE 稳定更新。

设计：
- 状态: 当前解 x（d 维）+ 当前最优适应度（1 维），共 d+1 维
- 动作: 高斯策略（MLP 输出均值，学习 log_std）采样的更新步长 delta（d 维）
- 奖励: 相对适应度增量 r = f(x') - f(x)（即"每一步都要变好"的密集奖励）
- 每轮 rollout 收集一条轨迹，PPO 更新（GAE + clip + 价值网络）多 epoch
- 网络为 2 层 MLP（tanh），手写前向传播与反向传播（解析梯度）
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
    """两层 MLP 前向：x -> tanh(xW1+b1) -> W2 + b2。返回 (隐层, 输出)。"""
    h = np.tanh(x @ w1 + b1)
    out = h @ w2 + b2
    return h, out


class _PPONetworks:
    """PPO 策略网络（高斯均值）与价值网络，带手写反向传播。"""

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
        """采样动作与对数概率（不含常数项，与 PPO 目标一致）。"""
        _, mean = _mlp_forward(s, self.w1, self.b1, self.w2, self.b2)
        std = np.exp(self.log_std)
        a = mean + std * rng.normal(size=mean.shape)
        logp = -0.5 * np.sum((a - mean) ** 2 / std**2 + 2 * self.log_std)
        return a, float(logp)

    def value(self, s: np.ndarray) -> float:
        """价值估计（标量）。"""
        _, out = _mlp_forward(s, self.vw1, self.vb1, self.vw2, self.vb2)
        return float(out[0])

    def policy_and_value(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量前向：返回 (均值, 价值, 隐层激活)。"""
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
    """PPO 裁剪目标 L = min(r*A, clip(r)*A) 的解析梯度。

    Returns:
        (gw1, gb1, gw2, gb2, g_logstd)：策略网络各参数梯度
    """
    std = np.exp(net.log_std)
    inv_std = 1.0 / std
    dz = (actions - mean) * inv_std  # (n, d)
    # log pi(a|s) = -0.5*sum(dz^2) - sum(log_std)（与 act 一致，无常数项）
    logp_new = -0.5 * np.sum(dz**2, axis=1) - np.sum(net.log_std)
    ratio = np.exp(logp_new - np.array(logp_old))
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    # min 分支：adv*ratio <= adv*clipped 时取未裁剪项
    mask = (advantages * ratio <= advantages * clipped).astype(float)
    coef = mask * advantages * ratio  # (n,)
    n = len(states)
    # d logp / d mean = dz * inv_std；d logp / d log_std = dz^2 - 1（对轨迹长度平均）
    g_mean = coef[:, None] * (dz * inv_std) / n
    g_logstd = np.sum(coef[:, None] * (dz**2 - 1.0), axis=0) / n
    # 反向传播到两层 MLP
    dh2 = g_mean @ net.w2.T
    dh = dh2 * (1 - hidden**2)
    gw2 = hidden.T @ g_mean
    gb2 = np.sum(g_mean, axis=0)
    gw1 = states.T @ dh
    gb1 = np.sum(dh, axis=0)
    return gw1, gb1, gw2, gb2, g_logstd


class PPOTool(OptimizationTool):
    """PPO 强化学习优化器：策略网络学习参数更新方向。

    超参（可通过策略基因进化）：
    - ppo_lr: 学习率
    - ppo_clip: PPO 裁剪系数
    - ppo_gamma: GAE 折扣因子
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
            """归一化状态：参数 ∈ [-1, 1]，适应度经 tanh 压缩。"""
            x_norm = (v - low) / span * 2.0 - 1.0
            return np.concatenate([x_norm, [float(np.tanh(fv / 10.0))]])

        while n_evals < budget:
            # ---- rollout：收集一条轨迹（动作空间归一化，奖励截断防爆炸） ----
            states, actions, rewards, values, logps, dones = [], [], [], [], [], []
            s = _state(x, f)
            for _ in range(min(64, budget - n_evals)):
                a, logp = net.act(s, rng)
                x_new = np.clip(x + a * span * self.step_scale, low, high)
                f_new = float(problem.scalarize(x_new, weights))
                n_evals += 1
                # 相对改进幅度奖励：sign * min(1, |Δf| / (|f|+1))。
                # f 尺度跨多量级（如 Rosenbrock 百万级），相对形式不饱和
                # 且保留改进方向与幅度信息
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

            # ---- PPO 更新（多 epoch，小批量全量） ----
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
                # 价值网络（MSE，一步梯度）
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
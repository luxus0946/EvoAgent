"""Agent individual and evolvable strategy genome.

An individual is an evolvable optimization strategy: choice of initial tool, switch timing, early-stop patience, and tool hyperparameters.
An individual runs the strategy on the simulated environment to obtain fitness; population evolution drives self-improvement of the strategy.
"""

import uuid
from dataclasses import dataclass, field

import numpy as np

from evoagent.environment.fitness import normalize_weights
from evoagent.tools.base import TOOL_NAMES

# Parameter ranges for continuous genome fields (mutations are clamped to valid bounds)
_TOOL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "cma_sigma": (0.05, 0.5),
    "ga_mutation": (0.02, 0.4),
    "sa_t0": (0.005, 0.2),
    "sa_alpha": (0.9, 0.9999),
    "sa_sigma": (0.03, 0.3),
    "bo_xi": (0.0, 0.1),
    "ppo_lr": (0.001, 0.05),
    "ppo_clip": (0.05, 0.4),
    "ppo_gamma": (0.9, 0.999),
}


@dataclass
class StrategyGenome:
    """Evolvable strategy genome."""

    initial_tool: str = "cma_es"
    second_tool: str = "bo"
    switch_after_ratio: float = 0.5
    stop_patience: float = 0.3
    tool_params: dict[str, float] = field(
        default_factory=lambda: {
            "cma_sigma": 0.25,
            "ga_mutation": 0.15,
            "sa_t0": 0.05,
            "sa_alpha": 0.995,
            "sa_sigma": 0.1,
            "bo_xi": 0.01,
            "ppo_lr": 0.01,
            "ppo_clip": 0.2,
            "ppo_gamma": 0.99,
        }
    )
    weights: np.ndarray | None = None

    def clone(self) -> "StrategyGenome":
        """Deep-copy the genome."""
        return StrategyGenome(
            initial_tool=self.initial_tool,
            second_tool=self.second_tool,
            switch_after_ratio=self.switch_after_ratio,
            stop_patience=self.stop_patience,
            tool_params=dict(self.tool_params),
            weights=None if self.weights is None else self.weights.copy(),
        )


@dataclass
class AgentIndividual:
    """Agent individual."""

    agent_id: str
    genome: StrategyGenome
    fitness: float | None = None
    objectives: np.ndarray | None = None
    pareto_rank: int | None = None
    crowding: float | None = None
    best_params: np.ndarray | None = None
    n_evals: int = 0
    n_improvements: int = 0
    genome_prompt: object | None = None
    mode: str = "prompt"

    def clone(self) -> "AgentIndividual":
        """Deep-copy the individual (including evaluation result fields, so fitness is preserved for migration and similar scenarios)."""
        clone_prompt = (
            None if self.genome_prompt is None else self.genome_prompt.clone()
        )
        return AgentIndividual(
            agent_id=self.agent_id,
            genome=self.genome.clone(),
            fitness=self.fitness,
            objectives=None if self.objectives is None else self.objectives.copy(),
            pareto_rank=self.pareto_rank,
            crowding=self.crowding,
            best_params=None if self.best_params is None else self.best_params.copy(),
            n_evals=self.n_evals,
            n_improvements=self.n_improvements,
            genome_prompt=clone_prompt,
            mode=self.mode,
        )


def random_genome(rng: np.random.Generator, n_objectives: int = 1) -> StrategyGenome:
    """Generate a random strategy genome.

    Args:
        rng: random number generator
        n_objectives: number of objectives (a random weight vector is generated when > 1)

    Returns:
        a random strategy genome
    """
    tools = rng.choice(TOOL_NAMES, size=2, replace=True)
    tool_params = {
        key: float(rng.uniform(lo, hi))
        for key, (lo, hi) in _TOOL_PARAM_RANGES.items()
    }
    weights = None
    if n_objectives > 1:
        w = rng.dirichlet(np.ones(n_objectives))
        weights = normalize_weights(w)
    return StrategyGenome(
        initial_tool=str(tools[0]),
        second_tool=str(tools[1]),
        switch_after_ratio=float(rng.uniform(0.05, 0.95)),
        stop_patience=float(rng.uniform(0.05, 0.5)),
        tool_params=tool_params,
        weights=weights,
    )


def random_individual(
    rng: np.random.Generator, n_objectives: int = 1
) -> AgentIndividual:
    """Generate a random agent individual."""
    return AgentIndividual(
        agent_id=str(uuid.uuid4())[:8],
        genome=random_genome(rng, n_objectives),
    )


def make_rng_derived(agent_id: str, seed: int) -> np.random.Generator:
    """Derive an independent RNG from the individual ID (ensures reproducible strategy evaluation)."""
    return np.random.default_rng(int(seed) + sum(ord(c) for c in agent_id))

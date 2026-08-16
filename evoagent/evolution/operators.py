"""Evolutionary operators: selection / crossover / mutation (applied to agent strategy genomes)."""

import numpy as np

from evoagent.core.individual import AgentIndividual, StrategyGenome
from evoagent.tools.base import TOOL_NAMES

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

_CONTINUOUS_FIELDS = ["switch_after_ratio", "stop_patience"]


# ---------------------------------------------------------------- selection operators


def tournament_selection(
    population: list[AgentIndividual],
    k: int,
    tournament_size: int = 3,
    rng: np.random.Generator | None = None,
) -> list[AgentIndividual]:
    """Tournament selection: pick k individuals with high fitness from the population.

    Args:
        population: population with evaluated fitness
        k: number of individuals to select
        tournament_size: tournament size
        rng: random number generator

    Returns:
        list of selected individuals (with replacement)
    """
    if rng is None:
        rng = np.random.default_rng()
    selected: list[AgentIndividual] = []
    for _ in range(k):
        candidates = rng.choice(len(population), tournament_size, replace=False)
        best = max(candidates, key=lambda i: population[i].fitness)
        selected.append(population[int(best)])
    return selected


def roulette_selection(
    population: list[AgentIndividual],
    k: int,
    rng: np.random.Generator | None = None,
) -> list[AgentIndividual]:
    """Roulette wheel selection: selection probability proportional to fitness."""
    if rng is None:
        rng = np.random.default_rng()
    fitness = np.array([ind.fitness for ind in population], dtype=float)
    prob = fitness - fitness.min() + 1e-12
    if prob.sum() <= 0:
        prob = np.ones_like(prob)
    prob /= prob.sum()
    idx = rng.choice(len(population), size=k, p=prob)
    return [population[i] for i in idx]


def rank_selection(
    population: list[AgentIndividual],
    k: int,
    rng: np.random.Generator | None = None,
) -> list[AgentIndividual]:
    """Rank selection: linear probabilities assigned by fitness rank."""
    if rng is None:
        rng = np.random.default_rng()
    order = np.argsort([ind.fitness for ind in population])
    n = len(population)
    prob = np.arange(1, n + 1, dtype=float)
    prob /= prob.sum()
    idx = rng.choice(order, size=k, p=prob)
    return [population[i] for i in idx]


# ---------------------------------------------------------------- crossover operators


def crossover_uniform(
    parent1: AgentIndividual,
    parent2: AgentIndividual,
    probability: float = 0.5,
    rng: np.random.Generator | None = None,
) -> AgentIndividual:
    """Uniform crossover: genome fields swapped by probability; continuous parameters blended arithmetically.

    Args:
        parent1: parent 1
        parent2: parent 2
        probability: swap probability for discrete fields
        rng: random number generator

    Returns:
        offspring individual
    """
    if rng is None:
        rng = np.random.default_rng()
    g1, g2 = parent1.genome, parent2.genome
    child = AgentIndividual(agent_id=parent1.agent_id, genome=g1.clone())

    if rng.random() < probability:
        child.genome.initial_tool = g2.initial_tool
    if rng.random() < probability:
        child.genome.second_tool = g2.second_tool
    for f in _CONTINUOUS_FIELDS:
        if rng.random() < 0.5:
            setattr(child.genome, f, float(getattr(g2, f)))
        else:
            setattr(child.genome, f, (getattr(g1, f) + getattr(g2, f)) / 2.0)
    for key in child.genome.tool_params:
        if rng.random() < 0.5:
            child.genome.tool_params[key] = float(
                0.5 * (g1.tool_params[key] + g2.tool_params[key])
            )
        elif rng.random() < 0.5:
            child.genome.tool_params[key] = g2.tool_params[key]
    if g1.weights is not None and g2.weights is not None:
        child.genome.weights = normalize_weights_after_mix(g1.weights, g2.weights, rng)
    return child


def crossover_arithmetic(
    parent1: AgentIndividual,
    parent2: AgentIndividual,
    alpha: float = 0.5,
) -> AgentIndividual:
    """Arithmetic crossover: linear combination of continuous fields."""
    g1, g2 = parent1.genome, parent2.genome
    child = AgentIndividual(agent_id=parent1.agent_id, genome=g1.clone())
    for f in _CONTINUOUS_FIELDS:
        setattr(
            child.genome,
            f,
            float(alpha * getattr(g1, f) + (1 - alpha) * getattr(g2, f)),
        )
    for key in child.genome.tool_params:
        child.genome.tool_params[key] = float(
            alpha * g1.tool_params[key] + (1 - alpha) * g2.tool_params[key]
        )
    return child


def normalize_weights_after_mix(
    w1: np.ndarray, w2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Mix two parent weight vectors and normalize."""
    if rng.random() < 0.5:
        mixed = 0.5 * (w1 + w2)
    else:
        mixed = w1 if rng.random() < 0.5 else w2
    total = mixed.sum()
    return mixed / total if total > 0 else np.full_like(mixed, 1.0 / len(mixed))


# ---------------------------------------------------------------- mutation operators


def mutate_genome(
    genome: StrategyGenome,
    rate: float,
    rng: np.random.Generator | None = None,
) -> StrategyGenome:
    """Gaussian/discrete mixed mutation: mutate each genome field with probability rate.

    Args:
        genome: genome to mutate
        rate: mutation rate
        rng: random number generator

    Returns:
        the mutated genome (same object; the caller is responsible for cloning)
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < rate:
        genome.initial_tool = str(rng.choice([t for t in TOOL_NAMES if t != genome.initial_tool]))
    if rng.random() < rate:
        genome.second_tool = str(rng.choice([t for t in TOOL_NAMES if t != genome.second_tool]))
    for f in _CONTINUOUS_FIELDS:
        if rng.random() < rate:
            lo, hi = _field_range(f)
            value = float(getattr(genome, f))
            setattr(genome, f, float(np.clip(value + rng.normal(0.0, 0.1), lo, hi)))
    for key, (lo, hi) in _TOOL_PARAM_RANGES.items():
        if rng.random() < rate:
            value = genome.tool_params[key]
            genome.tool_params[key] = float(np.clip(value + rng.normal(0.0, 0.1 * (hi - lo)), lo, hi))
    if genome.weights is not None and rng.random() < rate:
        perturb = rng.normal(0.0, 0.1, len(genome.weights))
        genome.weights = normalize_weights_after_mix(genome.weights + perturb, genome.weights, rng)
    return genome


def mutate_individual(
    individual: AgentIndividual,
    rate: float,
    rng: np.random.Generator | None = None,
) -> AgentIndividual:
    """Mutate the individual's genome and return a clone."""
    if rng is None:
        rng = np.random.default_rng()
    mutant = individual.clone()
    mutate_genome(mutant.genome, rate, rng)
    return mutant


# ---------------------------------------------------------------- EoH operator-style mutation

_E1_TOOLS = ["random_search", "sa", "bo"]


def mutate_genome_eoh(
    genome: StrategyGenome,
    rate: float,
    rng: np.random.Generator | None = None,
    exploit_ratio: float = 0.7,
) -> StrategyGenome:
    """EoH-style operatorized mutation (inspired by the operator design in FeiLiu36/EoH evolution.py).

    Exploration operators (structural changes):
    - e1: coarse-grained operator replacement — swap in a different tool
    - e2: operator combination/swap — swap the tool order
    Exploitation operators (parameter fine-tuning):
    - m1: fine-tuning — small-step Gaussian perturbation of a single continuous parameter
    - m2: coordinated perturbation — perturb all continuous parameters along the same direction
    - m3: weight fine-tuning — small perturbation of objective weights followed by normalization

    Args:
        genome: genome to mutate (in place)
        rate: mutation trigger probability
        rng: random number generator
        exploit_ratio: share of exploitation operators (EoH favors fine-tuning)

    Returns:
        the mutated genome
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() >= rate:
        return genome
    if rng.random() < exploit_ratio:
        operator = str(rng.choice(["m1", "m2", "m3"], p=[0.5, 0.3, 0.2]))
    else:
        operator = str(rng.choice(["e1", "e2"]))
    _apply_eoh_operator(genome, operator, rng)
    return genome


def _apply_eoh_operator(
    genome: StrategyGenome, operator: str, rng: np.random.Generator
) -> None:
    """Apply a single EoH mutation operator."""
    if operator == "e1":
        target = "initial_tool" if rng.random() < 0.5 else "second_tool"
        setattr(genome, target, str(rng.choice(_E1_TOOLS)))
    elif operator == "e2":
        genome.initial_tool, genome.second_tool = (
            genome.second_tool,
            genome.initial_tool,
        )
    elif operator == "m1":
        fields = _CONTINUOUS_FIELDS + list(genome.tool_params.keys())
        field = str(rng.choice(fields))
        if field in _CONTINUOUS_FIELDS:
            lo, hi = _field_range(field)
            value = float(getattr(genome, field))
            setattr(genome, field, float(np.clip(value + rng.normal(0.0, 0.05), lo, hi)))
        else:
            lo, hi = _TOOL_PARAM_RANGES[field]
            value = genome.tool_params[field]
            genome.tool_params[field] = float(
                np.clip(value + rng.normal(0.0, 0.05 * (hi - lo)), lo, hi)
            )
    elif operator == "m2":
        direction = float(rng.choice([-1.0, 1.0]))
        for f in _CONTINUOUS_FIELDS:
            lo, hi = _field_range(f)
            value = float(getattr(genome, f))
            setattr(
                genome,
                f,
                float(np.clip(value + direction * rng.normal(0.0, 0.03), lo, hi)),
            )
        for key, (lo, hi) in _TOOL_PARAM_RANGES.items():
            value = genome.tool_params[key]
            genome.tool_params[key] = float(
                np.clip(
                    value + direction * rng.normal(0.0, 0.03 * (hi - lo)), lo, hi
                )
            )
    elif operator == "m3":
        if genome.weights is not None:
            perturb = rng.normal(0.0, 0.05, len(genome.weights))
            genome.weights = normalize_weights_after_mix(
                genome.weights + perturb, genome.weights, rng
            )


def mutate_individual_eoh(
    individual: AgentIndividual,
    rate: float,
    rng: np.random.Generator | None = None,
) -> AgentIndividual:
    """Individual wrapper for EoH operator-style mutation (returns a clone)."""
    if rng is None:
        rng = np.random.default_rng()
    mutant = individual.clone()
    mutate_genome_eoh(mutant.genome, rate, rng)
    return mutant


def _field_range(field: str) -> tuple[float, float]:
    return (0.05, 0.95) if field == "switch_after_ratio" else (0.05, 0.5)

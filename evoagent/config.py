"""全局配置管理：所有可配置参数集中于此，支持环境变量覆盖。"""

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvolutionConfig:
    """进化框架配置（单岛参数，岛屿数大于 1 时启用岛屿模型）。"""

    population_size: int = 8
    max_generations: int = 10
    n_islands: int = 3
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    selection_pressure: float = 0.3
    elite_ratio: float = 0.1
    migration_interval: int = 3
    migration_rate: float = 0.2
    eval_budget_per_individual: int = 300
    multi_objective: bool = False
    n_objectives: int = 3
    fitness_weights: np.ndarray | None = None
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    """对比实验配置。"""

    total_budget: int = 800
    baseline_tools: list[str] = field(
        default_factory=lambda: ["random_search", "sa", "ga", "cma_es", "bo"]
    )
    n_seeds: int = 3
    problems: list[str] = field(
        default_factory=lambda: ["semiconductor", "rosenbrock", "ackley", "rastrigin"]
    )
    output_dir: str = "./data/results"


@dataclass
class AppConfig:
    """全局配置聚合。"""

    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    log_level: str = "INFO"


def from_env() -> AppConfig:
    """从环境变量加载配置（仅覆盖非默认值场景）。"""
    config = AppConfig()
    config.log_level = os.getenv("EVOAGENT_LOG_LEVEL", config.log_level)
    config.evolution.random_seed = int(
        os.getenv("EVOAGENT_SEED", str(config.evolution.random_seed))
    )
    return config

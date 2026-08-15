"""进化循环主逻辑：评估 -> 选择 -> 交叉 -> 变异 -> 迁移 -> 精英保留。"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evoagent.config import EvolutionConfig
from evoagent.environment.fitness import (
    hypervolume_2d,
    non_dominated_front,
    reference_point,
)
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.island_model import IslandModel

logger = logging.getLogger("evoagent.evolution")


@dataclass
class GenerationRecord:
    """每代记录。"""

    generation: int
    best_fitness: float
    mean_fitness: float
    diversity: float
    n_evals: int
    elapsed_time: float
    best_params: np.ndarray | None = None


@dataclass
class EvolutionResult:
    """进化实验结果。"""

    best_individual: object
    best_fitness: float
    best_params: np.ndarray
    generation_history: list[GenerationRecord] = field(default_factory=list)
    total_evals: int = 0
    elapsed_time: float = 0.0
    archive: np.ndarray | None = None
    archive_history: list[np.ndarray] = field(default_factory=list)
    pareto_front: np.ndarray | None = None
    hypervolume_history: list[float] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def run_evolution(
    problem: OptimizationProblem,
    config: EvolutionConfig,
    checkpoint_path: str | None = None,
    resume: bool = False,
) -> EvolutionResult:
    """运行一次完整的进化实验。

    Args:
        problem: 优化问题
        config: 进化配置
        checkpoint_path: 检查点文件路径（非 None 时每代保存）
        resume: 是否从检查点断点续跑（需 checkpoint_path 存在）

    Returns:
        EvolutionResult：最优个体、历史记录、Pareto 档案等
    """
    start = time.perf_counter()
    model = _build_model(problem, config)

    archive: list[np.ndarray] = []
    archive_history: list[np.ndarray] = []
    hv_history: list[float] = []
    total_evals = 0
    history: list[GenerationRecord] = []

    resume_started = False
    if resume and checkpoint_path and Path(checkpoint_path).exists():
        from evoagent.evolution.checkpoint import load_checkpoint, restore_model

        ckpt = load_checkpoint(checkpoint_path)
        model, state = restore_model(ckpt, problem, config.multi_objective)
        archive = state["archive"]
        archive_history = state["archive_history"]
        hv_history = state["hv_history"]
        total_evals = state["total_evals"]
        history = [
            GenerationRecord(
                generation=rec["generation"],
                best_fitness=rec["best_fitness"],
                mean_fitness=rec["mean_fitness"],
                diversity=rec["diversity"],
                n_evals=rec["n_evals"],
                elapsed_time=rec["elapsed_time"],
                best_params=(
                    None
                    if rec.get("best_params") is None
                    else np.array(rec["best_params"])
                ),
            )
            for rec in state["history"]
        ]
        resume_started = True
        logger.info("断点续跑: 从 generation=%d 继续", model.generation + 1)

    first_gen = model.generation + 1 if resume_started else model.generation
    for gen in range(first_gen, config.max_generations):
        if resume_started:
            model.next_generation()
            resume_started = False
        t0 = time.perf_counter()
        total_evals += model.evaluate_all()
        stats = model.stats()
        best = model.best_individual()

        if config.multi_objective:
            archive = _update_archive(
                archive,
                [ind.objectives for ind in _all_individuals(model)],
            )
            front = _final_front(archive)
            archive_history.append(front)
            if problem.n_objectives == 2:
                if len(front) > 0:
                    ref = reference_point(front)
                    hv_history.append(hypervolume_2d(front, ref))

        best_fitness = best.fitness
        mean_fitness = float(np.mean([s.mean_fitness for s in stats]))
        diversity = float(np.mean([s.diversity for s in stats]))
        t1 = time.perf_counter()

        history.append(
            GenerationRecord(
                generation=gen,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                diversity=diversity,
                n_evals=total_evals,
                elapsed_time=t1 - t0,
                best_params=best.best_params.copy(),
            )
        )
        if checkpoint_path:
            _save_checkpoint(
                checkpoint_path,
                model,
                config,
                total_evals,
                archive,
                archive_history,
                hv_history,
                history,
            )
        if gen < config.max_generations - 1:
            model.next_generation()

    best = model.best_individual()
    front = None
    if config.multi_objective:
        front = _final_front(archive)

    return EvolutionResult(
        best_individual=best,
        best_fitness=best.fitness,
        best_params=best.best_params,
        generation_history=history,
        total_evals=total_evals,
        elapsed_time=time.perf_counter() - start,
        archive=np.array(archive) if archive else None,
        archive_history=archive_history,
        pareto_front=front,
        hypervolume_history=hv_history,
        config=_config_to_dict(config),
    )


def _build_model(problem: OptimizationProblem, config: EvolutionConfig) -> IslandModel:
    """按配置构建岛屿模型。"""
    if config.n_islands > 1:
        names = ["explore", "balance", "exploit"][: config.n_islands]
    else:
        names = ["balance"]
    return IslandModel(
        problem=problem,
        island_names=names,
        population_size=config.population_size,
        seed=config.random_seed,
        migration_interval=config.migration_interval,
        migration_rate=config.migration_rate,
        multi_objective=config.multi_objective,
        eval_budget_per_individual=config.eval_budget_per_individual,
        fitness_weights=config.fitness_weights,
    )


def _save_checkpoint(
    checkpoint_path: str,
    model: IslandModel,
    config: EvolutionConfig,
    total_evals: int,
    archive: list[np.ndarray],
    archive_history: list[np.ndarray],
    hv_history: list[float],
    history: list[GenerationRecord],
) -> None:
    """序列化并保存检查点。"""
    from evoagent.evolution.checkpoint import save_checkpoint

    history_dicts = [
        {
            "generation": rec.generation,
            "best_fitness": rec.best_fitness,
            "mean_fitness": rec.mean_fitness,
            "diversity": rec.diversity,
            "n_evals": rec.n_evals,
            "elapsed_time": rec.elapsed_time,
            "best_params": (
                None if rec.best_params is None else rec.best_params.tolist()
            ),
        }
        for rec in history
    ]
    save_checkpoint(
        checkpoint_path,
        model,
        {
            "config": _config_to_dict(config),
            "total_evals": total_evals,
            "archive": archive,
            "archive_history": archive_history,
            "hv_history": hv_history,
            "history": history_dicts,
        },
    )


def _all_individuals(model: IslandModel) -> list:
    """收集所有岛的个体。"""
    return [ind for island in model.islands for ind in island.individuals]


def _update_archive(archive: list[np.ndarray], objectives: list) -> list[np.ndarray]:
    """将新目标向量并入全局非支配档案。"""
    merged = list(archive)
    merged.extend([np.asarray(o, dtype=float) for o in objectives])
    arr = np.array(merged)
    keep = non_dominated_front(arr)
    return [arr[i] for i in keep]


def _final_front(archive: list[np.ndarray]) -> np.ndarray:
    """最终 Pareto 前沿（无噪声目标）。"""
    if not archive:
        return np.empty((0, 0))
    arr = np.array(archive)
    return arr[non_dominated_front(arr)]


def _config_to_dict(config: EvolutionConfig) -> dict:
    """将配置转为可序列化字典。"""
    return {
        "population_size": config.population_size,
        "max_generations": config.max_generations,
        "n_islands": config.n_islands,
        "mutation_rate": config.mutation_rate,
        "crossover_rate": config.crossover_rate,
        "selection_pressure": config.selection_pressure,
        "elite_ratio": config.elite_ratio,
        "migration_interval": config.migration_interval,
        "migration_rate": config.migration_rate,
        "eval_budget_per_individual": config.eval_budget_per_individual,
        "multi_objective": config.multi_objective,
        "n_objectives": config.n_objectives,
        "fitness_weights": (
            None if config.fitness_weights is None else config.fitness_weights.tolist()
        ),
        "random_seed": config.random_seed,
    }

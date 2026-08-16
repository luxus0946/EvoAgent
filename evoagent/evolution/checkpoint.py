"""Checkpointing and resume (inspired by the full-state checkpoint of OpenEvolve controller.py).

Saved contents:
- per-island population individuals (genome/fitness/best params/objective vectors) + per-island RNG state
- generation counter, total evaluations, Pareto archive, per-generation history
- experiment config (to validate consistency when resuming)

After restoration the RNG state is identical, so a resumed run matches a continuous run exactly (verifiable).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evoagent.core.individual import StrategyGenome
from evoagent.evolution.island_model import IslandModel
from evoagent.evolution.population import Population

logger = logging.getLogger("evoagent.checkpoint")

FORMAT_VERSION = 1


@dataclass
class EvolutionCheckpoint:
    """Evolution experiment checkpoint."""

    config: dict
    generation: int
    total_evals: int
    islands: list[dict]
    archive: list[list[float]]
    archive_history: list[list[list[float]]]
    hv_history: list[float]
    history: list[dict]
    rng_state: dict | None = None


def save_checkpoint(path: str | Path, model: IslandModel, state: dict) -> None:
    """Save a checkpoint to a JSON file.

    Args:
        path: checkpoint file path
        model: island model (used to serialize each island's population and RNG state)
        state: experiment state (config/archive/history/hv_history/total_evals, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = EvolutionCheckpoint(
        config=state["config"],
        generation=model.generation,
        total_evals=state["total_evals"],
        islands=[_serialize_island(island) for island in model.islands],
        archive=[_to_list(v) for v in state.get("archive", [])],
        archive_history=[
            [_to_list(v) for v in front] for front in state.get("archive_history", [])
        ],
        hv_history=[float(v) for v in state.get("hv_history", [])],
        history=state.get("history", []),
        rng_state=_rng_state_dict(model),
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(_dataclass_to_dict(checkpoint), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    tmp.replace(path)
    logger.info("检查点已保存: %s (generation=%d)", path, model.generation)


def load_checkpoint(path: str | Path) -> EvolutionCheckpoint:
    """Load a checkpoint.

    Args:
        path: checkpoint file path

    Returns:
        EvolutionCheckpoint

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the version is incompatible
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"检查点不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"检查点版本不兼容: {data.get('format_version')} != {FORMAT_VERSION}"
        )
    return EvolutionCheckpoint(**{k: v for k, v in data.items() if k != "format_version"})


def restore_model(
    checkpoint: EvolutionCheckpoint,
    problem,
    multi_objective: bool,
) -> tuple[IslandModel, dict]:
    """Restore the island model and experiment state from a checkpoint.

    Args:
        checkpoint: checkpoint
        problem: optimization problem
        multi_objective: multi-objective mode (for validation)

    Returns:
        (model, state): restored model and state dict

    Raises:
        ValueError: if the config is inconsistent
    """
    cfg = checkpoint.config
    if cfg.get("multi_objective") != multi_objective:
        raise ValueError("检查点 multi_objective 与当前配置不一致")

    model = IslandModel(
        problem=problem,
        island_names=[isl["name"] for isl in checkpoint.islands],
        population_size=cfg["population_size"],
        seed=cfg["random_seed"],
        migration_interval=cfg["migration_interval"],
        migration_rate=cfg["migration_rate"],
        multi_objective=cfg["multi_objective"],
        eval_budget_per_individual=cfg["eval_budget_per_individual"],
        fitness_weights=(
            None
            if cfg.get("fitness_weights") is None
            else np.array(cfg["fitness_weights"])
        ),
    )
    model.generation = checkpoint.generation
    for isl_data, island in zip(checkpoint.islands, model.islands):
        _restore_island(island, isl_data, problem.n_objectives)

    if checkpoint.rng_state:
        _restore_rng_state(model, checkpoint.rng_state)

    state = {
        "config": checkpoint.config,
        "total_evals": checkpoint.total_evals,
        "archive": [np.array(v) for v in checkpoint.archive],
        "archive_history": [
            np.array(front) if front else np.empty((0, 0))
            for front in checkpoint.archive_history
        ],
        "hv_history": list(checkpoint.hv_history),
        "history": list(checkpoint.history),
    }
    logger.info("检查点已恢复: generation=%d", checkpoint.generation)
    return model, state


def _serialize_island(island: Population) -> dict:
    map_elites = {}
    if island.map_elites is not None:
        map_elites = {
            f"{a},{b}": _serialize_individual(ind)
            for (a, b), ind in island.map_elites.grid.items()
        }
    return {
        "name": getattr(island, "name", ""),
        "generation": island.generation,
        "rng_state": island.rng.bit_generator.state,
        "individuals": [_serialize_individual(ind) for ind in island.individuals],
        "map_elites": map_elites,
    }


def _serialize_individual(ind) -> dict:
    genome = ind.genome
    return {
        "agent_id": ind.agent_id,
        "genome": {
            "initial_tool": genome.initial_tool,
            "second_tool": genome.second_tool,
            "switch_after_ratio": genome.switch_after_ratio,
            "stop_patience": genome.stop_patience,
            "tool_params": dict(genome.tool_params),
            "weights": _to_list(genome.weights),
        },
        "fitness": ind.fitness,
        "objectives": _to_list(ind.objectives),
        "pareto_rank": ind.pareto_rank,
        "crowding": ind.crowding,
        "best_params": _to_list(ind.best_params),
        "n_evals": ind.n_evals,
        "n_improvements": ind.n_improvements,
    }


def _restore_island(island: Population, data: dict, n_objectives: int) -> None:
    island.name = data.get("name", island.name)
    island.generation = data.get("generation", island.generation)
    island.rng = np.random.default_rng(0)
    island.rng.bit_generator.state = data["rng_state"]
    island.individuals = [
        _restore_individual(rec) for rec in data["individuals"]
    ]
    if island.map_elites is not None:
        for key, rec in (data.get("map_elites") or {}).items():
            a, b = key.split(",")
            island.map_elites.grid[(int(a), int(b))] = _restore_individual(rec)


def _restore_individual(rec: dict):
    from evoagent.core.individual import AgentIndividual

    genome = StrategyGenome(
        initial_tool=rec["genome"]["initial_tool"],
        second_tool=rec["genome"]["second_tool"],
        switch_after_ratio=rec["genome"]["switch_after_ratio"],
        stop_patience=rec["genome"]["stop_patience"],
        tool_params=dict(rec["genome"]["tool_params"]),
        weights=_from_list(rec["genome"]["weights"]),
    )
    return AgentIndividual(
        agent_id=rec["agent_id"],
        genome=genome,
        fitness=rec["fitness"],
        objectives=_from_list(rec["objectives"]),
        pareto_rank=rec["pareto_rank"],
        crowding=rec["crowding"],
        best_params=_from_list(rec["best_params"]),
        n_evals=rec["n_evals"],
        n_improvements=rec["n_improvements"],
    )


def _rng_state_dict(model: IslandModel) -> dict:
    return {isl.name: isl.rng.bit_generator.state for isl in model.islands}


def _restore_rng_state(model: IslandModel, state: dict) -> None:
    for isl in model.islands:
        if isl.name in state:
            isl.rng = np.random.default_rng(0)
            isl.rng.bit_generator.state = state[isl.name]


def _to_list(value) -> list | None:
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.tolist()


def _from_list(value):
    return None if value is None else np.array(value, dtype=float)


def _dataclass_to_dict(obj: EvolutionCheckpoint) -> dict:
    d = {
        "format_version": FORMAT_VERSION,
        "config": obj.config,
        "generation": obj.generation,
        "total_evals": obj.total_evals,
        "islands": obj.islands,
        "archive": obj.archive,
        "archive_history": obj.archive_history,
        "hv_history": obj.hv_history,
        "history": obj.history,
        "rng_state": obj.rng_state,
    }
    return d
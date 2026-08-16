"""EvoAgent REST API：进化实验任务队列服务。

- POST /api/evolve: 提交进化实验（后台任务执行），返回 task_id
- GET  /api/tasks/{task_id}: 查询任务状态与结果
- GET  /api/problems: 可用问题列表
- GET  /api/health: 健康检查

启动: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoagent.config import EvolutionConfig
from evoagent.environment.benchmarks import BENCHMARK_REGISTRY
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.evolution.evolutionary_loop import run_evolution

logger = logging.getLogger("evoagent.api")

PROBLEM_REGISTRY = {
    "semiconductor": SemiconductorSimulator,
    **{name: cls for name, cls in BENCHMARK_REGISTRY.items()},
}

app = FastAPI(
    title="EvoAgent API",
    description="进化 + LLM Agent 自动优化框架 REST 接口",
    version="1.0.0",
)

# 内存任务表：task_id -> {status, result, error, created_at, finished_at}
TASKS: dict[str, dict[str, Any]] = {}


class EvolveRequest(BaseModel):
    """进化实验请求体。"""

    problem: str = Field("semiconductor", description="问题名")
    generations: int = Field(5, ge=1, le=100)
    population: int = Field(6, ge=2, le=100)
    islands: int = Field(3, ge=1, le=8)
    budget: int = Field(200, ge=10, le=5000)
    seed: int = Field(42)
    multi_objective: bool = Field(False)


class TaskOut(BaseModel):
    """任务查询响应。"""

    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float
    finished_at: float | None = None


def _run_experiment(request: EvolveRequest) -> dict[str, Any]:
    """执行进化实验并序列化结果。"""
    problem_cls = PROBLEM_REGISTRY.get(request.problem)
    if problem_cls is None:
        raise ValueError(f"未知问题: {request.problem}")

    problem = problem_cls()
    weights = (
        np.array([0.5, 0.3, 0.2]) if request.problem == "semiconductor" else None
    )
    config = EvolutionConfig(
        population_size=request.population,
        max_generations=request.generations,
        n_islands=request.islands,
        eval_budget_per_individual=request.budget,
        multi_objective=request.multi_objective,
        n_objectives=problem.n_objectives,
        fitness_weights=weights,
        random_seed=request.seed,
    )
    result = run_evolution(problem, config)
    return {
        "problem": request.problem,
        "best_fitness": result.best_fitness,
        "clean_fitness": problem.scalarize_clean(result.best_params, weights),
        "best_params": [float(v) for v in result.best_params],
        "total_evals": result.total_evals,
        "elapsed_time": result.elapsed_time,
        "generation_history": [
            {
                "generation": g.generation,
                "best_fitness": g.best_fitness,
                "mean_fitness": g.mean_fitness,
                "diversity": g.diversity,
            }
            for g in result.generation_history
        ],
    }


def _execute_task(task_id: str, request: EvolveRequest) -> None:
    """后台任务：执行实验并写回任务表。"""
    task = TASKS[task_id]
    try:
        task["result"] = _run_experiment(request)
        task["status"] = "completed"
    except Exception as e:  # noqa: BLE001 - 任务错误需回写
        logger.exception("任务 %s 失败", task_id)
        task["status"] = "failed"
        task["error"] = str(e)
    finally:
        task["finished_at"] = time.time()


@app.get("/api/health")
def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@app.get("/api/problems")
def problems() -> dict[str, list[str]]:
    """可用问题列表。"""
    return {"problems": sorted(PROBLEM_REGISTRY.keys())}


@app.post("/api/evolve", response_model=TaskOut, status_code=202)
def submit_evolve(
    request: EvolveRequest, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """提交进化实验任务。"""
    task_id = uuid.uuid4().hex[:12]
    TASKS[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "finished_at": None,
    }
    background_tasks.add_task(_execute_task, task_id, request)
    return TASKS[task_id]


@app.get("/api/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str) -> dict[str, Any]:
    """查询任务状态与结果。"""
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return task


def main() -> None:
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
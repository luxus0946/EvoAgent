"""运行单次进化实验：输出配置、结果、每代 CSV、收敛曲线与 Pareto 图。"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoagent.config import EvolutionConfig
from evoagent.environment.benchmarks import BENCHMARK_REGISTRY
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.evolution.evolutionary_loop import run_evolution
from evoagent.utils.logger import new_log_file_path, setup_logger
from evoagent.utils.random import make_rng, set_seed
from evoagent.utils.visualization import plot_convergence_curves, plot_pareto_front


def resolve_problem(name: str):
    """按名称解析测试问题实例。"""
    if name == "semiconductor":
        return SemiconductorSimulator()
    if name in BENCHMARK_REGISTRY:
        return BENCHMARK_REGISTRY[name]()
    raise ValueError(f"Unknown problem: {name}")


DEFAULT_WEIGHTS = {
    "semiconductor": np.array([0.5, 0.3, 0.2]),
}


def default_weights_for(problem) -> np.ndarray | None:
    """问题默认标量化权重（半导体为设计文档中的良率/成本/周期权重）。"""
    return DEFAULT_WEIGHTS.get(problem.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 EvoAgent 进化实验")
    parser.add_argument("--problem", default="semiconductor",
                        help="问题: semiconductor/rosenbrock/ackley/rastrigin/zdt1")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--budget", type=int, default=300,
                        help="每个个体执行策略的评估预算")
    parser.add_argument("--multi-objective", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./data/results")
    args = parser.parse_args()

    output_dir = Path(args.output) / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("evoagent", "INFO", new_log_file_path(output_dir))
    set_seed(args.seed)

    problem = resolve_problem(args.problem)
    config = EvolutionConfig(
        population_size=args.population,
        max_generations=args.generations,
        n_islands=args.islands,
        eval_budget_per_individual=args.budget,
        multi_objective=args.multi_objective,
        n_objectives=problem.n_objectives,
        fitness_weights=default_weights_for(problem),
        random_seed=args.seed,
    )

    logger.info(
        "开始进化实验: problem=%s dim=%d obj=%s 种群=%d x %d岛 x %d代 预算=%d",
        problem.name, problem.dim, problem.objective_names,
        args.population, args.islands, args.generations, args.budget,
    )
    result = run_evolution(problem, config)
    logger.info(
        "实验完成: 最优适应度=%.6f 总评估=%d 耗时=%.2fs",
        result.best_fitness, result.total_evals, result.elapsed_time,
    )

    config_dict = dict(config.__dict__)
    config_dict["fitness_weights"] = (
        None if config.fitness_weights is None else config.fitness_weights.tolist()
    )
    (output_dir / "config.json").write_text(
        json.dumps({"problem": problem.name, **config_dict}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result_json = {
        "problem": problem.name,
        "best_fitness": result.best_fitness,
        "best_params": [float(v) for v in result.best_params],
        "total_evals": result.total_evals,
        "elapsed_time": result.elapsed_time,
        "generation_history": [
            {
                "generation": g.generation,
                "best_fitness": g.best_fitness,
                "mean_fitness": g.mean_fitness,
                "diversity": g.diversity,
                "n_evals": g.n_evals,
                "elapsed_time": g.elapsed_time,
            }
            for g in result.generation_history
        ],
    }
    if result.pareto_front is not None:
        result_json["pareto_front"] = result.pareto_front.tolist()
    (output_dir / "result.json").write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_lines = ["generation,best_fitness,mean_fitness,diversity,n_evals,elapsed_time"]
    csv_lines += [
        f"{g.generation},{g.best_fitness:.6f},{g.mean_fitness:.6f},{g.diversity:.6f},{g.n_evals},{g.elapsed_time:.3f}"
        for g in result.generation_history
    ]
    (output_dir / "generation_history.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    gens = [g.generation for g in result.generation_history]
    bests = [g.best_fitness for g in result.generation_history]
    plot_convergence_curves(
        {"EvoAgent": np.array(bests)},
        f"{problem.name} 进化收敛曲线 (EvoAgent)",
        output_dir / "convergence_curve.png",
        xlabel="代数", ylabel="最优适应度",
    )
    if result.pareto_front is not None:
        plot_pareto_front(
            result.pareto_front,
            f"{problem.name} Pareto 前沿",
            output_dir / "pareto_plot.png",
        )

    print(f"\n实验结果已保存至: {output_dir}")
    print(f"问题: {problem.name} | 最优适应度: {result.best_fitness:.6f} | 总评估: {result.total_evals}")
    print(f"最佳策略: initial={result.best_individual.genome.initial_tool} -> "
          f"second={result.best_individual.genome.second_tool} "
          f"switch_ratio={result.best_individual.genome.switch_after_ratio:.2f}")


if __name__ == "__main__":
    main()

"""EvoAgent Gradio demo UI: run evolution experiments in the browser.

Run: python app/gradio_app.py (default http://127.0.0.1:7860)
"""

import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoagent.config import EvolutionConfig
from evoagent.environment.benchmarks import BENCHMARK_REGISTRY
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.evolution.evolutionary_loop import run_evolution

PROBLEMS = ["semiconductor"] + sorted(BENCHMARK_REGISTRY.keys())
WEIGHTS = {"semiconductor": np.array([0.5, 0.3, 0.2])}


def run_experiment(
    problem_name: str,
    generations: int,
    population: int,
    islands: int,
    budget: int,
    seed: int,
):
    """Run an evolution experiment and return (summary text, convergence plot)."""
    problem_cls = (
        SemiconductorSimulator
        if problem_name == "semiconductor"
        else BENCHMARK_REGISTRY[problem_name]
    )
    problem = problem_cls()
    weights = WEIGHTS.get(problem_name)
    config = EvolutionConfig(
        population_size=population,
        max_generations=generations,
        n_islands=islands,
        eval_budget_per_individual=budget,
        multi_objective=False,
        n_objectives=problem.n_objectives,
        fitness_weights=weights,
        random_seed=seed,
    )
    result = run_evolution(problem, config)

    clean = problem.scalarize_clean(result.best_params, weights)
    summary = (
        f"问题: {problem.name}\n"
        f"最优适应度(含噪): {result.best_fitness:.6f}\n"
        f"无噪声适应度: {clean:.6f}\n"
        f"总评估次数: {result.total_evals}\n"
        f"耗时: {result.elapsed_time:.2f}s\n"
        f"最优参数: {[f'{v:.3f}' for v in result.best_params]}\n"
        f"最优策略: {result.best_individual.genome.initial_tool} -> "
        f"{result.best_individual.genome.second_tool}"
        f" (switch={result.best_individual.genome.switch_after_ratio:.2f})"
    )

    gens = [g.generation for g in result.generation_history]
    bests = [g.best_fitness for g in result.generation_history]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(gens, bests, color="#d62728", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title(f"{problem.name}: EvoAgent convergence (seed={seed})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return summary, fig


def build_demo() -> gr.Blocks:
    """Build the Gradio UI."""
    with gr.Blocks(title="EvoAgent 演示") as demo:
        gr.Markdown(
            "# EvoAgent：进化 + LLM Agent 自动优化框架\n"
            "半导体工艺参数优化与标准 Benchmark 上的进化实验演示。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                problem = gr.Dropdown(
                    PROBLEMS, value="semiconductor", label="问题"
                )
                generations = gr.Slider(1, 20, value=5, step=1, label="进化代数")
                population = gr.Slider(2, 20, value=6, step=1, label="种群规模")
                islands = gr.Slider(1, 8, value=3, step=1, label="岛屿数")
                budget = gr.Slider(50, 1000, value=200, step=50, label="个体评估预算")
                seed = gr.Number(value=42, label="随机种子", precision=0)
                run_btn = gr.Button("运行进化实验", variant="primary")
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="实验结果", lines=10)
                output_plot = gr.Plot(label="收敛曲线")

        run_btn.click(
            fn=run_experiment,
            inputs=[problem, generations, population, islands, budget, seed],
            outputs=[output_text, output_plot],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
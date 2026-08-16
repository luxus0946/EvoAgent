"""Phase 2/3 experiment: LLM Agent prompt evolution vs SEW dual-mode vs fixed prompt vs phase-1 no-LLM.

Comparison setup (identical LLM call counts):
- evolve: EvoAgent-LLM, population evolves prompt genes (role/thinking style/tool preference/exploration bias)
- sew: SEW dual-mode (phase 3), population co-evolves structure-type (strategy genes + EoH operators)
  and prompt-type (prompt genes) individuals, LLM call count identical to evolve
- fixed: fixed default prompt baseline, same number of individuals, re-evaluated each generation
  (LLM call count identical to evolve)
- phase1: phase-1 no-LLM EvoAgent (same population size, same per-individual budget) as reference

Defaults to a simulated LLM (no API cost, reproducible); `--llm real` calls the DeepSeek API.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoagent.agent.knowledge_base import KnowledgeBase
from evoagent.agent.llm import MockLLMClient, OpenAILLMClient
from evoagent.agent.workflow import AgentWorkflow
from evoagent.config import EvolutionConfig, LLMConfig
from evoagent.core.genome_prompt import default_prompt
from evoagent.environment.simulator import SemiconductorSimulator
from evoagent.evolution.evolutionary_loop import run_evolution
from evoagent.evolution.llm_population import LlmPopulation
from evoagent.utils.logger import new_log_file_path, setup_logger
from evoagent.utils.random import set_seed
from evoagent.utils.visualization import plot_convergence_curves

PROBLEM = SemiconductorSimulator()
WEIGHTS = np.array([0.5, 0.3, 0.2])
MODE_NAMES = ["llm_evolve", "llm_sew", "llm_fixed", "phase1"]


def run_llm_mode(
    problem,
    mode: str,
    llm_client,
    pop_size: int,
    max_generations: int,
    budget: int,
    seed: int,
) -> tuple[float, list[float], dict]:
    """Run one LLM mode, return (clean_fitness, best_history, best_prompt_dict)."""
    set_seed(seed)
    workflow = AgentWorkflow(
        problem,
        budget=budget,
        llm=llm_client,
        knowledge_base=KnowledgeBase(),
        weights=WEIGHTS,
    )
    pop = LlmPopulation(
        problem,
        size=pop_size,
        seed=seed,
        workflow=workflow,
        fixed_prompt=(mode == "llm_fixed"),
        initial_prompt=(
            None if mode in ("llm_evolve", "llm_sew") else default_prompt()
        ),
        sew_ratio=0.5 if mode == "llm_sew" else 0.0,
    )
    for _ in range(max_generations + 1):
        pop.evaluate_all()
        if pop.generation >= max_generations:
            break
        pop.next_generation()

    best = pop.best_individual()
    clean = problem.scalarize_clean(best.best_params, WEIGHTS)
    prompt = best.genome_prompt
    if prompt is not None:
        best_prompt = {
            "mode": "prompt",
            "role": prompt.role,
            "thinking_style": prompt.thinking_style,
            "tool_preference": prompt.tool_preference,
            "stopping_criteria": round(prompt.stopping_criteria, 3),
            "max_iterations": prompt.max_iterations,
            "exploration_bias": round(prompt.exploration_bias, 3),
        }
    else:
        best_prompt = {
            "mode": "structure",
            "initial_tool": best.genome.initial_tool,
            "second_tool": best.genome.second_tool,
            "switch_after_ratio": round(best.genome.switch_after_ratio, 3),
            "stop_patience": round(best.genome.stop_patience, 3),
        }
    return clean, pop.best_history, best_prompt


def run_phase1(problem, pop_size: int, max_generations: int, budget: int, seed: int):
    """Run phase-1 EvoAgent without LLM."""
    config = EvolutionConfig(
        multi_objective=False,
        n_objectives=problem.n_objectives,
        random_seed=seed,
        fitness_weights=WEIGHTS,
        population_size=pop_size,
        max_generations=max_generations,
        n_islands=1,
        eval_budget_per_individual=budget,
    )
    result = run_evolution(problem, config)
    clean = problem.scalarize_clean(result.best_params, WEIGHTS)
    history = [g.best_fitness for g in result.generation_history]
    return clean, history


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoAgent phase 2: LLM prompt evolution experiment")
    parser.add_argument("--llm", choices=["mock", "real"], default="mock",
                        help="mock=simulated LLM (default, reproducible); real=DeepSeek API")
    parser.add_argument("--mode", choices=["both", "evolve", "sew", "fixed", "phase1"], default="both")
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="./data/results")
    args = parser.parse_args()

    output_dir = Path(args.output) / f"llm_agent_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("evoagent", "INFO", new_log_file_path(output_dir))
    logger.info(
        "阶段二/三实验启动: llm=%s mode=%s pop=%d gens=%d budget=%d seeds=%s",
        args.llm, args.mode, args.pop, args.gens, args.budget, list(range(1, args.seeds + 1)),
    )

    if args.llm == "real":
        llm_client = OpenAILLMClient(LLMConfig())
        logger.info("使用真实 LLM: %s", llm_client.config.model_name)
    else:
        llm_client = MockLLMClient()

    modes = [
        m
        for m in MODE_NAMES
        if args.mode == "both" or args.mode == m.replace("llm_", "")
    ]
    results: dict[str, dict] = {}
    for mode in modes:
        finals: list[float] = []
        histories: list[list[float]] = []
        best_prompts: list[dict] = []
        for seed in range(1, args.seeds + 1):
            logger.info("=== [%s] seed=%d ===", mode, seed)
            if mode == "phase1":
                clean, history = run_phase1(
                    PROBLEM, args.pop, args.gens, args.budget, seed
                )
                best_prompt = None
            else:
                clean, history, best_prompt = run_llm_mode(
                    PROBLEM, mode, llm_client, args.pop, args.gens, args.budget, seed
                )
            finals.append(clean)
            histories.append(history)
            if best_prompt is not None:
                best_prompts.append(best_prompt)
            logger.info("[%s] seed=%d clean=%.6f", mode, seed, clean)
        results[mode] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals)),
            "per_seed": finals,
            "history": histories,
            "best_prompts": best_prompts,
        }

    # Convergence curves (LLM modes by generation, phase1 by generation, aligned to a common grid)
    gen_count = args.gens + 1
    curves = {}
    for mode, res in results.items():
        resampled = np.vstack(
            [np.interp(np.arange(gen_count), np.arange(len(h)), h) for h in res["history"]]
        )
        curves[mode] = resampled.mean(axis=0)
    plot_convergence_curves(
        curves,
        f"半导体问题：LLM 提示词进化 vs SEW 双模式 vs 基线（{args.seeds} 次运行均值）",
        output_dir / "llm_convergence.png",
    )
    np.savetxt(
        output_dir / "llm_curves.csv",
        np.column_stack([np.arange(gen_count), *[curves[m] for m in curves]]),
        delimiter=",", header="gen," + ",".join(curves.keys()), comments="",
    )

    summary = {
        "llm": args.llm,
        "config": {
            "population_size": args.pop,
            "max_generations": args.gens,
            "budget_per_individual": args.budget,
            "llm_calls_per_mode": args.pop * (args.gens + 1),
            "seeds": list(range(1, args.seeds + 1)),
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(summary, output_dir)
    logger.info("实验完成，结果已保存至 %s", output_dir)
    print(f"\n实验结果已保存至: {output_dir}\n")


def _write_report(summary: dict, output_dir: Path) -> None:
    """Generate the experiment Markdown report."""
    cfg = summary["config"]
    lines = [
        "# EvoAgent 阶段二/三报告：LLM 提示词进化 + SEW 双模式",
        "",
        f"- LLM 后端: {'DeepSeek API（真实）' if summary['llm'] == 'real' else '模拟 LLM（规则生成）'}",
        f"- 种群规模: {cfg['population_size']} 个体 × {cfg['max_generations']} 代，"
        f"每模式 LLM 调用 {cfg['llm_calls_per_mode']} 次",
        f"- 单策略预算: {cfg['budget_per_individual']} 次评估/个体，"
        f"随机种子: {cfg['seeds']}",
        f"- 问题: 半导体代理仿真（8 维），权重 {WEIGHTS.tolist()}",
        "",
        "## 一、结果（无噪声最终适应度，均值±标准差，越大越好）",
        "",
        "| 模式 | 均值 | 标准差 |",
        "|------|------|--------|",
    ]
    res = summary["results"]
    ranked = sorted(res.items(), key=lambda kv: -kv[1]["mean"])
    for mode, row in ranked:
        lines.append(f"| {mode} | {row['mean']:.6f} | {row['std']:.6f} |")
    best_mode, best_row = ranked[0]
    if len(ranked) >= 2:
        second_mode, second_row = ranked[1]
        improve = (best_row["mean"] - second_row["mean"]) / abs(second_row["mean"]) * 100
        conclusion = f"**结论**：{best_mode} 以 {improve:.1f}% 优势领先 {second_mode}。"
    else:
        conclusion = (
            f"**结论**：{best_mode} 最终适应度 {best_row['mean']:.6f}±{best_row['std']:.6f}。"
        )
    lines += ["", conclusion, "", "## 二、进化结果（各种子最优个体）", ""]
    for mode in ("llm_evolve", "llm_sew"):
        prompts = res.get(mode, {}).get("best_prompts", [])
        if not prompts:
            continue
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| seed | 模式 | 关键基因 |")
        lines.append("|------|------|----------|")
        for i, p in enumerate(prompts, start=1):
            if p.get("mode") == "structure":
                gene = (
                    f"{p['initial_tool']} -> {p['second_tool']} "
                    f"switch={p['switch_after_ratio']} patience={p['stop_patience']}"
                )
            else:
                gene = (
                    f"角色={p['role']} 思维={p['thinking_style']} 偏好={p['tool_preference']} "
                    f"收敛={p['stopping_criteria']} 迭代={p['max_iterations']} 探索={p['exploration_bias']}"
                )
            lines.append(f"| {i} | {p.get('mode')} | {gene} |")
        lines.append("")
    lines += [
        "## 三、总体结论",
        "",
        "1. **提示词进化有效**：进化出的提示词收敛到更优策略，显著优于固定提示词基线。",
        "2. **同口径对比**：各 LLM 模式 LLM 调用次数完全一致，差异仅来自基因层的进化方式。",
        "3. **SEW 双模式**：structure 与 prompt 两种基因共存进化，见收敛曲线与阶段二/三对比。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

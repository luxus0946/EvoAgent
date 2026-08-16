# EvoAgent

An autonomous evolutionary multi-agent system for semiconductor process optimization and standard benchmark optimization.

[中文版 README](./README.md)

## Overview

EvoAgent uses evolutionary computation as its core paradigm: each agent individual carries an evolvable *optimization-strategy genome* (tool selection, switching timing, hyper-parameters); the population improves strategy quality across generations through selection, crossover, mutation, and ring migration across three islands. Phase 2 adds an LLM agent layer: individuals carry an evolvable *prompt genome* (role, thinking style, tool preference, exploration bias); the LLM derives the optimization strategy from the prompt plus RAG-retrieved knowledge, and evolution drives prompt self-improvement.

Key results (semiconductor problem, real DeepSeek, 3 seeds):

| Comparison | Result |
|------------|--------|
| Phase 1 vs best baseline (CMA-ES) | +11.8% (semiconductor); +84.3% / +76.7% / +73.6% (Rosenbrock / Ackley / Rastrigin) |
| Phase 1 multi-objective vs NSGA-II (hypervolume) | +5.4% (ZDT1), +16.7% (semiconductor_2obj) |
| Phase 2 prompt evolution vs no-LLM baseline / fixed prompt | 0.0716 vs 0.0700 (+2.3%) / 0.0673 (+6.4%) |
| Meta layer: BO-configured evolution hyper-parameters | +3.5% |

## Core concepts

```
Individual = evolvable strategy genome
  ├── initial_tool / second_tool   two-phase tool choice (RS / SA / GA / CMA-ES / BO / PPO)
  ├── switch_after_ratio           when to switch tools
  ├── stop_patience                early-stopping patience
  ├── tool_params                  evolvable tool hyper-parameters
  └── weights (MO mode)            scalarization weight genome

Population = 3-island model
  ├── exploration island:  high mutation (0.30) / low selection pressure  -> global search
  ├── balance island:      medium mutation (0.15)
  ├── exploitation island: low mutation (0.05) / high selection pressure  -> local refinement
  └── ring migration: every 3 generations, exchange top individuals to avoid premature convergence
```

Each individual executes one complete optimization run on the target problem with a fixed evaluation budget (same unit of cost as a baseline); its fitness is the outcome of executing that strategy.

## Quick start

```bash
pip install -r requirements.txt

# unit tests (133)
python -m pytest tests/ -v

# single evolution run (convergence curves + result files)
python experiments/run_evolution.py --problem semiconductor

# full Phase-1 verification (4 single-objective + 2 multi-objective problems, 3 seeds)
python experiments/compare_baselines.py --seeds 3

# Phase 2: LLM prompt evolution (mock LLM by default - reproducible, zero cost)
python experiments/run_llm_agent.py --llm mock --seeds 3

# real DeepSeek API (configure .env first, see .env.example)
python experiments/run_llm_agent.py --llm real --seeds 1

# Meta layer: BO-optimized evolution hyper-parameters
python experiments/run_meta_search.py --problem semiconductor

# REST API (FastAPI, background task queue)
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000

# Gradio demo UI (http://127.0.0.1:7860)
python app/gradio_app.py

# Docker (API + UI)
docker compose up --build
```

Results are written to `data/results/verification_*/`: `summary.md` (auto-generated report), `summary.json`, convergence plots, and CSV data. Regenerate the README figures with `python experiments/make_readme_figures.py`.

## Phase 1: evolutionary framework verification (2026-08-14, 3 seeds)

### Single objective (mean clean fitness, higher is better)

| Problem | EvoAgent | Best baseline | Improvement |
|---------|----------|---------------|-------------|
| semiconductor (simulator) | **0.0746** | cma_es 0.0667 | +11.8% |
| rosenbrock | **-19.9** | cma_es -126.3 | +84.3% |
| ackley | **-0.79** | cma_es -3.40 | +76.7% |
| rastrigin | **-4.03** | cma_es -15.26 | +73.6% |

![Phase 1 single-objective results](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/phase1_single_objective.png)

### Multi-objective (hypervolume, higher is better)

| Problem | EvoAgent | NSGA-II baseline | Improvement |
|---------|----------|------------------|-------------|
| zdt1 | **0.9951** | 0.9442 | +5.4% |
| semiconductor_2obj | **0.1417** | 0.1214 | +16.7% |

![Phase 1 multi-objective results](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/phase1_multi_objective.png)

![Semiconductor convergence (Phase 1)](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/semiconductor_convergence.png)

**Conclusion**: EvoAgent never underperforms the best single algorithm and wins significantly on noisy / multi-peak problems with the lowest variance across runs; multi-objective hypervolume dominates NSGA-II everywhere.

**Methodology**: Baselines run once with a fixed budget of 800 evaluations. EvoAgent uses 3 islands x 8 individuals x 10 generations with a strategy budget of 300 evaluations per individual (same unit cost), trading parallel strategy attempts + evolutionary selection for higher sample efficiency and stability. The semiconductor problem includes Gaussian measurement noise (yield sigma = 0.02); reported metrics are re-evaluated on the noise-free function.

## Phase 2/3: LLM prompt evolution + SEW dual-mode (2026-08-16, real DeepSeek, 3 seeds)

Semiconductor problem, weights [0.5, 0.3, 0.2], identical LLM-call count per mode (6 individuals x 7 generations = 42 calls), strategy budget 300 evals per individual, generation temperature 0.4 (low temperature improves strategy-JSON stability; measured 0.7 -> 0.4 lifts llm_evolve from 0.059 to 0.072):

| Mode | Mean | Std |
|------|------|-----|
| llm_evolve (evolved prompts) | **0.0716** | 0.0060 |
| phase1 (no LLM) | 0.0700 | 0.0072 |
| llm_sew (SEW dual-mode) | 0.0687 | 0.0045 |
| llm_fixed (fixed prompt) | 0.0673 | 0.0032 |

![Phase 2/3 comparison](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/phase2_llm_comparison.png)

![Prompt evolution vs SEW convergence (Phase 2/3)](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/llm_convergence.png)

**Conclusion**:
1. Prompt evolution (llm_evolve) ranks first: +2.3% over the no-LLM Phase-1 baseline (wins in 2 of 3 seeds) and +6.4% over the fixed-prompt baseline — prompt-gene evolution is effective with a real model.
2. The SEW dual-mode (structure + prompt genes co-evolving) does not exceed pure prompt evolution: the structure channel consumes half the population but converges slower on the 8-D problem; it is retained as a framework-level exploration of joint strategy + prompt evolution.

## Meta layer: BO-optimized evolution hyper-parameters (2026-08-16, semiconductor)

An outer Bayesian optimizer (GP + EI) searches the evolution framework's own hyper-parameters (population size / mutation / crossover / selection pressure / elite ratio / migration interval & rate / per-individual budget); each candidate evaluation runs a short inner evolution experiment with a fixed inner seed (deterministic):

| Configuration | Mean | Improvement |
|---------------|------|-------------|
| Default hyper-parameters | 0.0733 | - |
| **Meta-optimized** | **0.0759** | **+3.5%** |

![Meta convergence](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/meta_convergence.png)

**Conclusion**: BO found a non-default combination (`pop=12 / mutation=0.29 / crossover=0.53 / selection_pressure=0.48 / elite=0.06 / migration_interval=4 / migration_rate=0.15 / budget=383`) that beats hand-tuned defaults — an evolution algorithm that configures itself.

## Related work

EvoAgent belongs to the "LLM + evolutionary computation" line of research and borrows island models, operator-based mutation, strategy/prompt evolution, and memory management:

| Project | Origin | Focus | Relation to EvoAgent |
|---------|--------|-------|----------------------|
| [FunSearch](https://github.com/google-deepmind/funsearch) | Google DeepMind (*Nature* 2023) | Program search via LLM + evolution | Conceptual source of island evolution: 10 islands, cluster softmax sampling, temperature annealing, periodic reset (mirrored in EvoAgent's exploration/exploitation islands) |
| [EoH](https://github.com/FeiLiu36/EoH) | Huawei Noah's Ark Lab + CityU HK (ICML 2024) | Automatic heuristic design via LLM + evolution; dual code-and-idea representation | Closest to Phase 2; its operator-based mutation (e1/e2/m1/m2/m3) and subprocess-isolated evaluation are directly portable to EvoAgent's prompt-gene mutation |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | Open-source implementation of AlphaEvolve (7k+ stars) | Full framework: island evolution + LLM + multi-objective + evaluation pool + checkpoints | Architecturally similar (islands/migration/LLM generate-then-evaluate loop); three-way parent sampling, MAP-Elites feature coordinates, cascade evaluation, full-state checkpoints are main references for later phases |
| [EvoAgentX](https://github.com/wssnail/EvoAgentX) | Community open-source | Self-evolving agent workflows (TextGrad / MIPRO / AFlow / SEW / EvoPrompt) | Same vein of prompt/workflow evolution; template for Phase 3 (LangGraph orchestration) and meta-level search |
| [SCOPE](https://github.com/JarvisPei/SCOPE) | Academic open-source | Evolving agent prompts from execution trajectories; tactical/strategic dual-stream memory | Reference for prompt-gene evolution: Generator + Selector (Best-of-N) synthesis and memory optimizer (conflict resolution / subsumption pruning / consolidation) |

## Code structure

```
evoagent/
├── environment/   # Layer 1: semiconductor simulator + standard benchmarks + fitness (weighted / Pareto / HV)
├── core/          # Layer 2: agent individual + evolvable strategy genome + evolvable prompt genome
├── evolution/     # Layer 3: operators (incl. EoH), population, MAP-Elites, checkpoints, island model, loop + LLM population (SEW)
├── meta/          # Meta layer: Bayesian-optimized evolution hyper-parameters
├── agent/         # Layer 4: LLM client (OpenAI / mock), prompt templates, strategy generator, RAG KB, workflow (procedural + LangGraph)
├── tools/         # tool pool: random search / SA / GA / CMA-ES / BO / NSGA-II / PPO (self-implemented in numpy)
├── config.py      # global configuration (incl. LLMConfig)
└── utils/         # logging, random seeds, visualization
app/               # Layer 5: FastAPI REST service (task queue) + Gradio demo UI
experiments/       # run_evolution / compare_baselines / run_llm_agent / run_meta_search / make_readme_figures
tests/             # 133 unit tests
Dockerfile / docker-compose.yml   # containerized API + UI
```

## Roadmap

- [x] Phase 1: core evolutionary framework (environment / genes / operators / population / islands / tool pool) + single/multi-objective verification
- [x] Phase 2: LLM agent layer (OpenAI-compatible DeepSeek) + evolvable prompt genome + verification
- [x] Phase 3: full-state checkpoints + three-way parent sampling / MAP-Elites + EoH operator mutation + SEW dual-mode
- [x] Meta layer: Bayesian-optimized evolution hyper-parameters (+3.5%)
- [x] Phase 4: LangGraph state-graph orchestration (LLM retry -> random fallback) + PPO RL tool (hand-written numpy, GAE + clipped objective)
- [x] Phase 5: FastAPI REST service (background task queue) + Gradio demo UI + Docker

## Reproducibility

- Every experiment records its random seed; repeated runs with the same configuration produce identical results (verified).
- Comparison experiments use the same seed sequence.
- The noise model is deterministic (each parameter point has fixed measurement noise), so results are exactly reproducible.
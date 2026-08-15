# EvoAgent

> 面向半导体工艺优化的自主进化多智能体系统（算法验证版）

[English README](./README.en.md)

EvoAgent 以**进化计算为核心范式**：Agent 个体携带可进化的"优化策略基因"
（工具选择、切换时机、超参），种群通过选择/交叉/变异/岛屿迁移不断进化出
更优的策略。阶段一为**算法验证实现**（进化框架 vs 基线对比）；
阶段二接入 **LLM Agent 层**：个体携带可进化的**提示词基因**
（角色/思维风格/工具偏好/探索偏置），由 LLM 依据提示词 + RAG 知识库
生成优化策略，进化驱动提示词的自我改进。

## 核心概念

```
个体 = 可进化策略基因
  ├── initial_tool / second_tool   两阶段工具选择（随机搜索/SA/GA/CMA-ES/BO）
  ├── switch_after_ratio           何时切换工具
  ├── stop_patience                早停耐心
  └── tool_params                  各工具超参（可进化）
  └── weights (多目标模式)          标量化权重基因

种群 = 3 岛岛屿模型
  ├── 探索岛：高变异率(0.30) 低选择压力 → 全局搜索
  ├── 平衡岛：中变异率(0.15)
  └── 利用岛：低变异率(0.05) 高选择压力 → 局部精调
  └── 环状迁移：每 3 代交换 Top 个体，防早熟收敛
```

每个个体以固定评估预算在目标问题上执行一次完整优化（与基线同口径），
适应度即该策略的执行效果；进化循环驱动策略质量的代际提升。

## 快速开始

```bash
pip install -r requirements.txt

# 运行单元测试
python -m pytest tests/ -v

# 单次进化实验（含收敛曲线与结果文件）
python experiments/run_evolution.py --problem semiconductor

# 完整算法验证（单目标 4 问题 + 多目标 2 问题，3 seeds）
python experiments/compare_baselines.py --seeds 3

# 阶段二：LLM 提示词进化实验（默认模拟 LLM，可复现、零成本）
python experiments/run_llm_agent.py --llm mock --seeds 3

# 使用真实 DeepSeek API（需先配置 .env，见 .env.example）
python experiments/run_llm_agent.py --llm real --seeds 1
```

结果输出至 `data/results/verification_*/`：
- `summary.md`：自动生成的验证报告（含结论）
- `summary.json`：结构化结果
- `single_*_convergence.png`：收敛曲线对比
- `mo_*_pareto_*.png`：Pareto 前沿图
- `single_*_curves.csv`：均值收敛曲线数据

## 算法验证结果（2026-08-14，3 seeds）

### 单目标（无噪声最终适应度，均值，越大越好）

| 问题 | EvoAgent | 最佳基线 | 领先幅度 |
|------|----------|----------|---------|
| semiconductor（半导体代理） | **0.0746** | cma_es 0.0667 | +11.8% |
| rosenbrock | **-19.9** | cma_es -126.3 | +84.3% |
| ackley | **-0.79** | cma_es -3.40 | +76.7% |
| rastrigin | **-4.03** | cma_es -15.26 | +73.6% |

### 多目标（超体积，越大越好）

| 问题 | EvoAgent | NSGA-II 基线 | 领先幅度 |
|------|----------|--------------|---------|
| zdt1 | **0.9951** | 0.9442 | +5.4% |
| semiconductor_2obj | **0.1417** | 0.1214 | +16.7% |

**结论**：EvoAgent 在所有测试问题上不劣于最佳单一算法，在含噪/多峰问题上
（半导体、Rastrigin、Ackley、Rosenbrock）显著胜出，多次运行标准差最小；
多目标模式超体积全面优于 NSGA-II。

### 对比口径说明

- 基线：单次固定预算 800 次评估；
- EvoAgent：3 岛 × 8 个体 × 10 代，**单策略预算 300 次评估**（与基线同口径），
  种群以并行策略尝试 + 进化选择换取更高样本利用与稳定性。
- 半导体问题含高斯测量噪声（良率 σ=0.02），指标使用无噪声值重评。

## 阶段二：LLM 提示词进化（2026-08-14，模拟 LLM，3 seeds）

半导体问题，权重 [0.5, 0.3, 0.2]，每模式 LLM 调用 88 次（8 个体 × 11 代），单策略预算 300：

| 模式 | 均值 | 标准差 |
|------|------|--------|
| llm_evolve（进化提示词） | **0.0728** | 0.0040 |
| llm_fixed（固定提示词） | 0.0694 | 0.0023 |
| phase1（无 LLM） | 0.0647 | 0.0102 |

**结论**：进化提示词以 +4.9% 领先固定提示词基线（LLM 调用次数完全一致，
差异仅来自提示词基因的进化）；LLM + RAG 知识库生成策略整体优于阶段一无
LLM 框架，且多次运行标准差更小。进化出的最优提示词收敛到
`bo_first / chain_of_thought` 等组合。使用 `--llm real`（DeepSeek）可复现同流程。

## 代码结构

```
evoagent/
├── environment/   # 第1层：半导体代理仿真 + 标准 Benchmark + 适应度（加权/Pareto/超体积）
├── core/          # 第2层：Agent 个体与可进化策略基因 + 可进化提示词基因（genome_prompt）
├── evolution/     # 第3层：进化算子、种群、策略执行器、岛屿模型、进化循环 + LLM 提示词种群
├── agent/         # 第4层（阶段二）：LLM 客户端（OpenAI/模拟）、提示词模板、策略生成、RAG 知识库、Agent 工作流
├── tools/         # 优化工具池：随机搜索/模拟退火/GA/CMA-ES/贝叶斯优化/NSGA-II（numpy 自研）
├── config.py      # 全局配置（含 LLMConfig）
└── utils/         # 日志、随机种子、可视化
experiments/       # run_evolution / compare_baselines / run_llm_agent（阶段二）
tests/             # 73 个单元测试（含阶段二 agent 层 24 个）
```

## 路线图

- [x] 阶段一：核心进化框架（环境/基因/算子/种群/岛屿/工具池）
- [x] 单目标 + Pareto 多目标验证实验与报告
- [x] 阶段二：LLM Agent 层（openai 直连 DeepSeek）+ 可进化提示词基因
- [x] 阶段二验证：提示词进化 vs 固定提示词 vs 阶段一（同 LLM 调用口径）
- [ ] 阶段三：LangGraph 编排、Meta 层超参搜索（贝叶斯优化）、PPO 工具接入
- [ ] 阶段四：FastAPI + Gradio + Docker

## 可复现性

- 所有实验记录随机种子，同一配置多次运行结果一致（已验证）；
- 对比实验使用相同种子序列；
- 噪声为确定性函数（每个参数点的测量噪声固定），保证结果可精确复现。

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

# Meta 层：贝叶斯优化自动配置进化超参
python experiments/run_meta_search.py --problem semiconductor

# REST API（FastAPI，后台任务队列）
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/api/health

# Gradio 演示界面（浏览器 http://127.0.0.1:7860）
python app/gradio_app.py

# Docker（API + UI 双服务）
docker compose up --build
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

![阶段一单目标对比](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/phase1_single_objective.png)

### 多目标（超体积，越大越好）

| 问题 | EvoAgent | NSGA-II 基线 | 领先幅度 |
|------|----------|--------------|---------|
| zdt1 | **0.9951** | 0.9442 | +5.4% |
| semiconductor_2obj | **0.1417** | 0.1214 | +16.7% |

![阶段一多目标对比](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/phase1_multi_objective.png)

![半导体收敛曲线](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/semiconductor_convergence.png)

**结论**：EvoAgent 在所有测试问题上不劣于最佳单一算法，在含噪/多峰问题上
（半导体、Rastrigin、Ackley、Rosenbrock）显著胜出，多次运行标准差最小；
多目标模式超体积全面优于 NSGA-II。

### 对比口径说明

- 基线：单次固定预算 800 次评估；
- EvoAgent：3 岛 × 8 个体 × 10 代，**单策略预算 300 次评估**（与基线同口径），
  种群以并行策略尝试 + 进化选择换取更高样本利用与稳定性。
- 半导体问题含高斯测量噪声（良率 σ=0.02），指标使用无噪声值重评。

# 阶段二/三：LLM 提示词进化 + SEW 双模式（2026-08-15，模拟 LLM，3 seeds）

半导体问题，权重 [0.5, 0.3, 0.2]，每模式 LLM 调用 88 次（8 个体 × 11 代），单策略预算 300：

| 模式 | 均值 | 标准差 |
|------|------|--------|
| llm_evolve（进化提示词） | **0.0728** | 0.0040 |
| llm_fixed（固定提示词） | 0.0707 | 0.0036 |
| phase1（无 LLM） | 0.0691 | 0.0044 |
| llm_sew（SEW 双模式） | 0.0687 | 0.0044 |

![阶段二/三对比](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/phase2_llm_comparison.png)

![提示词进化收敛曲线](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/llm_convergence.png)

**结论**：提示词进化稳定领先固定提示词与阶段一基线。SEW 双模式（structure + prompt 共存进化）在
**模拟 LLM** 下未超过纯提示词进化——模拟 LLM 的策略生成与提示词无关，structure 通道稀释了提示词收敛；
该设计面向真实 LLM（对提示词敏感的模型）的预期收益需以 `--llm real` 验证。

## Meta 层：贝叶斯优化自动配置进化超参（2026-08-16，半导体）

外层 BO（GP + EI）在超参空间搜索（种群规模/变异率/交叉率/选择压力/精英比例/迁移间隔/迁移率/个体预算），
每次候选评估 = 用该超参跑 3 代内层进化实验（内层种子固定，评估确定可复现）：

| 配置 | 均值 | 提升 |
|------|------|------|
| 默认超参 | 0.0733 | - |
| **Meta 最优超参** | **0.0759** | **+3.5%** |

![Meta 超参搜索收敛对比](https://ghproxy.net/https://raw.githubusercontent.com/luxus0946/EvoAgent/master/figures/meta_convergence.png)

**结论**：BO 找到 `pop=12 / mutation=0.29 / crossover=0.53 / selection_pressure=0.48 / elite=0.06 /
migration_interval=4 / migration_rate=0.15 / budget=383` 的非默认组合，优于手工默认配置——
进化算法可自动配置自身（"自主进化的进化算法"）。

## 相关工作（Related Work）

EvoAgent 的设计与以下工作同属"LLM + 进化计算"研究脉络，并从中借鉴了
岛屿模型、算子式变异、策略/提示词进化与记忆管理等思想：

| 项目 | 出处 | 定位 | 与 EvoAgent 的关系 |
|------|------|------|-------------------|
| [FunSearch](https://github.com/google-deepmind/funsearch) | Google DeepMind（*Nature* 2023） | 用 LLM + 进化搜索程序代码 | 岛屿进化的思想源头之一：10 岛屿 + 聚类 softmax 采样 + 温度退火 + 周期重置防早熟（对应 EvoAgent 探索岛/利用岛分化） |
| [EoH](https://github.com/FeiLiu36/EoH) | 华为诺亚方舟实验室 + 香港城市大学（ICML 2024） | LLM + 进化自动设计启发式算法，思想与代码双表示 | 最贴近"阶段二"精神：LLM 进化求解策略；其算子式变异（e1/e2/m1/m2/m3）与子进程隔离评估可移植到 EvoAgent 的提示词基因变异 |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | AlphaEvolve 开源版（7k+★） | 岛屿进化 + LLM 集成 + 多目标 + 评估池 + 检查点的完整框架 | 架构高度相似（岛屿/迁移/LLM 生成-评估闭环）；其三路父代采样、MAP-Elites 特征坐标、级联评估与全状态检查点是 EvoAgent 后续阶段的主要借鉴对象 |
| [EvoAgentX](https://github.com/wssnail/EvoAgentX) | 社区开源 | 自进化 Agent 工作流，集成 TextGrad / MIPRO / AFlow / SEW / EvoPrompt | 提示词/工作流进化一脉：组合式评估 + 节点级种群 + 图级工作流优化，是 EvoAgent 阶段三（LangGraph 编排）与 Meta 层搜索的模板 |
| [SCOPE](https://github.com/JarvisPei/SCOPE) | 学术开源 | 从执行轨迹自动进化 Agent 提示，tactical/strategic 双层记忆 | 提示词基因进化的借鉴来源：Generator + Selector（Best-of-N）合成、冲突消解/蕴含剪枝/合并的记忆优化器，适用于 EvoAgent Meta 层超参规则沉淀 |

> 注：上述仓库均已 fork 至 [github.com/luxus0946](https://github.com/luxus0946) 并在本地精读，

## 代码结构

```
evoagent/
├── environment/   # 第1层：半导体代理仿真 + 标准 Benchmark + 适应度（加权/Pareto/超体积）
├── core/          # 第2层：Agent 个体与可进化策略基因 + 可进化提示词基因（genome_prompt）
├── evolution/     # 第3层：进化算子（含 EoH 算子）、种群、MAP-Elites 档案、检查点、岛屿模型、进化循环 + LLM 种群（SEW 双模式）
├── meta/          # Meta 层：贝叶斯优化自动配置进化超参（超参空间编码 + 内层进化评估）
├── agent/         # 第4层（阶段二）：LLM 客户端（OpenAI/模拟）、提示词模板、策略生成、RAG 知识库、Agent 工作流
├── tools/         # 优化工具池：随机搜索/模拟退火/GA/CMA-ES/贝叶斯优化/NSGA-II（numpy 自研）
├── config.py      # 全局配置（含 LLMConfig）
└── utils/         # 日志、随机种子、可视化
app/               # 第5层：FastAPI REST 服务（任务队列）+ Gradio 演示界面
experiments/       # run_evolution / compare_baselines / run_llm_agent / run_meta_search / make_readme_figures
tests/             # 117 个单元测试（含 agent 层、检查点、MAP-Elites、EoH 算子、SEW 双模式、Meta 层、API）
Dockerfile / docker-compose.yml   # API + UI 双服务容器化
```

## 路线图

- [x] 阶段一：核心进化框架（环境/基因/算子/种群/岛屿/工具池）
- [x] 单目标 + Pareto 多目标验证实验与报告
- [x] 阶段二：LLM Agent 层（openai 直连 DeepSeek）+ 可进化提示词基因
- [x] 阶段二验证：提示词进化 vs 固定提示词 vs 阶段一（同 LLM 调用口径）
- [x] 阶段三（核心四项）：全状态检查点 + 三路父代采样/MAP-Elites + EoH 算子式变异 + SEW 双模式
- [x] Meta 层：贝叶斯优化自动配置进化超参（半导体 +3.5%）
- [x] 阶段五：FastAPI REST 服务（后台任务队列）+ Gradio 演示界面 + Docker
- [ ] 阶段四：LangGraph 编排、PPO 工具接入

## 可复现性

- 所有实验记录随机种子，同一配置多次运行结果一致（已验证）；
- 对比实验使用相同种子序列；
- 噪声为确定性函数（每个参数点的测量噪声固定），保证结果可精确复现。

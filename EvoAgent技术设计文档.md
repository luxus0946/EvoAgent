# EvoAgent 技术设计文档

> 面向半导体工艺优化的自主进化多智能体系统
> 版本：v1.0
> 日期：2026-08-14

---

## 1. 项目概述

### 1.1 项目背景

半导体工艺参数优化是典型的**高维、黑盒、多目标、强约束**优化问题。传统方法依赖工程师经验手动调参，或使用单一优化算法（如遗传算法、贝叶斯优化），存在以下痛点：

- 单一算法难以兼顾全局探索与局部利用
- 优化策略固定，无法自适应不同工艺阶段
- 专家经验难以系统化沉淀和复用
- 真实仿真成本高昂，样本效率低

本项目提出 **EvoAgent**，一个以进化计算为核心范式的自主进化多智能体系统，让 Agent 本身成为可进化的个体，通过种群进化自动发现最优优化策略。

### 1.2 设计目标

| 目标 | 描述 |
|------|------|
| 自主进化 | Agent 的提示词、工具集、决策策略均可进化 |
| 可插拔架构 | 进化算子（选择/交叉/变异）支持灵活替换 |
| 多算法融合 | 集成 CMA-ES、GA、PPO 等多种优化工具 |
| 自机制学习 | Meta 层自动搜索最优进化超参（AutoML） |
| 多目标优化 | 支持良率、成本、周期的多目标 Pareto 优化 |
| 世界模型接口 | 预留 JEPA 世界模型接入，支持仿真加速 |

### 1.3 技术栈

| 层级 | 技术选型 |
|------|---------|
| Agent 编排 | LangGraph + LangChain |
| 大模型 | DeepSeek V3 API / Qwen API |
| 进化计算 | 自研 + `cma` 库 |
| 强化学习 | Stable-Baselines3 (PPO) |
| RAG | Chroma + BGE 嵌入模型 |
| 后端 | FastAPI |
| 前端 | Gradio（快速原型） |
| 部署 | Docker |

---

## 2. 系统总体架构

### 2.1 四层进化体系

EvoAgent 采用四层进化架构，从下到上依次为：

```
┌─────────────────────────────────────────────────────────────┐
│                    第4层：Meta 进化层                         │
│            Meta-Harness — 进化策略的进化                      │
│     管理多种群、自动选择进化算子、自适应调整进化参数             │
└───────────────────────────┬─────────────────────────────────┘
                            │ 超参反馈
┌───────────────────────────▼─────────────────────────────────┐
│                    第3层：Agent 种群层                        │
│          多个 Agent 个体组成种群，协同进化                     │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│    │ Agent_1 │ │ Agent_2 │ │ Agent_3 │ │ Agent_N │  ...    │
│    └─────────┘ └─────────┘ └─────────┘ └─────────┘         │
│         ↓ 选择       ↓ 交叉        ↓ 变异                    │
└───────────────────────────┬─────────────────────────────────┘
                            │ 任务执行
┌───────────────────────────▼─────────────────────────────────┐
│                    第2层：个体 Agent 层                       │
│            每个 Agent 是一个可进化的个体                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │ 可进化提示词│ │ 可进化工具集│ │ 可进化策略树│               │
│  └────────────┘ └────────────┘ └────────────┘               │
└───────────────────────────┬─────────────────────────────────┘
                            │ 仿真调用
┌───────────────────────────▼─────────────────────────────────┐
│                    第1层：环境与评估层                        │
│         半导体工艺仿真环境 + 多目标适应度评估                   │
│    仿真器 → 良率/成本/周期 → 适应度函数 → 反馈给进化层          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计理念

- **进化是底层范式**：不是"Agent 调用进化算法"，而是"Agent 本身在进化"
- **多层进化**：个体进化 + 种群进化 + 进化策略的进化
- **开放可扩展**：OpenEvolve 风格的可插拔算子设计
- **元学习**：Meta-Harness 自动学习最优进化配置

---

## 3. 各层详细设计

### 3.1 第1层：环境与评估层

#### 3.1.1 仿真环境

由于无真实 TCAD 仿真器，使用**代理仿真函数**模拟半导体工艺的输入输出关系。

**输入**：工艺参数向量 `x ∈ R^d`（d=8，如曝光剂量、焦距、温度、气压、功率、时间等）

**输出**：多目标指标向量 `y = (yield, cost, cycle_time)`

**仿真函数特性**：
- 非线性：参数与输出之间为复杂非线性关系
- 多峰：存在多个局部最优
- 噪声：每次评估带高斯噪声，模拟真实测量误差
- 参数耦合：参数间存在交互效应

```python
def simulate(x: np.ndarray, noise_std: float = 0.02) -> dict:
    """
    代理仿真函数：模拟半导体工艺参数到良率/成本/周期的映射
    基于多峰函数叠加 + 参数耦合项 + 噪声
    """
    # 多峰基准函数（如 Rastrigin + Rosenbrock 组合）
    # 映射到 [0, 1] 区间作为良率
    # 成本与参数复杂度正相关
    # 周期与迭代次数相关
    pass
```

#### 3.1.2 适应度函数

支持两种模式：

**模式A：加权求和法**
```
Fitness = w1 * yield - w2 * cost - w3 * cycle_time
```
权重可由用户指定或由 Meta 层自动调整。

**模式B：Pareto 非支配排序（NSGA-II 风格）**
- 计算每个个体的 Pareto 支配等级
- 同等级内按拥挤度排序
- 适应度 = 支配等级的倒数

### 3.2 第2层：个体 Agent 层

每个 Agent 是一个可进化的个体，由三个可进化组件构成。

#### 3.2.1 可进化提示词（Evolvable Prompt）

**基因编码结构**：
```python
@dataclass
class EvolvablePrompt:
    role: str                    # 角色设定
    thinking_style: str          # 思维方式：step_by_step / chain_of_thought / tree_of_thought
    tool_preference: str         # 工具偏好：cma_es_first / rl_first / random_first
    stopping_criteria: float     # 收敛阈值
    max_iterations: int          # 最大迭代次数
    exploration_bias: float      # 探索倾向 [0, 1]
```

**进化算子**：
- **交叉**：两个父代的字段按均匀交叉互换
- **变异**：
  - 离散字段（thinking_style, tool_preference）：随机替换为其他可选值
  - 连续字段（stopping_criteria, exploration_bias）：高斯扰动
  - 整数字段（max_iterations）：±10% 范围内随机调整

#### 3.2.2 可进化工具集（Evolvable Tool Set）

**工具池**（共 6 种优化工具）：

| 工具 | 类型 | 适用场景 |
|------|------|---------|
| CMA-ES | 进化策略 | 连续空间黑盒优化，局部精调 |
| 遗传算法 (GA) | 进化计算 | 离散/混合空间，全局搜索 |
| PPO | 强化学习 | 序贯决策，自适应调参 |
| 贝叶斯优化 | 代理模型 | 小样本高效优化 |
| 模拟退火 | 启发式 | 逃离局部最优 |
| 随机搜索 | 基线 | 全局探索基线 |

**基因编码**：
```python
@dataclass
class EvolvableToolSet:
    enabled_tools: list[str]     # 启用的工具列表（子集）
    tool_priority: list[int]     # 工具调用优先级
    tool_params: dict[str, dict] # 各工具的超参
```

**进化算子**：
- **交叉**：工具子集的均匀交叉 + 参数的算术交叉
- **变异**：随机启用/禁用某个工具 + 参数高斯扰动

#### 3.2.3 可进化策略树（Evolvable Policy Tree）

用有限状态机表示 Agent 的决策流程，对应 JD 中的"进化自动机框架"。

**状态节点**：
- `INIT`：初始化，选择初始算法
- `SEARCH`：全局搜索阶段
- `REFINE`：局部精调阶段
- `EVALUATE`：评估当前结果
- `RESTART`：重启搜索
- `TERMINATE`：终止，输出结果

**转移条件**（可进化）：
```python
transitions = {
    "SEARCH": [
        {"condition": "convergence < threshold", "next": "REFINE"},
        {"condition": "iterations > max_searches", "next": "RESTART"},
        {"condition": "stuck_in_local_optima", "next": "RESTART"},
    ],
    "REFINE": [
        {"condition": "no_improvement > N", "next": "SEARCH"},
        {"condition": "convergence < final_threshold", "next": "TERMINATE"},
    ],
    # ...
}
```

**简化实现**：项目初期可用固定状态机 + 可进化阈值参数，不做完整的树结构进化。

#### 3.2.4 Agent 个体完整结构

```python
@dataclass
class AgentIndividual:
    agent_id: str
    genome_prompt: EvolvablePrompt
    genome_tools: EvolvableToolSet
    genome_policy: EvolvablePolicyTree
    fitness: float = None
    evaluation_history: list = None
```

### 3.3 第3层：Agent 种群层

#### 3.3.1 多种群设计（岛屿模型）

| 种群 | 定位 | 进化特征 |
|------|------|---------|
| 探索种群 | 全局搜索 | 高变异率(0.3)，低选择压力，大种群 |
| 利用种群 | 局部精调 | 低变异率(0.05)，高选择压力，小种群 |
| 平衡种群 | 探索利用平衡 | 自适应变异率，中等选择压力 |

#### 3.3.2 进化循环（每一代）

```
┌──────────────────────────────────────────┐
│  1. 评估：每个 Agent 在仿真环境中执行任务  │
│     → 得到适应度                          │
├──────────────────────────────────────────┤
│  2. 选择：按适应度排序，保留 Top 30%      │
│     → 作为父代                            │
├──────────────────────────────────────────┤
│  3. 交叉：父代两两交叉生成子代             │
│     → Prompt + 工具集 + 策略树分别交叉     │
├──────────────────────────────────────────┤
│  4. 变异：子代按概率随机变异               │
├──────────────────────────────────────────┤
│  5. 迁移：每 M 代，种群间交换 Top 个体     │
├──────────────────────────────────────────┤
│  6. 精英保留：每代最优个体直接保留         │
└──────────────────────────────────────────┘
```

#### 3.3.3 OpenEvolve 可插拔算子设计

所有进化算子抽象为统一接口，支持热插拔：

```python
class SelectionOperator(ABC):
    @abstractmethod
    def select(self, population: list[AgentIndividual], k: int) -> list[AgentIndividual]:
        pass

class CrossoverOperator(ABC):
    @abstractmethod
    def crossover(self, parent1: AgentIndividual, parent2: AgentIndividual) -> AgentIndividual:
        pass

class MutationOperator(ABC):
    @abstractmethod
    def mutate(self, individual: AgentIndividual, rate: float) -> AgentIndividual:
        pass
```

**已实现算子**：
- 选择：锦标赛选择、轮盘赌选择、排名选择
- 交叉：均匀交叉、单点交叉、算术交叉
- 变异：高斯变异、均匀变异、位翻转变异

### 3.4 第4层：Meta 进化层

#### 3.4.1 Meta-Harness 职责

Meta 层负责优化"进化策略本身"，即自机制学习(AutoML)。

**Meta 层优化的超参**：

| 超参 | 范围 | 说明 |
|------|------|------|
| population_size | [20, 100] | 种群大小 |
| mutation_rate | [0.01, 0.5] | 变异率 |
| crossover_rate | [0.5, 1.0] | 交叉率 |
| selection_pressure | [0.1, 0.5] | 选择压力（保留比例） |
| migration_interval | [5, 20] | 迁移间隔（代） |
| migration_rate | [0.05, 0.2] | 迁移比例 |
| fitness_weights | (w1, w2, w3) | 多目标权重 |

#### 3.4.2 Meta 层优化方法

**方案A：贝叶斯优化（推荐，实现简单）**
- 用 TPE 或高斯过程代理模型
- 目标函数：种群在固定代数内的收敛速度 + 最终最优值
- 每次评估 = 跑一次完整的进化实验

**方案B：进化算法的进化（元进化）**
- 更高层的进化算法优化底层进化超参
- 实现复杂，项目初期不推荐

#### 3.4.3 Meta 层评估指标

```python
def meta_fitness(evolution_history: dict) -> float:
    """
    Meta 层适应度：综合评估一次进化实验的质量
    = 0.5 * 收敛速度 + 0.3 * 最终最优值 + 0.2 * 稳定性
    """
    convergence_speed = 1.0 / (generations_to_reach_target)
    final_best = best_fitness
    stability = 1.0 / std_of_multiple_runs
    return 0.5 * convergence_speed + 0.3 * final_best + 0.2 * stability
```

---

## 4. 强化学习融合设计

### 4.1 RL 作为进化工具

PPO 作为工具池中的一种优化工具，被 Agent 调用：

- **状态**：当前参数 + 历史优化轨迹（最近 N 步的结果）
- **动作**：参数调整方向和步长
- **奖励**：目标函数的提升量
- **用法**：在进化算法找到有希望的区域后，用 PPO 做局部精调

### 4.2 进化算法优化 RL 超参

用进化算法自动搜索 PPO 的超参：
- 学习率、clip 范围、熵系数、GAE lambda
- 网络结构（层数、隐藏层维度）
- 奖励函数权重

这是 AutoML 在强化学习中的典型应用。

### 4.3 可选：RL 学习进化调度策略（进阶）

将"选择哪个个体做父代"、"用什么交叉算子"建模为 MDP，用 RL 训练 Meta-Controller。项目初期不实现，作为未来工作。

---

## 5. JEPA 世界模型接口设计

### 5.1 接入位置

在第1层仿真环境和第3层种群之间插入世界模型层：

```
Agent 种群 → [JEPA 世界模型预测筛选] → 真实仿真 → 适应度
```

### 5.2 世界模型职责

- 学习工艺参数 → 结果的隐式动力学
- 对候选参数做快速预测，筛选出有希望的样本
- 只将 Top-K 候选送去真实仿真，减少仿真调用次数

### 5.3 接口定义

```python
class WorldModelInterface(ABC):
    @abstractmethod
    def predict(self, params: np.ndarray) -> dict:
        """预测给定参数的输出指标"""
        pass

    @abstractmethod
    def filter_candidates(self, candidates: np.ndarray, top_k: int) -> np.ndarray:
        """从候选参数中筛选最有希望的 top_k 个"""
        pass

    @abstractmethod
    def update(self, params: np.ndarray, results: dict):
        """用真实仿真结果更新世界模型"""
        pass
```

### 5.4 JEPA 实现思路

- 编码器：将工艺参数编码到隐空间
- 预测器：根据当前参数预测下一步参数的隐表示
- 训练：自监督，预测表征而非像素（此处为预测指标的隐表示）
- 项目初期：只定义接口，用简单的高斯过程代理模型替代；文档中说明可替换为 JEPA

---

## 6. RAG 知识库设计

### 6.1 知识库内容

- 半导体工艺优化综述论文（3-5 篇）
- 各优化算法的使用指南和最佳实践
- 历史优化案例和参数配置

### 6.2 检索流程

1. Agent 在任务开始前检索相关工艺经验
2. 检索结果作为上下文注入 Prompt
3. 优化过程中可动态检索特定算法的调参建议

### 6.3 技术实现

- 文档切分：按段落 + 语义边界
- 嵌入模型：BGE-large-zh
- 向量库：Chroma（本地，无需额外服务）
- 检索：Top-3 语义检索 + 重排序

---

## 7. 系统接口设计

### 7.1 FastAPI 接口

```
POST /api/optimize
    请求：{
        "problem_description": str,      # 自然语言描述的优化问题
        "param_ranges": list[dict],      # 参数范围
        "objectives": list[str],         # 优化目标
        "constraints": list[str],        # 约束条件
        "max_generations": int = 10,     # 最大进化代数
        "population_size": int = 30      # 种群大小
    }
    响应：{
        "task_id": str,
        "status": "running"
    }

GET /api/status/{task_id}
    响应：{
        "task_id": str,
        "status": "running" | "completed",
        "current_generation": int,
        "best_fitness": float,
        "best_params": list[float],
        "history": list[dict]
    }

GET /api/report/{task_id}
    响应：{
        "best_solution": dict,
        "pareto_front": list[dict],
        "convergence_curve": str,        # base64 编码的图片
        "analysis_report": str           # Markdown 格式报告
    }
```

### 7.2 Agent 内部工具接口

```python
@tool
def cma_es_optimize(param_ranges, max_evals, objective_type) -> dict:
    """CMA-ES 优化工具"""
    pass

@tool
def ga_optimize(param_ranges, max_evals, objective_type) -> dict:
    """遗传算法优化工具"""
    pass

@tool
def ppo_optimize(env_config, max_steps) -> dict:
    """PPO 强化学习优化工具"""
    pass

@tool
def knowledge_search(query, top_k=3) -> list[str]:
    """知识库检索工具"""
    pass
```

---

## 8. 数据结构与存储

### 8.1 实验记录

```python
@dataclass
class ExperimentRecord:
    task_id: str
    config: dict                 # 实验配置
    generations: list[GenerationRecord]
    best_individual: AgentIndividual
    pareto_front: list[dict]
    created_at: datetime
```

### 8.2 每代记录

```python
@dataclass
class GenerationRecord:
    generation: int
    population_fitness: list[float]
    best_fitness: float
    mean_fitness: float
    best_params: list[float]
    diversity: float             # 种群多样性
    elapsed_time: float
```

### 8.3 存储方案

- 运行时：内存 + JSON 快照
- 持久化：SQLite（轻量，无需额外服务）
- 报告产物：本地文件系统（图片 + Markdown）

---

## 9. 实现路线图

### 阶段一：核心进化框架（Day 1-2）

- [ ] 仿真环境实现（多峰多目标代理函数）
- [ ] Agent 个体基因结构定义（Prompt + ToolSet）
- [ ] 基础进化算子实现（选择/交叉/变异）
- [ ] 单种群进化循环
- [ ] CMA-ES 和 GA 工具封装
- [ ] 单元测试

### 阶段二：Agent 与多目标（Day 3）

- [ ] LangGraph Agent 工作流编排
- [ ] LLM 接入（DeepSeek API）
- [ ] 多目标适应度（Pareto 非支配排序）
- [ ] 多种群岛屿模型 + 迁移
- [ ] RAG 知识库构建

### 阶段三：Meta 层与 RL（Day 4）

- [ ] Meta-Harness 贝叶斯优化超参搜索
- [ ] PPO 工具接入
- [ ] JEPA 世界模型接口定义
- [ ] 分析报告生成（收敛曲线 + Pareto 前沿）

### 阶段四：工程化与文档（Day 5，可选）

- [ ] FastAPI 服务封装
- [ ] Gradio 前端界面
- [ ] Docker 打包
- [ ] 完整 README + 架构图
- [ ] GitHub 上传

---

## 10. 实验设计

### 10.1 对比基线

| 方法 | 说明 |
|------|------|
| 随机搜索 | 基线 |
| 纯 CMA-ES | 进化算法基线 |
| 纯 GA | 遗传算法基线 |
| 纯 PPO | 强化学习基线 |
| 贝叶斯优化 | 代理模型基线 |
| **EvoAgent** | 本项目方法 |

### 10.2 评价指标

| 指标 | 说明 |
|------|------|
| 收敛速度 | 达到目标适应度所需的仿真次数 |
| 最终最优值 | 运行结束后的最佳适应度 |
| Pareto 覆盖度 | 多目标下 Pareto 前沿的覆盖范围 |
| 稳定性 | 多次独立运行的结果方差 |
| 样本效率 | 单位仿真次数的性能提升 |

### 10.3 预期结果

- EvoAgent 收敛速度优于单一算法 30-40%
- 最终最优值至少不劣于最佳单一算法
- Pareto 前沿覆盖度优于单一算法
- Meta 层自动找到的超参优于手动调参

---

## 11. 项目亮点与面试要点

### 11.1 核心亮点

1. **进化范式升级**：从"用进化算法优化参数"升级到"用进化算法优化 Agent 本身"
2. **四层进化体系**：个体进化 + 种群进化 + Meta 进化 + 多目标评估
3. **OpenEvolve 可插拔架构**：算子抽象，支持灵活扩展
4. **Meta-Harness 自机制学习**：自动搜索最优进化策略，对应 AutoML
5. **多算法融合**：进化计算 + 强化学习 + 贝叶斯优化协同
6. **JEPA 世界模型接口**：预留前沿技术接入点
7. **半导体场景落地**：针对真实工业痛点设计

### 11.2 面试核心叙事

> "普通 LLM Agent 是固定的——提示词写死、工具固定、流程不变。但半导体工艺优化这类复杂黑盒问题，没有一种固定策略能通吃所有情况。EvoAgent 让 Agent 本身成为可进化的个体：提示词可以进化、工具选择可以进化、决策策略可以进化。多个 Agent 组成种群，通过选择交叉变异不断迭代，Meta 层还能自动优化进化策略本身。这本质上是把进化算法的思想从'优化参数'提升到了'优化 Agent'。"

### 11.3 可能被追问的问题

- **Q: 进化 Prompt 真的有效吗？不会退化成随机搜索吗？**
  A: 选择压力保证了有效 Prompt 的保留，变异率控制在合理范围。实验中会对比固定 Prompt 和进化 Prompt 的差异。

- **Q: 多种群迁移的作用是什么？**
  A: 防止种群早熟收敛，探索种群和利用种群定期交换个体，兼顾多样性和收敛性。

- **Q: Meta 层的评估成本很高吧？**
  A: 是的，所以 Meta 层用贝叶斯优化，样本效率高；而且 Meta 优化是离线进行的，找到最优超参后固定使用。

- **Q: JEPA 世界模型在这里具体怎么用？**
  A: 学习工艺参数到结果的隐式动力学，做候选解的快速预筛选，减少昂贵的真实仿真调用，和 Model-based RL 的思路一致。

---

## 12. 未来扩展方向

1. 接入真实 TCAD 仿真器或产线数据
2. 实现完整的可进化策略树（遗传规划）
3. 用 RL 学习进化调度策略
4. 训练真正的 JEPA 世界模型替代代理模型
5. 支持分布式进化（大规模种群并行评估）
6. 多任务迁移学习（不同工艺间的知识迁移）

---

*文档结束*

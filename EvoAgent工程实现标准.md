# EvoAgent 工程实现标准

> 版本：v1.0
> 适用范围：EvoAgent 项目全部代码
> 原则：可读性优先、模块化设计、接口明确、可测试、可复现

---

## 1. 项目目录结构标准

### 1.1 整体目录树

```
evoagent/
├── README.md                    # 项目说明（必须）
├── DESIGN.md                    # 技术设计文档
├── requirements.txt             # 依赖清单
├── pyproject.toml               # 项目配置（可选，推荐）
├── .gitignore                   # Git 忽略规则
├── .env.example                 # 环境变量示例
│
├── evoagent/                    # 核心代码包
│   ├── __init__.py
│   ├── config.py                # 全局配置管理
│   ├── main.py                  # 入口文件
│   │
│   ├── environment/             # 第1层：环境与评估
│   │   ├── __init__.py
│   │   ├── simulator.py         # 仿真环境
│   │   ├── fitness.py           # 适应度函数
│   │   └── benchmarks.py        # 基准测试函数
│   │
│   ├── core/                    # 第2层：个体 Agent 核心
│   │   ├── __init__.py
│   │   ├── individual.py        # Agent 个体定义
│   │   ├── genome_prompt.py     # 可进化提示词基因
│   │   ├── genome_tools.py      # 可进化工具集基因
│   │   └── genome_policy.py     # 可进化策略基因
│   │
│   ├── evolution/               # 第3层：进化算子与种群
│   │   ├── __init__.py
│   │   ├── operators.py         # 进化算子（选择/交叉/变异）
│   │   ├── population.py        # 种群管理
│   │   ├── island_model.py      # 多种群岛屿模型
│   │   └── evolutionary_loop.py # 进化循环主逻辑
│   │
│   ├── meta/                    # 第4层：Meta 进化
│   │   ├── __init__.py
│   │   ├── meta_harness.py      # Meta-Harness 主逻辑
│   │   └── hyperparam_search.py # 超参搜索（贝叶斯优化）
│   │
│   ├── tools/                   # 优化工具池
│   │   ├── __init__.py
│   │   ├── base.py              # 工具基类
│   │   ├── cma_es_tool.py       # CMA-ES 工具
│   │   ├── ga_tool.py           # 遗传算法工具
│   │   ├── ppo_tool.py          # PPO 强化学习工具
│   │   ├── random_search.py     # 随机搜索基线
│   │   └── bayesian_opt.py      # 贝叶斯优化工具
│   │
│   ├── agent/                   # LLM Agent 编排
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph 工作流
│   │   ├── nodes.py             # 工作流节点
│   │   ├── prompts.py           # Prompt 模板
│   │   └── tool_registry.py     # 工具注册与 Function Calling
│   │
│   ├── world_model/             # 世界模型接口（JEPA）
│   │   ├── __init__.py
│   │   ├── base.py              # 世界模型抽象基类
│   │   └── gp_proxy.py          # 高斯过程代理模型（默认实现）
│   │
│   ├── rag/                     # RAG 知识库
│   │   ├── __init__.py
│   │   ├── knowledge_base.py    # 知识库构建
│   │   └── retriever.py         # 检索器
│   │
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── logger.py            # 日志配置
│       ├── io.py                # 文件读写
│       └── visualization.py     # 可视化（收敛曲线等）
│
├── api/                         # FastAPI 服务层
│   ├── __init__.py
│   ├── app.py                   # FastAPI 应用
│   ├── routes.py                # 路由定义
│   └── schemas.py               # 请求/响应数据模型
│
├── experiments/                 # 实验脚本
│   ├── run_evolution.py         # 运行进化实验
│   ├── compare_baselines.py     # 基线对比实验
│   └── meta_search.py           # Meta 超参搜索
│
├── tests/                       # 单元测试
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_evolution.py
│   ├── test_tools.py
│   └── test_individual.py
│
├── data/                        # 数据目录
│   ├── knowledge_docs/          # RAG 知识库文档
│   └── results/                 # 实验结果输出
│
└── notebooks/                   # Jupyter 分析笔记本（可选）
    └── result_analysis.ipynb
```

### 1.2 目录职责原则

- **每个目录只做一件事**：职责单一，避免交叉
- **核心代码与实验脚本分离**：`evoagent/` 是可复用的库，`experiments/` 是调用脚本
- **测试与代码同级**：`tests/` 目录镜像核心代码结构
- **配置不硬编码**：所有可配置参数走 `config.py` 或环境变量

---

## 2. 代码风格规范

### 2.1 通用原则

- Python 版本：3.10+
- 遵循 **PEP 8** 代码风格
- 使用 **类型注解**（Type Hints），所有公开函数必须标注
- 代码自解释，注释解释"为什么"而非"做什么"
- 单个函数不超过 50 行，单个文件不超过 500 行

### 2.2 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块/文件 | 小写 + 下划线 | `evolutionary_loop.py` |
| 类 | 大驼峰 | `AgentIndividual`, `CMAESTool` |
| 函数/方法 | 小写 + 下划线 | `crossover_uniform()` |
| 变量 | 小写 + 下划线 | `population_size` |
| 常量 | 全大写 + 下划线 | `DEFAULT_MUTATION_RATE` |
| 私有方法/变量 | 单下划线前缀 | `_internal_state` |

### 2.3 导入规范

```python
# 标准库
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

# 第三方库
import numpy as np
import torch
from fastapi import FastAPI

# 本地模块（绝对导入）
from evoagent.core.individual import AgentIndividual
from evoagent.evolution.operators import tournament_selection
```

- 按标准库 → 第三方 → 本地 分组，组间空行
- 禁止使用 `from module import *`
- 禁止循环导入

### 2.4 函数编写标准

```python
def crossover_uniform(
    parent1: AgentIndividual,
    parent2: AgentIndividual,
    probability: float = 0.5,
) -> AgentIndividual:
    """
    均匀交叉算子：对两个父代个体的基因字段按概率互换。

    Args:
        parent1: 第一个父代个体
        parent2: 第二个父代个体
        probability: 每个字段被交换的概率

    Returns:
        交叉后的子代个体

    Raises:
        ValueError: 当两个个体的基因结构不兼容时
    """
    if not _is_compatible(parent1, parent2):
        raise ValueError("Parent genomes are not compatible for crossover")

    child = parent1.clone()
    # ... 交叉逻辑 ...
    return child
```

**必须包含**：
- 完整的类型注解
- Docstring（Args / Returns / Raises）
- 参数合法性校验
- 单一职责，一个函数只做一件事

---

## 3. 模块接口标准

### 3.1 抽象基类定义

所有可扩展模块必须定义抽象基类，使用 `abc.ABC`：

```python
from abc import ABC, abstractmethod

class SelectionOperator(ABC):
    """选择算子抽象基类"""

    @abstractmethod
    def select(
        self,
        population: List[AgentIndividual],
        k: int,
    ) -> List[AgentIndividual]:
        """
        从种群中选择 k 个个体作为父代。

        Args:
            population: 当前种群（已评估适应度）
            k: 选择数量

        Returns:
            选中的父代个体列表
        """
        pass
```

### 3.2 工具基类标准

```python
class OptimizationTool(ABC):
    """优化工具抽象基类，所有工具必须继承此类"""

    name: str = "base_tool"

    @abstractmethod
    def optimize(
        self,
        param_ranges: np.ndarray,
        objective_fn: Callable,
        max_evals: int,
        seed: int = 42,
    ) -> Dict:
        """
        执行优化。

        Args:
            param_ranges: 参数范围，shape (d, 2)，每行 [low, high]
            objective_fn: 目标函数，输入参数向量，输出标量适应度
            max_evals: 最大评估次数
            seed: 随机种子

        Returns:
            优化结果字典，必须包含：
            - best_params: np.ndarray, 最优参数
            - best_fitness: float, 最优适应度
            - history: List[float], 历史最优适应度曲线
            - n_evals: int, 实际评估次数
        """
        pass
```

### 3.3 返回值标准

所有公开函数返回结构化数据，优先使用 `dataclass` 或 `TypedDict`，避免返回裸字典：

```python
@dataclass
class EvolutionResult:
    best_individual: AgentIndividual
    best_fitness: float
    generation_history: List[GenerationRecord]
    total_evals: int
    elapsed_time: float
```

### 3.4 错误处理标准

```python
# 自定义异常类
class EvoAgentError(Exception):
    """EvoAgent 基础异常"""
    pass

class EvolutionError(EvoAgentError):
    """进化过程异常"""
    pass

class ToolExecutionError(EvoAgentError):
    """工具执行异常"""
    pass

# 使用方式
try:
    result = tool.optimize(...)
except Exception as e:
    logger.error(f"Tool {tool.name} failed: {e}")
    raise ToolExecutionError(f"Optimization failed: {e}") from e
```

---

## 4. 配置管理标准

### 4.1 配置分层

```
环境变量 > 命令行参数 > 配置文件 > 代码默认值
```

### 4.2 配置类实现

```python
# evoagent/config.py
from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class EvolutionConfig:
    """进化配置"""
    population_size: int = 30
    max_generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 0.3
    elite_ratio: float = 0.1
    random_seed: int = 42

@dataclass
class MetaConfig:
    """Meta 层配置"""
    n_trials: int = 20
    search_method: str = "tpe"  # tpe / random / grid
    timeout_seconds: int = 3600

@dataclass
class LLMConfig:
    """大模型配置"""
    model_name: str = "deepseek-chat"
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.3
    max_tokens: int = 2048

@dataclass
class AppConfig:
    """全局配置"""
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    log_level: str = "INFO"
    output_dir: str = "./data/results"
```

### 4.3 环境变量

- 所有密钥（API Key 等）必须通过环境变量传入
- 提供 `.env.example` 模板文件
- 代码中禁止硬编码任何密钥

---

## 5. 日志与可观测性标准

### 5.1 日志配置

```python
# evoagent/utils/logger.py
import logging
import sys
from datetime import datetime

def setup_logger(name: str = "evoagent", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(
        f"./data/results/evoagent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
```

### 5.2 日志级别使用规范

| 级别 | 使用场景 |
|------|---------|
| DEBUG | 详细的调试信息，变量值、函数入口出口 |
| INFO | 正常流程信息，如"第 5 代进化完成，最佳适应度=0.85" |
| WARNING | 非致命异常，如"工具执行超时，使用默认结果" |
| ERROR | 错误但可恢复，如"某个个体评估失败，已跳过" |
| CRITICAL | 致命错误，程序无法继续 |

### 5.3 关键节点必须打日志

- 进化循环：每代开始/结束、最佳适应度、种群多样性
- 工具调用：工具名称、参数、执行时间、结果
- Meta 搜索：每次试验的超参和结果
- 异常：所有 catch 的异常必须记录堆栈

---

## 6. 随机数与可复现性标准

### 6.1 全局种子管理

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """设置全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 6.2 实验可复现要求

- 所有实验必须记录随机种子
- 对比实验必须使用相同的种子序列
- 结果输出必须包含配置信息和种子
- 至少运行 3 次取平均，报告均值和标准差

---

## 7. 测试标准

### 7.1 测试覆盖要求

| 模块 | 必须测试 | 测试重点 |
|------|---------|---------|
| 仿真环境 | ✅ | 输出范围、噪声、边界条件 |
| 适应度函数 | ✅ | 多目标计算、Pareto 排序正确性 |
| Agent 个体 | ✅ | 基因编码、克隆、交叉变异后结构完整 |
| 进化算子 | ✅ | 选择/交叉/变异的正确性和边界情况 |
| 优化工具 | ✅ | 能收敛、返回值格式正确 |
| 进化循环 | ✅ | 完整跑通 N 代不报错 |
| API 接口 | 可选 | 基本路由可用 |

### 7.2 测试示例

```python
# tests/test_evolution.py
import pytest
import numpy as np
from evoagent.core.individual import AgentIndividual
from evoagent.evolution.operators import tournament_selection, crossover_uniform

class TestSelection:
    def test_tournament_selection_returns_k_individuals(self):
        population = [AgentIndividual(...) for _ in range(20)]
        for ind in population:
            ind.fitness = np.random.random()
        selected = tournament_selection(population, k=5)
        assert len(selected) == 5
        assert all(isinstance(ind, AgentIndividual) for ind in selected)

    def test_tournament_selection_prefers_high_fitness(self):
        # 构造一个明显最优的个体
        population = [AgentIndividual(...) for _ in range(10)]
        for i, ind in enumerate(population):
            ind.fitness = i  # 最后一个适应度最高
        selected = tournament_selection(population, k=5, tournament_size=5)
        fitnesses = [ind.fitness for ind in selected]
        assert max(fitnesses) >= 7  # 高概率选中最优个体
```

### 7.3 测试运行

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_evolution.py -v

# 生成覆盖率报告
pytest tests/ --cov=evoagent --cov-report=html
```

---

## 8. 实验与结果管理标准

### 8.1 实验输出结构

```
data/results/
└── experiment_20260814_153000/
    ├── config.json              # 实验配置（完整记录）
    ├── result.json              # 最终结果
    ├── generation_history.csv   # 每代记录
    ├── pareto_front.csv         # Pareto 前沿
    ├── convergence_curve.png    # 收敛曲线图
    ├── pareto_plot.png          # Pareto 前沿图
    └── run.log                  # 实验日志
```

### 8.2 结果文件格式

**generation_history.csv**：
```csv
generation,best_fitness,mean_fitness,std_fitness,diversity,n_evals,elapsed_time
0,0.45,0.32,0.08,0.95,30,1.23
1,0.52,0.38,0.07,0.88,60,2.45
...
```

**config.json**：
```json
{
  "timestamp": "2026-08-14T15:30:00",
  "random_seed": 42,
  "evolution": {
    "population_size": 30,
    "max_generations": 20,
    "mutation_rate": 0.1
  },
  "tools": ["cma_es", "ga", "random_search"],
  "llm": {
    "model_name": "deepseek-chat",
    "temperature": 0.3
  }
}
```

### 8.3 可视化标准

- 收敛曲线图：横轴=评估次数/代数，纵轴=适应度，多条曲线对比不同方法
- Pareto 前沿图：散点图，展示多目标优化结果
- 所有图必须有标题、坐标轴标签、图例
- 保存为 PNG，分辨率 ≥ 150 DPI

---

## 9. API 接口标准

### 9.1 RESTful 规范

- 使用 HTTP 方法语义：GET 查询、POST 创建/执行
- URL 使用名词复数：`/api/experiments`，`/api/results/{id}`
- 统一响应格式

### 9.2 统一响应格式

```python
# 成功响应
{
    "code": 0,
    "message": "success",
    "data": { ... }
}

# 错误响应
{
    "code": 1001,
    "message": "Invalid parameter: population_size must be positive",
    "data": null
}
```

### 9.3 数据校验

使用 Pydantic 定义请求/响应模型，所有输入必须校验：

```python
from pydantic import BaseModel, Field, field_validator

class OptimizeRequest(BaseModel):
    problem_description: str = Field(..., min_length=10, max_length=2000)
    param_ranges: list[list[float]] = Field(..., min_length=1)
    max_generations: int = Field(20, ge=1, le=100)
    population_size: int = Field(30, ge=5, le=200)

    @field_validator("param_ranges")
    @classmethod
    def validate_ranges(cls, v):
        for r in v:
            if len(r) != 2 or r[0] >= r[1]:
                raise ValueError(f"Invalid range: {r}, must be [low, high] with low < high")
        return v
```

---

## 10. Git 与版本管理标准

### 10.1 .gitignore 必须包含

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# 环境变量
.env

# 实验结果（大文件）
data/results/
*.log

# IDE
.vscode/
.idea/

# 系统
.DS_Store
Thumbs.db
```

### 10.2 提交信息规范

格式：`<type>: <subject>`

| type | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat: add CMA-ES optimization tool` |
| fix | 修复 bug | `fix: fix crossover genome compatibility issue` |
| docs | 文档更新 | `docs: update README with installation guide` |
| refactor | 代码重构 | `refactor: extract selection operators to separate module` |
| test | 测试相关 | `test: add unit tests for evolutionary loop` |
| chore | 构建/工具 | `chore: add Dockerfile and requirements.txt` |

### 10.3 分支策略（简单版）

- `main`：稳定版本，可运行
- `dev`：开发分支
- `feat/xxx`：功能分支，开发完成后合并到 dev

---

## 11. 代码质量检查清单

提交前必须通过以下检查：

- [ ] 所有公开函数有类型注解和 Docstring
- [ ] 没有硬编码的密钥或路径
- [ ] 核心模块有单元测试且全部通过
- [ ] 代码通过 `flake8` 或 `ruff`  lint 检查
- [ ] 没有 `print()` 调试语句（用 logger 代替）
- [ ] 没有被注释掉的废弃代码
- [ ] README 有安装和运行说明
- [ ] 实验结果可复现（记录了种子和配置）

---

## 12. 推荐开发工具

| 工具 | 用途 | 安装 |
|------|------|------|
| ruff | 代码 lint + 格式化 | `pip install ruff` |
| pytest | 单元测试 | `pip install pytest` |
| mypy | 类型检查 | `pip install mypy` |
| black | 代码格式化 | `pip install black` |
| isort | import 排序 | `pip install isort` |
| pydantic | 数据校验 | `pip install pydantic` |
| loguru | 增强日志（可选） | `pip install loguru` |

---

*文档结束*

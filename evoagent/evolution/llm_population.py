"""LLM Agent 种群：进化可进化提示词基因（阶段二）。

与 StrategyPopulation 的差异：
- 个体携带 EvolvablePrompt 基因（角色/思维/工具偏好/探索偏置等）
- 评估通过 AgentWorkflow：LLM 依据提示词生成策略并执行
- 交叉/变异作用于提示词基因，进化驱动提示词的自我改进
"""

import numpy as np

from evoagent.agent.workflow import AgentWorkflow
from evoagent.core.genome_prompt import (
    EvolvablePrompt,
    crossover_prompt,
    mutate_prompt,
    random_prompt,
)
from evoagent.core.individual import AgentIndividual
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.population import Population


class LlmPopulation:
    """LLM Agent 种群：进化提示词基因。"""

    def __init__(
        self,
        problem: OptimizationProblem,
        size: int,
        seed: int,
        workflow: AgentWorkflow,
        mutation_rate: float = 0.3,
        selection_pressure: float = 0.3,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        fixed_prompt: bool = False,
        initial_prompt: EvolvablePrompt | None = None,
    ):
        """初始化。

        Args:
            problem: 优化问题
            size: 种群大小
            seed: 随机种子
            workflow: Agent 工作流
            mutation_rate: 提示词变异率
            selection_pressure: 选择压力
            crossover_rate: 交叉率
            elite_ratio: 精英保留比例
            fixed_prompt: 是否固定提示词（基线模式，不做进化）
            initial_prompt: 初始提示词（None 时随机生成）
        """
        self.problem = problem
        self.size = size
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.fixed_prompt = fixed_prompt
        self.workflow = workflow
        self.rng = np.random.default_rng(seed)
        self.individuals = self._init_individuals(initial_prompt)
        self.generation = 0
        self.best_history: list[float] = []

    def _init_individuals(
        self, initial_prompt: EvolvablePrompt | None
    ) -> list[AgentIndividual]:
        if initial_prompt is None and self.fixed_prompt:
            from evoagent.core.genome_prompt import default_prompt

            initial_prompt = default_prompt()
        inds = []
        for i in range(self.size):
            prompt = (
                initial_prompt.clone()
                if initial_prompt is not None
                else random_prompt(self.rng)
            )
            inds.append(AgentIndividual(agent_id=f"llm-{i}", genome=None, genome_prompt=prompt))
        return inds

    # ------------------------------------------------------------ 评估

    def evaluate_all(self) -> None:
        """通过 Agent 工作流评估所有个体。"""
        for i, ind in enumerate(self.individuals):
            result_ind = self.workflow.run(
                ind.genome_prompt,
                seed=int(self.rng.integers(0, 2**31)),
            )
            ind.genome = result_ind.genome
            ind.fitness = result_ind.fitness
            ind.objectives = result_ind.objectives
            ind.best_params = result_ind.best_params
            ind.n_evals = result_ind.n_evals
            ind.n_improvements = result_ind.n_improvements
        self.best_history.append(self.best_individual().fitness)

    # ------------------------------------------------------------ 统计

    def best_individual(self) -> AgentIndividual:
        """返回适应度最高的个体。"""
        return max(self.individuals, key=lambda ind: ind.fitness)

    def mean_fitness(self) -> float:
        """种群平均适应度。"""
        return float(np.mean([ind.fitness for ind in self.individuals]))

    def best_prompt(self) -> EvolvablePrompt:
        """返回当前最优提示词基因。"""
        return self.best_individual().genome_prompt

    # ------------------------------------------------------------ 进化

    def next_generation(self) -> None:
        """选择 -> 提示词交叉/变异 -> 精英保留，推进一代。"""
        sorted_ind = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        n_elite = max(1, int(self.size * self.elite_ratio))
        n_parents = max(2, int(self.size * self.selection_pressure))
        n_children = self.size - n_elite

        if self.fixed_prompt:
            elite_prompt = sorted_ind[0].genome_prompt
            children = [self._new_individual(elite_prompt) for _ in range(n_children)]
        else:
            parents = self._tournament(sorted_ind, n_parents)
            children = []
            while len(children) < n_children:
                p1, p2 = parents[self.rng.integers(len(parents))], parents[
                    self.rng.integers(len(parents))
                ]
                if self.rng.random() < self.crossover_rate:
                    child_prompt = crossover_prompt(
                        p1.genome_prompt, p2.genome_prompt, rng=self.rng
                    )
                else:
                    child_prompt = p1.genome_prompt.clone()
                mutate_prompt(child_prompt, self.mutation_rate, self.rng)
                children.append(
                    AgentIndividual(
                        agent_id=f"llm-c{self.generation}-{len(children)}",
                        genome=None,
                        genome_prompt=child_prompt,
                    )
                )

        self.individuals = sorted_ind[:n_elite] + children[:n_children]
        self.generation += 1

    def _new_individual(self, prompt: EvolvablePrompt) -> AgentIndividual:
        """用固定提示词创建新个体（基线模式）。"""
        return AgentIndividual(
            agent_id=f"llm-f{self.generation}-{self.rng.integers(10000)}",
            genome=None,
            genome_prompt=prompt.clone(),
        )

    def _tournament(
        self, sorted_ind: list[AgentIndividual], k: int
    ) -> list[AgentIndividual]:
        """锦标赛选择父代。"""
        selected: list[AgentIndividual] = []
        for _ in range(k):
            a, b = self.rng.choice(len(sorted_ind), 2, replace=False)
            selected.append(sorted_ind[a] if sorted_ind[a].fitness >= sorted_ind[b].fitness else sorted_ind[b])
        return selected
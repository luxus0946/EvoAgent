"""LLM agent population: evolves evolvable prompt genomes (phase 2) + SEW dual modes (phase 3, step 4).

Differences from StrategyPopulation:
- individuals carry EvolvablePrompt genomes (role/thinking style/tool preference/exploration bias, etc.)
- evaluation goes through AgentWorkflow: the LLM generates and executes a strategy from the prompt
- crossover/mutation act on prompt genomes; evolution drives self-improvement of the prompts

SEW dual modes (inspired by EvoAgentX): the population can simultaneously contain
- prompt-mode individuals: genome is EvolvablePrompt, evolving prompts
- structure-mode individuals: genome is StrategyGenome, evolving the strategy structure itself (EoH operators)
Both modes share the same LLM workflow and evolve according to their own genomes.
"""

import numpy as np

from evoagent.agent.workflow import AgentWorkflow
from evoagent.core.genome_prompt import (
    EvolvablePrompt,
    crossover_prompt,
    mutate_prompt,
    random_prompt,
)
from evoagent.core.individual import AgentIndividual, random_genome
from evoagent.environment.problem import OptimizationProblem
from evoagent.evolution.operators import crossover_uniform, mutate_individual_eoh
from evoagent.evolution.population import Population


class LlmPopulation:
    """LLM agent population: evolves prompt genomes."""

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
        sew_ratio: float = 0.0,
    ):
        """Initialize.

        Args:
            problem: optimization problem
            size: population size
            seed: random seed
            workflow: agent workflow
            mutation_rate: prompt mutation rate
            selection_pressure: selection pressure
            crossover_rate: crossover rate
            elite_ratio: elite retention ratio
            fixed_prompt: whether to fix the prompt (baseline mode, no evolution)
            initial_prompt: initial prompt (randomly generated when None)
            sew_ratio: share of structure-mode individuals in SEW dual mode (0 means pure prompt mode)
        """
        self.problem = problem
        self.size = size
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.fixed_prompt = fixed_prompt
        self.sew_ratio = 0.0 if fixed_prompt else sew_ratio
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
        n_structure = int(self.size * self.sew_ratio)
        for i in range(self.size):
            if i < n_structure:
                ind = AgentIndividual(
                    agent_id=f"llm-s{i}",
                    genome=random_genome(self.rng, self.problem.n_objectives),
                    genome_prompt=None,
                    mode="structure",
                )
            else:
                prompt = (
                    initial_prompt.clone()
                    if initial_prompt is not None
                    else random_prompt(self.rng)
                )
                ind = AgentIndividual(
                    agent_id=f"llm-{i}",
                    genome=None,
                    genome_prompt=prompt,
                    mode="prompt",
                )
            inds.append(ind)
        return inds

    # ------------------------------------------------------------ evaluation

    def evaluate_all(self) -> None:
        """Evaluate all individuals via the agent workflow (structure mode goes through the neutral prompt channel)."""
        for i, ind in enumerate(self.individuals):
            if ind.mode == "structure":
                result_ind = self.workflow.run(
                    None, seed=int(self.rng.integers(0, 2**31)), mode="structure"
                )
            else:
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

    # ------------------------------------------------------------ statistics

    def best_individual(self) -> AgentIndividual:
        """Return the individual with the highest fitness."""
        return max(self.individuals, key=lambda ind: ind.fitness)

    def mean_fitness(self) -> float:
        """Mean population fitness."""
        return float(np.mean([ind.fitness for ind in self.individuals]))

    def best_prompt(self) -> EvolvablePrompt:
        """Return the current best prompt genome."""
        return self.best_individual().genome_prompt

    # ------------------------------------------------------------ evolution

    def next_generation(self) -> None:
        """Select, crossover/mutate each genome type, retain elites, and advance one generation."""
        sorted_ind = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        n_elite = max(1, int(self.size * self.elite_ratio))
        n_parents = max(2, int(self.size * self.selection_pressure))
        n_children = self.size - n_elite

        if self.fixed_prompt:
            elite_prompt = sorted_ind[0].genome_prompt
            children = [self._new_individual(elite_prompt) for _ in range(n_children)]
        else:
            structure_ind = [ind for ind in self.individuals if ind.mode == "structure"]
            prompt_ind = [ind for ind in self.individuals if ind.mode == "prompt"]
            n_child_s = int(n_children * len(structure_ind) / self.size)
            n_child_p = n_children - n_child_s
            if structure_ind and n_child_s < 1:
                n_child_s = 1
                n_child_p = max(0, n_child_p - 1)

            children: list[AgentIndividual] = []
            if structure_ind:
                parents_s = self._tournament(
                    sorted(structure_ind, key=lambda ind: ind.fitness, reverse=True),
                    n_parents,
                )
                children += self._breed_structure(parents_s, n_child_s)
            if prompt_ind:
                parents_p = self._tournament(
                    sorted(prompt_ind, key=lambda ind: ind.fitness, reverse=True),
                    n_parents,
                )
                children += self._breed_prompt(parents_p, n_child_p)

        self.individuals = sorted_ind[:n_elite] + children[:n_children]
        self.generation += 1

    def _breed_structure(
        self, parents: list[AgentIndividual], k: int
    ) -> list[AgentIndividual]:
        """Structure-mode breeding: strategy genome crossover + EoH operator-style mutation."""
        children: list[AgentIndividual] = []
        while len(children) < k:
            p1, p2 = parents[self.rng.integers(len(parents))], parents[
                self.rng.integers(len(parents))
            ]
            if self.rng.random() < self.crossover_rate:
                child = crossover_uniform(p1, p2, rng=self.rng)
            else:
                child = p1.clone()
            child = mutate_individual_eoh(child, self.mutation_rate, self.rng)
            child.genome_prompt = None
            child.mode = "structure"
            children.append(child)
        return children

    def _breed_prompt(
        self, parents: list[AgentIndividual], k: int
    ) -> list[AgentIndividual]:
        """Prompt-mode breeding: prompt genome crossover + mutation."""
        children: list[AgentIndividual] = []
        while len(children) < k:
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
                    mode="prompt",
                )
            )
        return children

    def _new_individual(self, prompt: EvolvablePrompt) -> AgentIndividual:
        """Create a new individual with a fixed prompt (baseline mode)."""
        return AgentIndividual(
            agent_id=f"llm-f{self.generation}-{self.rng.integers(10000)}",
            genome=None,
            genome_prompt=prompt.clone(),
        )

    def _tournament(
        self, sorted_ind: list[AgentIndividual], k: int
    ) -> list[AgentIndividual]:
        """Tournament selection of parents (falls back to repeated selection when only one individual remains in the group)."""
        if len(sorted_ind) < 2:
            return list(sorted_ind) * k
        selected: list[AgentIndividual] = []
        for _ in range(k):
            a, b = self.rng.choice(len(sorted_ind), 2, replace=False)
            selected.append(sorted_ind[a] if sorted_ind[a].fitness >= sorted_ind[b].fitness else sorted_ind[b])
        return selected
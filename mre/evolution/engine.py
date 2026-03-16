"""
mre.evolution.engine
─────────────────────
EvolutionEngine — drives one generation of evolution:

  1. Select survivors (via SelectionEngine / Elo)
  2. Breed new offspring from the top-K parents (crossover)
  3. Mutate a fraction of offspring
  4. Replenish the population back to the target size
  5. Reset pipeline caches so fresh pipelines are built next round

Usage
-----
    from mre.evolution.engine import EvolutionEngine
    from mre.evolution.selection import SelectionEngine

    selector = SelectionEngine(cull_fraction=0.25)
    engine   = EvolutionEngine(selector=selector, target_population=6)

    # After one TaskManager round:
    new_population = engine.evolve(population, scores)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from mre.agents.dna import AgentDNA
from mre.evolution.selection import SelectionEngine
from mre.utils import get_logger

logger = get_logger(__name__)


# ── Generation summary ────────────────────────────────────────────────────────

@dataclass
class GenerationSummary:
    generation: int
    population_before: int
    survivors: int
    culled: int
    offspring: int
    mutants: int
    population_after: int
    best_elo: float
    mean_score: float
    best_score: float
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"╔══ Generation {self.generation:03d} ═══════════════════════════════════════╗",
            f"  Population : {self.population_before} → {self.population_after}",
            f"  Survivors  : {self.survivors}  |  Culled: {self.culled}",
            f"  New agents : {self.offspring} offspring + {self.mutants} mutants",
            f"  Scores     : best={self.best_score:.4f}  mean={self.mean_score:.4f}",
            f"  Best Elo   : {self.best_elo:.1f}",
        ]
        for note in self.notes:
            lines.append(f"  ↳ {note}")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ── Evolution engine ──────────────────────────────────────────────────────────

class EvolutionEngine:
    """
    Drives one evolution cycle: select → crossover → mutate → replenish.

    Parameters
    ----------
    selector : SelectionEngine
        Handles Elo updates and survival selection.
    target_population : int
        Desired population size after replenishment (default 6).
    mutation_rate : float
        Per-gene mutation probability (default 0.25).
    elite_fraction : float
        Fraction of top agents that survive unmutated as elites (default 0.33).
    rng : random.Random, optional
        For reproducible experiments.
    """

    def __init__(
        self,
        selector: Optional[SelectionEngine] = None,
        target_population: int = 6,
        mutation_rate: float = 0.25,
        elite_fraction: float = 0.33,
        rng: Optional[random.Random] = None,
    ):
        self.selector          = selector or SelectionEngine()
        self.target_population = target_population
        self.mutation_rate     = mutation_rate
        self.elite_fraction    = elite_fraction
        self.rng               = rng or random.Random()
        self._generation       = 0

    # ── Main entry ────────────────────────────────────────────────────────────

    def evolve(
        self,
        population: List[AgentDNA],
        scores: List[float],
    ) -> List[AgentDNA]:
        """
        Run one generation of evolution and return the new population.

        Parameters
        ----------
        population : List[AgentDNA]
            Current population (will not be mutated in-place).
        scores : List[float]
            Fitness scores aligned with *population*.

        Returns
        -------
        List[AgentDNA]
            New population of size ``target_population``.
        """
        self._generation += 1
        notes: List[str] = []

        if not population:
            self._generation += 1
            return []

        mean_score = sum(scores) / len(scores) if scores else 0.0
        best_score = max(scores) if scores else 0.0

        # ── Step 1: update Elo ratings ───────────────────────────────────────
        self.selector.update(population, scores)

        # ── Step 2: survival selection ────────────────────────────────────────
        survivors, culled = self.selector.select(population)
        notes.append(
            f"Culled agents: {[d.agent_id for d in culled]}"
        )

        # ── Step 3: elites pass through unmutated ─────────────────────────────
        n_elites = max(1, int(len(survivors) * self.elite_fraction))
        elites   = self.selector.top_k(survivors, k=n_elites)
        non_elites = [d for d in survivors if d not in elites]

        new_pop: List[AgentDNA] = list(elites)

        # ── Step 4: breed offspring via crossover ─────────────────────────────
        parents = self.selector.top_k(survivors, k=min(4, len(survivors)))
        n_breed = max(0, self.target_population - len(new_pop))
        offspring_count = 0

        while len(new_pop) < self.target_population - max(0, len(culled)):
            if len(parents) >= 2:
                p_a, p_b = self.rng.sample(parents, 2)
                child_a, child_b = AgentDNA.crossover(p_a, p_b, rng=self.rng)
                new_pop.append(child_a)
                offspring_count += 1
                if len(new_pop) < self.target_population:
                    new_pop.append(child_b)
                    offspring_count += 1
            else:
                # Only one parent — clone + mutate
                child = parents[0].clone()
                child = child.mutate(rng=self.rng, mutation_rate=self.mutation_rate)
                new_pop.append(child)
                offspring_count += 1

            if offspring_count > self.target_population * 3:
                break  # safety valve

        # ── Step 5: mutate non-elite survivors ────────────────────────────────
        mutant_count = 0
        for i, dna in enumerate(new_pop):
            if dna in elites:
                continue  # elites are untouched
            if self.rng.random() < self.mutation_rate:
                new_pop[i] = dna.mutate(rng=self.rng, mutation_rate=self.mutation_rate)
                mutant_count += 1

        # ── Step 6: replenish to target size with fresh seeds if needed ────────
        while len(new_pop) < self.target_population:
            seed = AgentDNA()
            seed.generation = self._generation
            new_pop.append(seed)
            notes.append(f"Seeded fresh agent {seed.agent_id}.")

        # Trim to exact target
        new_pop = new_pop[: self.target_population]

        # ── Step 7: invalidate cached pipelines ───────────────────────────────
        for dna in new_pop:
            dna.metadata.pop("_pipeline_cache", None)

        best_elo = self.selector.elo(
            self.selector.top_k(survivors, k=1)[0].agent_id
        ) if survivors else self.selector.initial_elo

        summary = GenerationSummary(
            generation=self._generation,
            population_before=len(population),
            survivors=len(survivors),
            culled=len(culled),
            offspring=offspring_count,
            mutants=mutant_count,
            population_after=len(new_pop),
            best_elo=best_elo,
            mean_score=round(mean_score, 4),
            best_score=round(best_score, 4),
            notes=notes,
        )
        self._last_summary = summary
        logger.info(summary.report().split("\n")[0])  # log first line
        return new_pop

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def last_summary(self) -> Optional[GenerationSummary]:
        return getattr(self, "_last_summary", None)

    @property
    def generation(self) -> int:
        return self._generation

    def __repr__(self) -> str:
        return (
            f"EvolutionEngine(gen={self._generation}, "
            f"target={self.target_population}, "
            f"mutation={self.mutation_rate:.0%})"
        )

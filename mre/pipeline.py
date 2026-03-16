"""
mre.pipeline
─────────────
MREPipeline — the closed-loop "reason → evaluate → evolve" engine.

This is the top-level orchestrator for Phase 3.  It wires together:

  ReasoningAgentPool   (Phase 2)  — generates candidate solutions
  EvaluationCommission (Phase 3)  — scores each solution (4-dimensional)
  SelectionEngine      (Phase 3)  — updates Elo, culls weak agents
  EvolutionEngine      (Phase 3)  — crossover + mutate → next generation

Usage
-----
    from mre.pipeline import MREPipeline
    from mre.agents.dna import AgentDNA

    pipeline = MREPipeline(
        population_size=6,
        generations=4,
        target_score=0.90,
    )
    history = pipeline.run(
        problems=[
            {"text": "Solve x**2 - 5*x + 6 = 0",
             "context": {"equation": "x**2 - 5*x + 6 = 0"},
             "answer": "2"},
        ],
        seed_population=[AgentDNA()],
    )
    pipeline.print_history(history)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mre.agents.dna import AgentDNA
from mre.agents.agent import ReasoningAgentPool
from mre.agents.state import ReasoningState
from mre.evaluation.commission import EvaluationCommission, CommissionVerdict
from mre.evolution.selection import SelectionEngine
from mre.evolution.engine import EvolutionEngine, GenerationSummary
from mre.operators.stats import OperatorStats
from mre.utils import get_logger

logger = get_logger(__name__)


# ── Problem spec ──────────────────────────────────────────────────────────────

@dataclass
class Problem:
    text: str
    context: Dict[str, Any] = field(default_factory=dict)
    expected_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Problem":
        return cls(
            text=d["text"],
            context=d.get("context", {}),
            expected_answer=d.get("answer") or d.get("expected_answer"),
        )


# ── Per-generation record ────────────────────────────────────────────────────

@dataclass
class GenerationRecord:
    generation: int
    problem_text: str
    verdicts: List[CommissionVerdict]
    evo_summary: Optional[GenerationSummary]
    best_score: float
    best_answer: Optional[str]
    duration_sec: float

    def summary_line(self) -> str:
        return (
            f"Gen {self.generation:03d} | "
            f"best={self.best_score:.4f} | "
            f"answer={self.best_answer!r} | "
            f"{len(self.verdicts)} agents | "
            f"{self.duration_sec:.2f}s"
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MREPipeline:
    """
    Closed-loop Reason → Evaluate → Evolve pipeline.

    Parameters
    ----------
    population_size : int
        Target number of agents in each generation (default 6).
    generations : int
        Maximum evolution rounds per problem (default 5).
    target_score : float
        Stop early if any agent achieves this commission score (default 0.90).
    mutation_rate : float
        Per-gene mutation probability (default 0.25).
    cull_fraction : float
        Fraction of the bottom population culled per generation (default 0.20).
    llm_client : callable, optional
        ``(prompt: str) -> str`` for LLM-powered operators and critique.
    """

    def __init__(
        self,
        population_size: int = 6,
        generations: int = 5,
        target_score: float = 0.90,
        mutation_rate: float = 0.25,
        cull_fraction: float = 0.20,
        llm_client: Optional[Callable] = None,
    ):
        self.population_size = population_size
        self.generations     = generations
        self.target_score    = target_score
        self.mutation_rate   = mutation_rate
        self.llm_client      = llm_client

        self.commission  = EvaluationCommission(llm_client=llm_client)
        self.selector    = SelectionEngine(cull_fraction=cull_fraction)
        self.evo_engine  = EvolutionEngine(
            selector=self.selector,
            target_population=population_size,
            mutation_rate=mutation_rate,
        )
        self.op_stats = OperatorStats()

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(
        self,
        problems: List[Dict[str, Any]],
        seed_population: Optional[List[AgentDNA]] = None,
    ) -> List[GenerationRecord]:
        """
        Run the full closed loop across all problems and generations.

        Each problem is attempted by the full population.
        After each problem, the population evolves based on commission scores.

        Returns
        -------
        List[GenerationRecord]
            One record per (problem × generation) pair.
        """
        # Initialise population
        population = seed_population or [AgentDNA() for _ in range(self.population_size)]
        while len(population) < self.population_size:
            population.append(AgentDNA())
        population = population[: self.population_size]

        parsed_problems = [Problem.from_dict(p) for p in problems]
        history: List[GenerationRecord] = []

        for gen in range(1, self.generations + 1):
            logger.info("═══ Generation %d / %d ═══", gen, self.generations)

            # Pick a problem (round-robin across problems)
            problem = parsed_problems[(gen - 1) % len(parsed_problems)]
            logger.info("Problem: %s", problem.text[:60])

            t0 = time.perf_counter()

            # ── Reason ───────────────────────────────────────────────────────
            pool = ReasoningAgentPool.from_dna_list(
                population, llm_client=self.llm_client
            )
            results = pool.solve_all(problem.text, context=problem.context)

            # ── Evaluate ─────────────────────────────────────────────────────
            verdicts = self.commission.batch_evaluate(
                results,
                expected_answers=[problem.expected_answer] * len(results),
            )
            scores = [v.weighted_score for v in verdicts]

            # Track operator stats
            for r in results:
                self.op_stats.record(r)

            best_idx    = scores.index(max(scores)) if scores else 0
            best_score  = scores[best_idx] if scores else 0.0
            best_answer = results[best_idx].answer if results else None

            logger.info(
                "Gen %d evaluation: best=%.4f  mean=%.4f",
                gen, best_score,
                sum(scores) / len(scores) if scores else 0.0,
            )

            # ── Evolve ────────────────────────────────────────────────────────
            evo_summary: Optional[GenerationSummary] = None
            if gen < self.generations:
                population = self.evo_engine.evolve(population, scores)
                evo_summary = self.evo_engine.last_summary

            duration = time.perf_counter() - t0

            record = GenerationRecord(
                generation=gen,
                problem_text=problem.text,
                verdicts=verdicts,
                evo_summary=evo_summary,
                best_score=best_score,
                best_answer=best_answer,
                duration_sec=round(duration, 3),
            )
            history.append(record)

            logger.info(record.summary_line())

            # Early termination
            if best_score >= self.target_score:
                logger.info(
                    "Target score %.2f reached at generation %d — stopping.",
                    self.target_score, gen,
                )
                break

        return history

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_history(self, history: List[GenerationRecord]) -> None:
        print("╔══ MRE Pipeline Run History ══════════════════════════════╗")
        for rec in history:
            print(f"  {rec.summary_line()}")
            best_v = max(rec.verdicts, key=lambda v: v.weighted_score)
            print(f"    correctness={best_v.correctness:.3f}  "
                  f"logic={best_v.logic:.3f}  "
                  f"critique={best_v.critique:.3f}  "
                  f"concise={best_v.conciseness:.3f}")
        print("  ── Operator Stats ──────────────────────────────────────")
        print("  " + self.op_stats.leaderboard().replace("\n", "\n  "))
        print("  ── Elo Leaderboard ─────────────────────────────────────")
        # show final population
        if history:
            dummy_pop = [AgentDNA(agent_id=aid)
                         for aid in self.selector._ratings]
            print("  " + self.selector.leaderboard(dummy_pop).replace("\n", "\n  "))
        print("╚═══════════════════════════════════════════════════════════╝")

    def convergence_curve(self, history: List[GenerationRecord]) -> List[float]:
        """Return list of best scores per generation for plotting."""
        return [r.best_score for r in history]

    def __repr__(self) -> str:
        return (
            f"MREPipeline(pop={self.population_size}, "
            f"gen={self.generations}, "
            f"target={self.target_score})"
        )

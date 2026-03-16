"""
mre.agents.task_manager
────────────────────────
TaskManager — orchestrates the reasoning → evaluation → evolution loop.

Phase-2 scope: coordinates a pool of agents on a single problem,
collects results, scores them with a lightweight built-in judge,
and prepares the fitness data that Phase-3's EvolutionEngine will
consume.

Phase-3 will plug in the full EvaluationCommission and EvolutionEngine;
for now the manager exposes clean hooks (``on_round_complete``) that
those components will attach to.

Usage
-----
    from mre.agents.task_manager import TaskManager
    from mre.agents.dna import AgentDNA

    population = [AgentDNA() for _ in range(4)]
    manager = TaskManager(population=population, max_rounds=2)
    report = manager.run("Solve x**2 - 5*x + 6 = 0",
                         context={"equation": "x**2 - 5*x + 6 = 0"})
    print(report.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mre.agents.agent import ReasoningAgent, ReasoningAgentPool
from mre.agents.dna import AgentDNA
from mre.agents.state import ReasoningState
from mre.operators.pipeline import PipelineResult
from mre.operators.stats import OperatorStats
from mre.utils import get_logger

logger = get_logger(__name__)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _default_scorer(result: PipelineResult, expected_answer: Optional[str] = None) -> float:
    """
    Lightweight built-in scorer (no LLM required).

    Scoring dimensions (matches Phase-3 design):
      Correctness  40% — answer present and matches expected (if given)
      Logic        30% — no failed steps in history
      Efficiency   20% — inverse of step count (fewer steps = better)
      Confidence   10% — agent's self-reported confidence
    """
    state = result.final_state

    # Correctness
    if expected_answer is not None:
        correct = (
            state.answer is not None
            and str(expected_answer).strip() in str(state.answer)
        )
        correctness = 1.0 if correct else 0.0
    else:
        correctness = 1.0 if state.is_solved else 0.0

    # Logic (no failed steps)
    failed_steps = sum(1 for r in state.history if not r.success)
    logic = max(0.0, 1.0 - failed_steps * 0.25)

    # Efficiency
    n_steps = max(1, len(state.history))
    efficiency = 1.0 / n_steps  # simple inverse

    # Confidence
    confidence = state.confidence

    score = (
        0.40 * correctness
        + 0.30 * logic
        + 0.20 * efficiency
        + 0.10 * confidence
    )
    return round(score, 4)


# ── Round result ─────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    round_num: int
    agent_results: List[PipelineResult]
    scores: List[float]
    best_score: float
    best_answer: Optional[str]
    duration_sec: float

    def summary(self) -> str:
        lines = [
            f"Round {self.round_num:02d} | "
            f"best={self.best_score:.4f} | "
            f"answer={self.best_answer!r} | "
            f"{len(self.agent_results)} agents | "
            f"{self.duration_sec:.2f}s"
        ]
        for i, (r, s) in enumerate(zip(self.agent_results, self.scores)):
            lines.append(
                f"  Agent[{i}] score={s:.4f}  "
                f"solved={r.solved}  steps={len(r.final_state.history)}"
            )
        return "\n".join(lines)


# ── Task report ───────────────────────────────────────────────────────────────

@dataclass
class TaskReport:
    problem: str
    rounds: List[RoundResult]
    final_answer: Optional[str]
    total_duration_sec: float
    operator_stats: OperatorStats
    population: List[AgentDNA]

    def summary(self) -> str:
        lines = [
            "╔══ TaskManager Report ═══════════════════════════════════╗",
            f"  Problem : {self.problem[:70]}",
            f"  Rounds  : {len(self.rounds)}",
            f"  Answer  : {self.final_answer}",
            f"  Time    : {self.total_duration_sec:.2f}s",
            "  ── Rounds ──────────────────────────────────────────────",
        ]
        for rr in self.rounds:
            lines.append("  " + rr.summary().replace("\n", "\n  "))
        lines.append("  ── Operator Stats ──────────────────────────────────────")
        lines.append("  " + self.operator_stats.leaderboard().replace("\n", "\n  "))
        lines.append("╚═════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ── TaskManager ───────────────────────────────────────────────────────────────

class TaskManager:
    """
    Coordinates a population of agents across multiple reasoning rounds.

    Parameters
    ----------
    population : List[AgentDNA]
        Seed agent blueprints.
    max_rounds : int
        Maximum number of evaluation rounds.
    target_score : float
        Stop early if any agent achieves this score.
    scorer : callable, optional
        ``(result, expected_answer) -> float`` scoring function.
        Defaults to the built-in weighted scorer.
    llm_client : callable, optional
        Passed to SelfCritique operators.
    on_round_complete : callable, optional
        Hook called after every round with ``(round_result, population)``.
        Phase-3 EvolutionEngine attaches here.
    """

    def __init__(
        self,
        population: Optional[List[AgentDNA]] = None,
        max_rounds: int = 3,
        target_score: float = 0.85,
        scorer: Optional[Callable] = None,
        llm_client: Optional[Callable] = None,
        on_round_complete: Optional[Callable] = None,
    ):
        self.population: List[AgentDNA] = population or [AgentDNA()]
        self.max_rounds = max_rounds
        self.target_score = target_score
        self.scorer = scorer or _default_scorer
        self.llm_client = llm_client
        self.on_round_complete = on_round_complete
        self._stats = OperatorStats()

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        expected_answer: Optional[str] = None,
    ) -> TaskReport:
        """
        Run up to ``max_rounds`` rounds of reasoning on *problem*.

        Returns a TaskReport with per-round results and aggregate stats.
        """
        t0 = time.perf_counter()
        rounds: List[RoundResult] = []
        final_answer: Optional[str] = None

        for round_num in range(1, self.max_rounds + 1):
            logger.info("Round %d/%d — %d agents", round_num, self.max_rounds,
                        len(self.population))

            rr = self._run_round(round_num, problem, context, expected_answer)
            rounds.append(rr)

            # Record stats
            for result in rr.agent_results:
                self._stats.record(result)

            # Write fitness back into DNA
            for dna, score in zip(self.population, rr.scores):
                dna.fitness_score = score

            if rr.best_answer:
                final_answer = rr.best_answer

            logger.info("Round %d best score: %.4f", round_num, rr.best_score)

            # Hook for Phase-3 evolution
            if self.on_round_complete:
                self.on_round_complete(rr, self.population)

            # Early termination
            if rr.best_score >= self.target_score:
                logger.info("Target score %.2f reached — stopping early.", self.target_score)
                break

        total = time.perf_counter() - t0
        return TaskReport(
            problem=problem,
            rounds=rounds,
            final_answer=final_answer,
            total_duration_sec=round(total, 3),
            operator_stats=self._stats,
            population=self.population,
        )

    # ── Round execution ───────────────────────────────────────────────────────

    def _run_round(
        self,
        round_num: int,
        problem: str,
        context: Optional[Dict[str, Any]],
        expected_answer: Optional[str],
    ) -> RoundResult:
        t0 = time.perf_counter()
        pool = ReasoningAgentPool.from_dna_list(
            self.population, llm_client=self.llm_client
        )
        agent_results = pool.solve_all(problem, context=context)

        scores = [
            self.scorer(r, expected_answer) for r in agent_results
        ]

        best_idx = scores.index(max(scores)) if scores else 0
        best_score = scores[best_idx] if scores else 0.0
        best_answer = (
            agent_results[best_idx].answer if agent_results else None
        )

        return RoundResult(
            round_num=round_num,
            agent_results=agent_results,
            scores=scores,
            best_score=best_score,
            best_answer=best_answer,
            duration_sec=round(time.perf_counter() - t0, 3),
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> OperatorStats:
        return self._stats

    def sorted_population(self) -> List[AgentDNA]:
        """Return population sorted by descending fitness score."""
        return sorted(
            self.population,
            key=lambda d: d.fitness_score or 0.0,
            reverse=True,
        )

    def __repr__(self) -> str:
        return (
            f"TaskManager(agents={len(self.population)}, "
            f"max_rounds={self.max_rounds})"
        )

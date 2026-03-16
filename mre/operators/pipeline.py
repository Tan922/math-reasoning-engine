"""
mre.operators.pipeline
───────────────────────
OperatorPipeline — executes a list of operator names in sequence.

Usage
-----
    from mre.operators.pipeline import OperatorPipeline
    from mre.agents.dna import AgentDNA
    from mre.agents.state import ReasoningState

    dna = AgentDNA()
    state = ReasoningState.from_problem("Solve x**2 - 5*x + 6 = 0")
    state = state.evolve(context={"equation": "x**2 - 5*x + 6 = 0"})

    pipeline = OperatorPipeline.from_dna(dna)
    result = pipeline.run(state)
    print(result.answer)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from mre.agents.dna import AgentDNA
from mre.agents.state import ReasoningState
from mre.operators.base import BaseOperator, get_operator, list_operators
from mre.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run."""
    final_state: ReasoningState
    operator_sequence: List[str]
    total_duration_sec: float
    solved: bool
    answer: Optional[str]
    step_timings: List[Tuple[str, float]] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "╔══ Pipeline Result ══════════════════════════════════╗",
            f"  Solved   : {self.solved}",
            f"  Answer   : {self.answer}",
            f"  Duration : {self.total_duration_sec:.3f}s",
            f"  Operators: {' → '.join(self.operator_sequence)}",
            "  ── Steps ──────────────────────────────────────────",
        ]
        for name, dur in self.step_timings:
            lines.append(f"    {name:<25} {dur:.4f}s")
        lines.append("╚══════════════════════════════════════════════════════╝")
        return "\n".join(lines)


class OperatorPipeline:
    """
    Sequential pipeline of reasoning operators.

    Parameters
    ----------
    operators : List[BaseOperator]
        Ordered list of operator instances to execute.
    stop_on_answer : bool
        If True, halt the pipeline as soon as state.is_solved is True.
    stop_on_failure : bool
        If True, halt the pipeline if state.failed is True.
    max_steps : int
        Hard cap on number of operator executions (safety valve).
    """

    def __init__(
        self,
        operators: List[BaseOperator],
        stop_on_answer: bool = True,
        stop_on_failure: bool = False,
        max_steps: int = 20,
    ):
        self.operators = operators
        self.stop_on_answer = stop_on_answer
        self.stop_on_failure = stop_on_failure
        self.max_steps = max_steps

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_dna(
        cls,
        dna: AgentDNA,
        llm_client: Optional[Callable] = None,
        **kwargs,
    ) -> "OperatorPipeline":
        """
        Instantiate a pipeline from an AgentDNA's reasoning_gene.

        Parameters
        ----------
        dna : AgentDNA
        llm_client : callable, optional
            Passed to SelfCritique if present in the gene sequence.
        """
        ops: List[BaseOperator] = []
        for op_name in dna.reasoning_gene:
            try:
                if op_name == "SelfCritique":
                    from mre.operators.library import SelfCritique
                    ops.append(SelfCritique(llm_client=llm_client))
                else:
                    ops.append(get_operator(op_name))
            except KeyError:
                logger.warning("Unknown operator %r — skipping.", op_name)
        return cls(operators=ops, **kwargs)

    @classmethod
    def from_names(cls, names: List[str], **kwargs) -> "OperatorPipeline":
        ops = [get_operator(n) for n in names]
        return cls(operators=ops, **kwargs)

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, state: ReasoningState) -> PipelineResult:
        """Execute the pipeline and return a PipelineResult."""
        t0 = time.perf_counter()
        step_timings: List[Tuple[str, float]] = []

        for i, op in enumerate(self.operators[:self.max_steps]):
            t_step = time.perf_counter()
            logger.debug("Step %d/%d: %s", i + 1, len(self.operators), op.name)

            state = op.apply(state)
            step_timings.append((op.name, time.perf_counter() - t_step))

            if self.stop_on_answer and state.is_solved:
                logger.debug("Pipeline halted early: answer found after %d steps.", i + 1)
                break
            if self.stop_on_failure and state.failed:
                logger.debug("Pipeline halted: failure in %s.", op.name)
                break

        total = time.perf_counter() - t0
        return PipelineResult(
            final_state=state,
            operator_sequence=[op.name for op in self.operators],
            total_duration_sec=round(total, 4),
            solved=state.is_solved,
            answer=state.answer,
            step_timings=step_timings,
        )

    def __repr__(self) -> str:
        names = " → ".join(op.name for op in self.operators)
        return f"OperatorPipeline([{names}])"

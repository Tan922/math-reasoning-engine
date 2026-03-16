"""
mre.agents.state
────────────────
ReasoningState — the immutable (copy-on-write) carrier object that flows
through the operator pipeline.  Every operator receives a ReasoningState
and returns a *new* ReasoningState, never mutating the original.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """Records one operator's execution."""
    operator_name: str
    input_snapshot: str          # brief human-readable description of input
    output_snapshot: str         # brief human-readable description of output
    success: bool = True
    error_msg: str = ""
    duration_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningState:
    """
    The single unit of state passed between operators.

    Attributes
    ----------
    problem : str
        The original problem statement (never mutated).
    current_expression : str
        The main symbolic / textual expression being worked on.
    context : Dict[str, Any]
        Shared scratchpad for arbitrary intermediate values.
    history : List[StepRecord]
        Append-only log of every operator that has run.
    answer : Optional[str]
        Final answer once a terminal operator decides we're done.
    confidence : float
        Estimated confidence in the current state [0, 1].
    failed : bool
        Set by an operator to signal unrecoverable failure.
    failure_reason : str
        Human-readable reason if failed == True.
    """

    problem: str
    current_expression: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[StepRecord] = field(default_factory=list)
    answer: Optional[str] = None
    confidence: float = 1.0
    failed: bool = False
    failure_reason: str = ""

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_problem(cls, problem: str) -> "ReasoningState":
        return cls(problem=problem, current_expression=problem)

    # ── Copy-on-write helpers ─────────────────────────────────────────────────

    def evolve(self, **kwargs) -> "ReasoningState":
        """Return a *new* state with the given fields overridden."""
        new = copy.deepcopy(self)
        for k, v in kwargs.items():
            object.__setattr__(new, k, v)
        return new

    def add_step(self, record: StepRecord) -> "ReasoningState":
        new_history = list(self.history) + [record]
        return self.evolve(history=new_history)

    def set_answer(self, answer: str, confidence: float = 1.0) -> "ReasoningState":
        return self.evolve(answer=answer, confidence=confidence)

    def mark_failed(self, reason: str) -> "ReasoningState":
        return self.evolve(failed=True, failure_reason=reason)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_solved(self) -> bool:
        return self.answer is not None and not self.failed

    def summary(self) -> str:
        lines = [
            f"Problem : {self.problem[:80]}",
            f"Current : {self.current_expression[:80]}",
            f"Steps   : {len(self.history)}",
            f"Answer  : {self.answer}",
            f"Failed  : {self.failed}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ReasoningState(solved={self.is_solved}, "
            f"steps={len(self.history)}, "
            f"expr={self.current_expression[:40]!r})"
        )

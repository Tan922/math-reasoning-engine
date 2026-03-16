"""
mre.operators.base
──────────────────
BaseOperator — abstract base class every reasoning operator must implement.

Contract
--------
Every operator:
  1. Exposes a unique ``name`` class attribute.
  2. Implements ``apply(state) -> ReasoningState``.
  3. Returns a *new* ReasoningState (never mutates the input).
  4. Appends exactly one ``StepRecord`` to ``state.history`` via
     ``state.add_step()``.
  5. On unrecoverable error, returns ``state.mark_failed(reason)``
     rather than raising.

Registry
--------
All concrete subclasses auto-register into ``OPERATOR_REGISTRY`` so
the agent runtime can instantiate operators by name from the
reasoning_gene list.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Type

from mre.agents.state import ReasoningState, StepRecord


# ── Global registry ───────────────────────────────────────────────────────────

OPERATOR_REGISTRY: Dict[str, Type["BaseOperator"]] = {}


class _OperatorMeta(type(ABC)):
    """Metaclass that auto-registers every concrete BaseOperator subclass."""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases and name != "BaseOperator":
            key = getattr(cls, "name", name)
            OPERATOR_REGISTRY[key] = cls
        return cls


# ── Base class ────────────────────────────────────────────────────────────────

class BaseOperator(ABC, metaclass=_OperatorMeta):
    """
    Abstract base for all reasoning operators.

    Subclasses must:
      - Set a class-level ``name: ClassVar[str]`` attribute.
      - Implement ``_run(state) -> ReasoningState``.

    The public ``apply()`` method wraps ``_run()`` with timing,
    error handling, and history recording.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    # ── Public entry point ────────────────────────────────────────────────────

    def apply(self, state: ReasoningState) -> ReasoningState:
        """
        Execute this operator on *state*.

        Returns a new ReasoningState with one additional StepRecord
        appended to history.
        """
        if state.failed:
            return state  # propagate failure without running

        t0 = time.perf_counter()
        try:
            new_state = self._run(state)
            duration = time.perf_counter() - t0
            record = StepRecord(
                operator_name=self.name,
                input_snapshot=state.current_expression[:120],
                output_snapshot=new_state.current_expression[:120],
                success=not new_state.failed,
                error_msg=new_state.failure_reason if new_state.failed else "",
                duration_sec=round(duration, 4),
            )
            return new_state.add_step(record)
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - t0
            record = StepRecord(
                operator_name=self.name,
                input_snapshot=state.current_expression[:120],
                output_snapshot="",
                success=False,
                error_msg=str(exc),
                duration_sec=round(duration, 4),
            )
            failed_state = state.mark_failed(f"{self.name} raised: {exc}")
            return failed_state.add_step(record)

    # ── Abstract implementation ───────────────────────────────────────────────

    @abstractmethod
    def _run(self, state: ReasoningState) -> ReasoningState:
        """Core logic — override in every subclass."""

    # ── Utility ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ── Registry helpers ──────────────────────────────────────────────────────────

def get_operator(name: str) -> BaseOperator:
    """Instantiate an operator by its registry name."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator {name!r}. Available: {list(OPERATOR_REGISTRY)}")
    return OPERATOR_REGISTRY[name]()


def list_operators() -> Dict[str, str]:
    """Return {name: description} for all registered operators."""
    return {n: cls.description for n, cls in OPERATOR_REGISTRY.items()}

"""
mre.operators.stats
────────────────────
OperatorStats — tracks per-operator performance over time.

This is the data backbone for operator evolution (Phase 3):
the system observes which operators succeed most often and
at what cost, then promotes high-performing sequences into
composite "macro-operators".

Usage
-----
    from mre.operators.stats import OperatorStats
    from mre.operators.pipeline import PipelineResult

    tracker = OperatorStats()
    tracker.record(result)
    print(tracker.leaderboard())
    combos = tracker.top_sequences(min_length=2)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mre.operators.pipeline import PipelineResult


# ── Per-operator stats ────────────────────────────────────────────────────────

@dataclass
class OpStats:
    name: str
    calls: int = 0
    successes: int = 0
    total_duration: float = 0.0
    total_confidence_gain: float = 0.0   # sum of (conf_after - conf_before) when positive

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.calls if self.calls else 0.0

    @property
    def avg_confidence_gain(self) -> float:
        return self.total_confidence_gain / self.calls if self.calls else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "calls": self.calls,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "avg_duration_sec": round(self.avg_duration, 5),
            "avg_confidence_gain": round(self.avg_confidence_gain, 4),
        }


# ── Sequence tracker ──────────────────────────────────────────────────────────

@dataclass
class SeqStats:
    sequence: Tuple[str, ...]
    uses: int = 0
    solved_count: int = 0

    @property
    def solve_rate(self) -> float:
        return self.solved_count / self.uses if self.uses else 0.0

    def score(self) -> float:
        """Composite score: solve_rate weighted by log(uses+1) to penalise rare sequences."""
        import math
        return self.solve_rate * math.log(self.uses + 1)


# ── Main tracker ──────────────────────────────────────────────────────────────

class OperatorStats:
    """
    Aggregate operator and sequence statistics across many pipeline runs.

    Thread-safety: not thread-safe; use a lock if running parallel agents.
    """

    def __init__(self):
        self._ops: Dict[str, OpStats] = {}
        self._seqs: Dict[Tuple[str, ...], SeqStats] = {}
        self._total_runs: int = 0
        self._solved_runs: int = 0

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, result: PipelineResult) -> None:
        """Ingest one PipelineResult and update all counters."""
        self._total_runs += 1
        if result.solved:
            self._solved_runs += 1

        state = result.final_state
        confidence_values: List[float] = []

        for step in state.history:
            op_name = step.operator_name
            if op_name not in self._ops:
                self._ops[op_name] = OpStats(name=op_name)
            stats = self._ops[op_name]
            stats.calls += 1
            stats.total_duration += step.duration_sec
            if step.success:
                stats.successes += 1
            confidence_values.append(state.confidence)

        # Sequence tracking (all contiguous sub-sequences of length 2..n)
        seq = tuple(s.operator_name for s in state.history)
        n = len(seq)
        for length in range(2, min(n + 1, 6)):
            for start in range(n - length + 1):
                sub = seq[start : start + length]
                if sub not in self._seqs:
                    self._seqs[sub] = SeqStats(sequence=sub)
                self._seqs[sub].uses += 1
                if result.solved:
                    self._seqs[sub].solved_count += 1

    # ── Queries ──────────────────────────────────────────────────────────────

    def leaderboard(self, top_k: int = 10) -> str:
        """Return a formatted leaderboard of operators by success rate."""
        ranked = sorted(
            self._ops.values(),
            key=lambda s: (s.success_rate, s.calls),
            reverse=True,
        )[:top_k]
        lines = [
            "╔══ Operator Leaderboard ════════════════════════════════════╗",
            f"  Total runs: {self._total_runs}  |  Solved: {self._solved_runs}",
            "  {:<28} {:>8} {:>8} {:>10}".format(
                "Operator", "Calls", "Success%", "AvgDur(s)"
            ),
            "  " + "─" * 58,
        ]
        for s in ranked:
            lines.append(
                "  {:<28} {:>8} {:>7.1f}% {:>10.4f}".format(
                    s.name, s.calls, s.success_rate * 100, s.avg_duration
                )
            )
        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def top_sequences(
        self,
        min_length: int = 2,
        min_uses: int = 2,
        top_k: int = 10,
    ) -> List[SeqStats]:
        """Return the highest-scoring operator sequences."""
        candidates = [
            s for s in self._seqs.values()
            if len(s.sequence) >= min_length and s.uses >= min_uses
        ]
        return sorted(candidates, key=lambda s: s.score(), reverse=True)[:top_k]

    def suggest_macro_operators(
        self,
        min_solve_rate: float = 0.7,
        min_uses: int = 3,
    ) -> List[Tuple[str, ...]]:
        """
        Return sequences that qualify as macro-operator candidates:
        high solve rate AND used enough times to be statistically meaningful.
        These can be promoted into the operator library as composite operators.
        """
        return [
            s.sequence
            for s in self._seqs.values()
            if s.solve_rate >= min_solve_rate and s.uses >= min_uses
        ]

    def get_op(self, name: str) -> Optional[OpStats]:
        return self._ops.get(name)

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "total_runs": self._total_runs,
            "solved_runs": self._solved_runs,
            "operators": {n: s.to_dict() for n, s in self._ops.items()},
            "top_sequences": [
                {"sequence": list(s.sequence), "uses": s.uses,
                 "solved": s.solved_count, "score": round(s.score(), 4)}
                for s in self.top_sequences(top_k=20)
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "OperatorStats":
        tracker = cls()
        tracker._total_runs = d.get("total_runs", 0)
        tracker._solved_runs = d.get("solved_runs", 0)
        for n, s in d.get("operators", {}).items():
            o = OpStats(name=n)
            o.calls = s.get("calls", 0)
            o.successes = s.get("successes", 0)
            o.total_duration = s.get("avg_duration_sec", 0.0) * o.calls
            tracker._ops[n] = o
        return tracker

    def __repr__(self) -> str:
        return (
            f"OperatorStats(runs={self._total_runs}, "
            f"solved={self._solved_runs}, "
            f"operators_tracked={len(self._ops)})"
        )

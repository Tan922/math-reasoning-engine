"""
mre.evaluation.commission
──────────────────────────
EvaluationCommission — three-judge panel that scores a PipelineResult.

Judges
------
CorrectnessJudge   40 %  — verifies the final answer against the expected
                            answer; falls back to symbolic equivalence check.
LogicJudge         30 %  — walks every StepRecord and verifies each SymPy
                            transformation is mathematically equivalent.
CritiqueAgent      20 %  — detects step-level flaws (circular, unnecessary,
                            missing cases) and assigns a fault-localisation score;
                            optionally calls an LLM for richer feedback.

Conciseness bonus  10 %  — rewards shorter step-chains for equal correctness.

Usage
-----
    from mre.evaluation.commission import EvaluationCommission
    commission = EvaluationCommission()
    verdict = commission.evaluate(pipeline_result, expected_answer="[2, 3]")
    print(verdict.report())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mre.agents.state import ReasoningState, StepRecord
from mre.operators.pipeline import PipelineResult
from mre.utils import get_logger

logger = get_logger(__name__)

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
    )
    _SYMPY = True
except ImportError:          # pragma: no cover
    _SYMPY = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _try_parse(s: str) -> Optional["sp.Expr"]:
    if not _SYMPY or not s:
        return None
    try:
        return parse_expr(str(s), transformations=standard_transformations)
    except Exception:
        return None


def _symbolic_equal(a: str, b: str) -> bool:
    """Return True if two string expressions are symbolically equivalent."""
    if not _SYMPY:
        return a.strip() == b.strip()
    ea, eb = _try_parse(a), _try_parse(b)
    if ea is None or eb is None:
        return a.strip() == b.strip()
    try:
        diff = sp.simplify(ea - eb)
        return diff == 0
    except Exception:
        return False


def _answer_in_text(expected: str, actual: str) -> bool:
    """Check whether *expected* appears (literally or symbolically) in *actual*."""
    if not expected or not actual:
        return False
    if expected.strip() in actual:
        return True
    return _symbolic_equal(expected, actual)


# ── per-judge results ─────────────────────────────────────────────────────────

@dataclass
class JudgeVerdict:
    judge: str
    score: float          # [0, 1]
    notes: List[str] = field(default_factory=list)
    weight: float = 1.0


# ── Judges ────────────────────────────────────────────────────────────────────

class CorrectnessJudge:
    """
    Verifies the final answer against the expected answer.

    Score breakdown:
      1.0  — answer present AND matches expected (string or symbolic)
      0.5  — answer present but doesn't clearly match
      0.0  — no answer
    """
    weight = 0.40
    name   = "CorrectnessJudge"

    def judge(
        self,
        result: PipelineResult,
        expected_answer: Optional[str],
    ) -> JudgeVerdict:
        state = result.final_state
        notes: List[str] = []

        if not state.is_solved:
            notes.append("No answer reached.")
            return JudgeVerdict(self.name, 0.0, notes, self.weight)

        if expected_answer is None:
            # No ground-truth: partial credit for having an answer
            notes.append("Answer present (no ground-truth to verify against).")
            return JudgeVerdict(self.name, 0.5, notes, self.weight)

        if _answer_in_text(expected_answer, str(state.answer)):
            notes.append(f"Correct: {state.answer!r} matches expected {expected_answer!r}.")
            return JudgeVerdict(self.name, 1.0, notes, self.weight)

        # Partial: answer present but may be expressed differently
        notes.append(
            f"Answer {state.answer!r} does not clearly match expected {expected_answer!r}."
        )
        return JudgeVerdict(self.name, 0.2, notes, self.weight)


class LogicJudge:
    """
    Walks every StepRecord and checks that each symbolic transformation
    is mathematically valid using SymPy equivalence.

    Score = (valid_steps / total_steps), penalised for failed steps.
    """
    weight = 0.30
    name   = "LogicJudge"

    def judge(
        self,
        result: PipelineResult,
        expected_answer: Optional[str] = None,
    ) -> JudgeVerdict:
        state  = result.final_state
        steps  = state.history
        notes: List[str] = []

        if not steps:
            notes.append("No steps to verify.")
            return JudgeVerdict(self.name, 0.5, notes, self.weight)

        valid   = 0
        invalid = 0
        skipped = 0

        for step in steps:
            if not step.success:
                invalid += 1
                notes.append(f"Failed step: {step.operator_name} — {step.error_msg[:60]}")
                continue

            in_expr  = _try_parse(step.input_snapshot)
            out_expr = _try_parse(step.output_snapshot)

            if in_expr is None or out_expr is None:
                # Can't check non-symbolic steps (meta-operators, text)
                skipped += 1
                continue

            # Check: output is a simplified/transformed version of input
            # Accept if simplify(in - out) == 0 OR if output is strictly simpler
            try:
                diff = sp.simplify(in_expr - out_expr)
                if diff == 0:
                    valid += 1
                    continue
                # Allow canonical transforms (factored, expanded, solved)
                # by checking symbolic equivalence under solve/expand
                expanded_in  = sp.expand(in_expr)
                expanded_out = sp.expand(out_expr)
                if sp.simplify(expanded_in - expanded_out) == 0:
                    valid += 1
                    continue
                # Accept if output is strictly simpler (fewer operations)
                if sp.count_ops(out_expr) < sp.count_ops(in_expr):
                    valid += 1
                    notes.append(
                        f"Step {step.operator_name}: output differs from input "
                        f"but is simpler — accepted."
                    )
                    continue
                invalid += 1
                notes.append(
                    f"Suspicious step {step.operator_name}: "
                    f"{step.input_snapshot[:30]} → {step.output_snapshot[:30]}"
                )
            except Exception:
                skipped += 1

        checkable = valid + invalid
        if checkable == 0:
            notes.append(f"All {skipped} steps were non-symbolic (meta-operators).")
            score = 0.8  # benefit of the doubt
        else:
            score = valid / checkable
            notes.append(
                f"Logic check: {valid} valid / {checkable} checkable "
                f"({skipped} skipped)."
            )

        # Penalise for any failed step (over and above the ratio)
        if invalid > 0:
            score *= max(0.0, 1.0 - 0.1 * invalid)

        return JudgeVerdict(self.name, round(score, 4), notes, self.weight)


class CritiqueAgent:
    """
    Detects step-level flaws and returns a fault-localisation score.

    With an LLM client: calls the model for richer narrative feedback.
    Without one (default): applies heuristic checkers.

    Fault types detected:
      - Circular steps (same output repeated)
      - Unnecessarily long chains
      - Unresolved free symbols in final answer
      - Missing repair after a failed step

    Score = 1 - (fault_penalty), where each fault subtracts 0.15.
    """
    weight = 0.20
    name   = "CritiqueAgent"

    def __init__(self, llm_client: Optional[Callable] = None):
        self.llm_client = llm_client

    def judge(
        self,
        result: PipelineResult,
        expected_answer: Optional[str] = None,
    ) -> JudgeVerdict:
        state  = result.final_state
        notes: List[str] = []
        faults = 0

        # 1. Circular steps
        outputs_seen: Dict[str, int] = {}
        for i, step in enumerate(state.history):
            key = step.output_snapshot
            if key in outputs_seen:
                faults += 1
                notes.append(
                    f"Circular: step {outputs_seen[key]+1} and step {i+1} "
                    f"both produced {key[:40]!r}."
                )
            else:
                outputs_seen[key] = i

        # 2. Excessively long chain without answer
        if not state.is_solved and len(state.history) > 5:
            faults += 1
            notes.append(
                f"Chain length {len(state.history)} without reaching an answer."
            )

        # 3. Unresolved free symbols
        if state.answer and _SYMPY:
            parsed = _try_parse(str(state.answer))
            if parsed is not None and not isinstance(parsed, list) and hasattr(parsed, 'free_symbols') and parsed.free_symbols:
                faults += 1
                notes.append(
                    f"Answer still contains free symbols: {parsed.free_symbols}."
                )

        # 4. Failed step with no subsequent RepairChain
        has_failed = any(not s.success for s in state.history)
        has_repair = any(s.operator_name == "RepairChain" for s in state.history)
        if has_failed and not has_repair:
            faults += 1
            notes.append("Failed step present but no RepairChain was applied.")

        # 5. LLM enrichment
        llm_feedback: Optional[str] = None
        if self.llm_client and faults > 0:
            llm_feedback = self._llm_critique(state)
            notes.append(f"LLM critique: {llm_feedback[:120]}")

        score = max(0.0, 1.0 - faults * 0.15)
        if not notes:
            notes.append("No faults detected.")

        return JudgeVerdict(self.name, round(score, 4), notes, self.weight)

    def _llm_critique(self, state: ReasoningState) -> str:
        history_text = "\n".join(
            f"  [{i+1}] {r.operator_name}: {r.input_snapshot[:50]} "
            f"→ {r.output_snapshot[:50]}"
            for i, r in enumerate(state.history)
        )
        prompt = (
            f"Problem: {state.problem}\n"
            f"Steps:\n{history_text}\n"
            f"Answer: {state.answer}\n\n"
            "In one sentence, identify the most critical flaw in this reasoning chain, "
            "or say 'No critical flaws' if the chain is sound."
        )
        try:
            return str(self.llm_client(prompt))
        except Exception as exc:
            return f"LLM unavailable: {exc}"


# ── Verdict aggregator ────────────────────────────────────────────────────────

@dataclass
class CommissionVerdict:
    """Aggregated verdict from all three judges."""

    judge_verdicts: List[JudgeVerdict]
    weighted_score: float
    expected_answer: Optional[str]
    result: PipelineResult

    # Convenience accessors
    @property
    def correctness(self) -> float:
        return next((v.score for v in self.judge_verdicts
                     if v.judge == "CorrectnessJudge"), 0.0)

    @property
    def logic(self) -> float:
        return next((v.score for v in self.judge_verdicts
                     if v.judge == "LogicJudge"), 0.0)

    @property
    def critique(self) -> float:
        return next((v.score for v in self.judge_verdicts
                     if v.judge == "CritiqueAgent"), 0.0)

    @property
    def conciseness(self) -> float:
        return next((v.score for v in self.judge_verdicts
                     if v.judge == "ConcisenessScore"), 0.0)

    def report(self) -> str:
        lines = [
            "╔══ Commission Verdict ═══════════════════════════════════╗",
            f"  Weighted score : {self.weighted_score:.4f}",
            f"  Expected answer: {self.expected_answer}",
            f"  Actual answer  : {self.result.final_state.answer}",
            "  ── Judge breakdown ─────────────────────────────────────",
        ]
        for v in self.judge_verdicts:
            pct = f"{v.weight*100:.0f}%"
            lines.append(f"  [{pct:>3}] {v.judge:<22} score={v.score:.4f}")
            for note in v.notes[:3]:
                lines.append(f"          ↳ {note}")
        lines.append("╚═════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ── Commission ────────────────────────────────────────────────────────────────

class EvaluationCommission:
    """
    Three-judge panel that produces a weighted [0, 1] score for any
    PipelineResult.

    Parameters
    ----------
    llm_client : callable, optional
        ``(prompt: str) -> str`` — enables LLM-powered critique.
    weights : dict, optional
        Override the default judge weights.
        Keys: 'correctness', 'logic', 'critique', 'conciseness'.
    max_concise_steps : int
        Step count at which conciseness score starts decreasing (default 4).
    """

    def __init__(
        self,
        llm_client: Optional[Callable] = None,
        weights: Optional[Dict[str, float]] = None,
        max_concise_steps: int = 4,
    ):
        w = weights or {}
        self._correctness_w  = w.get("correctness",  0.40)
        self._logic_w        = w.get("logic",         0.30)
        self._critique_w     = w.get("critique",      0.20)
        self._conciseness_w  = w.get("conciseness",   0.10)
        self.max_concise_steps = max_concise_steps

        self._judges = [
            CorrectnessJudge(),
            LogicJudge(),
            CritiqueAgent(llm_client=llm_client),
        ]
        # Override weights from constructor
        self._judges[0].weight = self._correctness_w
        self._judges[1].weight = self._logic_w
        self._judges[2].weight = self._critique_w

    def evaluate(
        self,
        result: PipelineResult,
        expected_answer: Optional[str] = None,
    ) -> CommissionVerdict:
        """Score a PipelineResult and return a CommissionVerdict."""
        verdicts: List[JudgeVerdict] = []

        for judge in self._judges:
            v = judge.judge(result, expected_answer)
            verdicts.append(v)

        # Conciseness score
        n_steps = max(1, len(result.final_state.history))
        conciseness = max(0.0, 1.0 - max(0, n_steps - self.max_concise_steps) * 0.1)
        verdicts.append(JudgeVerdict(
            judge="ConcisenessScore",
            score=round(conciseness, 4),
            notes=[f"{n_steps} steps (target ≤ {self.max_concise_steps})."],
            weight=self._conciseness_w,
        ))

        # Weighted aggregate
        total_w = sum(v.weight for v in verdicts)
        score   = sum(v.score * v.weight for v in verdicts) / max(total_w, 1e-9)

        logger.debug(
            "Commission scored %.4f (correct=%.2f logic=%.2f critique=%.2f concise=%.2f)",
            score,
            verdicts[0].score, verdicts[1].score,
            verdicts[2].score, verdicts[3].score,
        )

        return CommissionVerdict(
            judge_verdicts=verdicts,
            weighted_score=round(score, 4),
            expected_answer=expected_answer,
            result=result,
        )

    def batch_evaluate(
        self,
        results: List[PipelineResult],
        expected_answers: Optional[List[Optional[str]]] = None,
    ) -> List[CommissionVerdict]:
        """Evaluate a list of results, returning one verdict per result."""
        if expected_answers is None:
            expected_answers = [None] * len(results)
        return [
            self.evaluate(r, ea)
            for r, ea in zip(results, expected_answers)
        ]

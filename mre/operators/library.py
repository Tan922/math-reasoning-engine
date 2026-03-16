"""
mre.operators.library
─────────────────────
Phase-2 operator library — six high-value operators.

  Symbolic  : SymbolicSimplify
  Logical   : DeductiveStep, ProofByContradiction
  Algebraic : EquationSolve
  Meta      : SelfCritique, RepairChain

Each follows the contract in mre.operators.base.BaseOperator:
  apply(state: ReasoningState) -> ReasoningState

SymPy is used for deterministic symbolic computation.
SelfCritique / RepairChain optionally call an LLM; they degrade
gracefully to heuristic mode when no LLM client is provided.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Callable, Dict, List, Optional

from mre.agents.state import ReasoningState
from mre.operators.base import BaseOperator

# SymPy is a required dependency (already in requirements.txt)
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
    _SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SYMPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TRANSFORMATIONS = (
    standard_transformations + (implicit_multiplication_application,)
    if _SYMPY_AVAILABLE
    else ()
)


def _safe_parse(expr_str: str) -> Optional["sp.Expr"]:
    """Try to parse a string as a SymPy expression; return None on failure.

    Tries with standard_transformations first (preserves function calls like
    sin/cos), then falls back to implicit_multiplication_application for
    expressions like '2x + 3'.
    """
    if not _SYMPY_AVAILABLE:
        return None
    # First try: standard only — correct for sin(x), cos(x), etc.
    try:
        return parse_expr(expr_str, transformations=standard_transformations)
    except Exception:
        pass
    # Second try: with implicit multiplication — correct for '2x+3'
    try:
        return parse_expr(expr_str, transformations=_TRANSFORMATIONS)
    except Exception:
        return None


def _extract_equation(text: str) -> Optional[str]:
    """
    Extract the first equation-like substring from free text.
    Looks for patterns such as  '2*x + 3 = 7'  or  'x**2 - 4 = 0'.
    """
    m = re.search(r"([A-Za-z0-9_\+\-\*/\^\(\)\s\.]+=[A-Za-z0-9_\+\-\*/\^\(\)\s\.]+)", text)
    return m.group(1).strip() if m else None


def _extract_math_expr(text: str) -> Optional[str]:
    """
    Extract a parseable mathematical expression from a natural-language sentence.
    Tries:
      1. Everything after 'simplify', 'expand', 'factorise'/'factor', 'evaluate' keywords.
      2. A parenthesised sub-expression.
      3. The last whitespace-delimited token that looks like math.
    """
    # Keyword-based extraction
    m = re.search(
        r"(?:simplify|expand|factor(?:ise)?|evaluate|compute|calculate)\s+(.+)$",
        text, re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip().rstrip('.')
        if _safe_parse(candidate) is not None:
            return candidate

    # Parenthesised expression
    m = re.search(r"\(([^()]+(?:\([^()]*\)[^()]*)*)\)", text)
    if m:
        candidate = m.group(0)  # include outer parens
        if _safe_parse(candidate) is not None:
            return candidate

    # Last token that contains math operators
    tokens = text.split()
    for tok in reversed(tokens):
        if re.search(r"[\+\-\*/\^]|x\*\*", tok):
            if _safe_parse(tok) is not None:
                return tok

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. SymbolicSimplify
# ─────────────────────────────────────────────────────────────────────────────

class SymbolicSimplify(BaseOperator):
    """
    Simplify the current expression using SymPy.

    Tries, in order: simplify → cancel → trigsimp → expand.
    Falls back to the original expression if none succeeds.
    """

    name = "SymbolicSimplify"
    description = (
        "Simplify the current symbolic expression with SymPy "
        "(simplify → cancel → trigsimp → expand)."
    )

    def _run(self, state: ReasoningState) -> ReasoningState:
        expr_str = state.current_expression.strip()
        if not _SYMPY_AVAILABLE:
            return state.evolve(
                context={**state.context, "simplify_note": "SymPy unavailable — skipped"},
            )

        parsed = _safe_parse(expr_str)
        if parsed is None:
            # Try to extract a math expression from natural-language text
            extracted = _extract_math_expr(expr_str)
            if extracted:
                parsed = _safe_parse(extracted)
                expr_str = extracted  # work on the extracted expression
            if parsed is None:
                note = f"SymbolicSimplify: could not parse {expr_str[:60]!r} — passed through"
                return state.evolve(context={**state.context, "simplify_note": note})

        simplified = None
        method_used = "none"
        for fn_name, fn in [
            ("simplify",  sp.simplify),
            ("cancel",    sp.cancel),
            ("trigsimp",  sp.trigsimp),
            ("expand",    sp.expand),
        ]:
            try:
                result = fn(parsed)
                if result != parsed:
                    simplified = result
                    method_used = fn_name
                    break
            except Exception:
                continue

        if simplified is None:
            simplified = parsed
            method_used = "identity"

        # If we only got "identity" on the full sentence, try extracting math
        if method_used == "identity" and parsed != _safe_parse(str(parsed)):
            pass  # already the extracted form
        if method_used == "identity":
            extracted = _extract_math_expr(expr_str)
            if extracted and extracted != expr_str:
                parsed2 = _safe_parse(extracted)
                if parsed2 is not None:
                    for fn_name2, fn2 in [
                        ("simplify", sp.simplify),
                        ("cancel",   sp.cancel),
                        ("trigsimp", sp.trigsimp),
                        ("expand",   sp.expand),
                    ]:
                        try:
                            result2 = fn2(parsed2)
                            if result2 != parsed2:
                                simplified = result2
                                method_used = fn_name2
                                break
                        except Exception:
                            continue
                    if method_used == "identity":
                        simplified = sp.simplify(parsed2)
                        method_used = "simplify_extracted"

        new_expr = str(simplified)
        ctx = {**state.context, "simplify_method": method_used, "simplify_expr": new_expr}
        new_state = state.evolve(current_expression=new_expr, context=ctx)
        # Treat simplified result as the answer when not solving an equation
        if not ctx.get("equation") and method_used not in ("identity",):
            new_state = new_state.set_answer(new_expr, confidence=0.85)
        return new_state


# ─────────────────────────────────────────────────────────────────────────────
# 2. DeductiveStep
# ─────────────────────────────────────────────────────────────────────────────

class DeductiveStep(BaseOperator):
    """
    Apply a single modus-ponens deductive step.

    Looks in ``state.context['rules']`` for a list of
    ``{"if": <pattern>, "then": <consequence>}`` rule dicts.
    For each rule whose antecedent matches a substring of the current
    expression the operator appends the consequent to the context
    ``'deduced'`` list and updates ``current_expression`` to the
    first matched consequent.
    """

    name = "DeductiveStep"
    description = (
        "Apply modus-ponens rules from state.context['rules'] "
        "to derive new conclusions."
    )

    def _run(self, state: ReasoningState) -> ReasoningState:
        rules: List[Dict[str, str]] = state.context.get("rules", [])
        expr = state.current_expression.lower()

        deduced = list(state.context.get("deduced", []))
        first_consequent: Optional[str] = None

        for rule in rules:
            antecedent = rule.get("if", "").lower()
            consequent = rule.get("then", "")
            if antecedent and antecedent in expr:
                deduced.append(consequent)
                if first_consequent is None:
                    first_consequent = consequent

        ctx = {**state.context, "deduced": deduced}
        if first_consequent:
            return state.evolve(current_expression=first_consequent, context=ctx)
        else:
            ctx["deductive_note"] = "No matching rules found."
            return state.evolve(context=ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ProofByContradiction
# ─────────────────────────────────────────────────────────────────────────────

class ProofByContradiction(BaseOperator):
    """
    Attempt a symbolic proof by contradiction.

    Assumes the negation of the current hypothesis, then tries to derive
    a contradiction using SymPy's ``satisfiable`` (for Boolean formulas)
    or checks if a system of equations becomes inconsistent.

    Works best with Boolean / polynomial expressions.
    """

    name = "ProofByContradiction"
    description = (
        "Negate the hypothesis and check for contradiction via "
        "SymPy satisfiable / system inconsistency."
    )

    def _run(self, state: ReasoningState) -> ReasoningState:
        if not _SYMPY_AVAILABLE:
            return state.evolve(
                context={**state.context, "contradiction_note": "SymPy unavailable"}
            )

        hypothesis = state.context.get("hypothesis", state.current_expression)
        ctx = {**state.context}

        # ── Try Boolean satisfiability ────────────────────────────────────────
        parsed = _safe_parse(hypothesis)
        if parsed is not None:
            try:
                negation = sp.Not(parsed)
                sat_result = sp.satisfiable(negation)
                if sat_result is False:
                    ctx["contradiction_found"] = True
                    ctx["contradiction_method"] = "boolean_unsat"
                    conclusion = (
                        f"Proved: {hypothesis} (negation is unsatisfiable)"
                    )
                    return state.evolve(
                        current_expression=conclusion,
                        answer=conclusion,
                        context=ctx,
                    )
            except Exception:
                pass

        # ── Try system inconsistency ─────────────────────────────────────────
        assumptions: List[str] = ctx.get("assumptions", [])
        if assumptions:
            sym_assumptions = [_safe_parse(a) for a in assumptions if _safe_parse(a)]
            if sym_assumptions:
                try:
                    system = [sp.Eq(a, 0) for a in sym_assumptions if not isinstance(a, sp.Eq)]
                    if not system:
                        system = sym_assumptions
                    sols = sp.solve(system)
                    if sols is False or sols == [] or sols == {}:
                        ctx["contradiction_found"] = True
                        ctx["contradiction_method"] = "system_inconsistency"
                        conclusion = "Proved by contradiction: assumptions are inconsistent."
                        return state.evolve(
                            current_expression=conclusion,
                            answer=conclusion,
                            context=ctx,
                        )
                except Exception:
                    pass

        ctx["contradiction_found"] = False
        ctx["contradiction_note"] = "No contradiction derived — may need stronger assumptions."
        return state.evolve(context=ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EquationSolve
# ─────────────────────────────────────────────────────────────────────────────

class EquationSolve(BaseOperator):
    """
    Solve an equation or system of equations with SymPy.

    Extracts an equation from ``current_expression`` or
    ``state.context['equation']``, identifies free symbols,
    and calls ``sp.solve()``.  Stores solutions in
    ``context['solutions']`` and sets ``current_expression``
    to the solution string.
    """

    name = "EquationSolve"
    description = (
        "Solve an algebraic equation/system with SymPy and store "
        "solutions in context['solutions']."
    )

    def _run(self, state: ReasoningState) -> ReasoningState:
        if not _SYMPY_AVAILABLE:
            return state.evolve(
                context={**state.context, "solve_note": "SymPy unavailable"}
            )

        ctx = {**state.context}

        # Locate the equation string
        eq_str = ctx.get("equation") or _extract_equation(state.current_expression)
        if not eq_str:
            ctx["solve_note"] = "No equation detected in expression."
            return state.evolve(context=ctx)

        # Parse left-hand and right-hand sides
        if "=" in eq_str:
            lhs_str, rhs_str = eq_str.split("=", 1)
            lhs = _safe_parse(lhs_str.strip())
            rhs = _safe_parse(rhs_str.strip())
            if lhs is None or rhs is None:
                ctx["solve_note"] = f"Could not parse equation: {eq_str!r}"
                return state.evolve(context=ctx)
            equation = sp.Eq(lhs, rhs)
        else:
            parsed = _safe_parse(eq_str)
            if parsed is None:
                ctx["solve_note"] = f"Could not parse: {eq_str!r}"
                return state.evolve(context=ctx)
            equation = parsed

        # Determine solve variable(s)
        free = equation.free_symbols if hasattr(equation, "free_symbols") else set()
        if not free:
            ctx["solve_note"] = "No free symbols found."
            return state.evolve(context=ctx)

        solve_for = ctx.get("solve_for")
        if solve_for:
            target = sp.Symbol(solve_for)
            symbols = [target] if target in free else list(free)
        else:
            # Prefer single-char symbols; fall back to first alphabetically
            single = sorted([s for s in free if len(str(s)) == 1], key=str)
            symbols = single if single else sorted(free, key=str)[:1]

        try:
            solutions = sp.solve(equation, symbols)
            ctx["solutions"] = str(solutions)
            ctx["solve_variable"] = str(symbols)
            sol_str = f"Solution for {symbols}: {solutions}"
            new_state = state.evolve(current_expression=sol_str, context=ctx)
            if solutions:
                new_state = new_state.set_answer(sol_str, confidence=0.9)
            return new_state
        except Exception as exc:
            ctx["solve_note"] = f"sp.solve failed: {exc}"
            return state.evolve(context=ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SelfCritique  (meta-operator)
# ─────────────────────────────────────────────────────────────────────────────

class SelfCritique(BaseOperator):
    """
    Meta-operator: reflect on the current reasoning chain and flag issues.

    With an LLM client:
        Calls the LLM to critique the reasoning history and suggest
        improvements, storing feedback in context['critique'].

    Without an LLM client (offline / test mode):
        Applies heuristic checks:
          - Detects circular steps (same expression repeated).
          - Flags empty or very short answers.
          - Checks for unresolved symbolic variables.
    """

    name = "SelfCritique"
    description = (
        "Reflect on the reasoning chain; detect errors or gaps. "
        "Uses LLM when available, otherwise applies heuristic checks."
    )

    def __init__(self, llm_client: Optional[Callable] = None):
        """
        Parameters
        ----------
        llm_client : callable, optional
            A function ``(prompt: str) -> str`` that calls an LLM.
            If None, heuristic mode is used.
        """
        self.llm_client = llm_client

    def _run(self, state: ReasoningState) -> ReasoningState:
        ctx = {**state.context}

        if self.llm_client is not None:
            critique = self._llm_critique(state)
        else:
            critique = self._heuristic_critique(state)

        ctx["critique"] = critique
        issues_found = bool(critique.get("issues"))
        ctx["critique_issues_found"] = issues_found

        # Lower confidence if issues detected
        new_confidence = state.confidence * (0.7 if issues_found else 1.0)
        return state.evolve(context=ctx, confidence=round(new_confidence, 4))

    # ── LLM path ──────────────────────────────────────────────────────────────

    def _llm_critique(self, state: ReasoningState) -> Dict[str, Any]:
        history_text = "\n".join(
            f"  [{i+1}] {r.operator_name}: {r.input_snapshot} → {r.output_snapshot}"
            for i, r in enumerate(state.history)
        )
        prompt = textwrap.dedent(f"""
            You are a rigorous mathematical proof-checker.

            Original problem: {state.problem}

            Reasoning steps taken:
            {history_text}

            Current expression: {state.current_expression}

            Please critique the reasoning above. Identify any:
            - Logical errors or unjustified leaps
            - Missing cases or edge cases
            - Circular arguments
            - Steps that could be simplified

            Reply as JSON with keys: "issues" (list of strings) and "suggestions" (list of strings).
            Reply ONLY with the JSON object, no markdown.
        """).strip()

        try:
            raw = self.llm_client(prompt)
            import json
            data = json.loads(raw)
            return data
        except Exception as exc:
            return {"issues": [], "suggestions": [], "llm_error": str(exc)}

    # ── Heuristic path ────────────────────────────────────────────────────────

    def _heuristic_critique(self, state: ReasoningState) -> Dict[str, Any]:
        issues: List[str] = []
        suggestions: List[str] = []

        # Check for circular steps (same output appearing twice)
        outputs = [r.output_snapshot for r in state.history]
        seen: Dict[str, int] = {}
        for i, out in enumerate(outputs):
            if out in seen:
                issues.append(
                    f"Circular step detected: step {seen[out]+1} and step {i+1} "
                    f"both produce {out[:40]!r}"
                )
            seen[out] = i

        # Check for failed steps
        failed_steps = [r for r in state.history if not r.success]
        if failed_steps:
            for r in failed_steps:
                issues.append(f"Step '{r.operator_name}' failed: {r.error_msg[:60]}")

        # Check for very short / empty current expression
        if len(state.current_expression.strip()) < 3:
            issues.append("Current expression is suspiciously short or empty.")
            suggestions.append("Check that the upstream operator returned a valid result.")

        # Check for unresolved SymPy symbols (e.g., 'x', 'y' still free)
        if _SYMPY_AVAILABLE:
            parsed = _safe_parse(state.current_expression)
            if parsed is not None:
                free = parsed.free_symbols
                if free:
                    suggestions.append(
                        f"Free symbols remain: {free} — consider running EquationSolve."
                    )

        # No answer yet but chain is long
        if state.answer is None and len(state.history) >= 4:
            suggestions.append("Many steps taken but no answer reached — consider RepairChain.")

        return {"issues": issues, "suggestions": suggestions, "mode": "heuristic"}


# ─────────────────────────────────────────────────────────────────────────────
# 6. RepairChain  (meta-operator)
# ─────────────────────────────────────────────────────────────────────────────

class RepairChain(BaseOperator):
    """
    Meta-operator: targeted repair of the most recent failed step.

    Repair strategy (in priority order):
      1. If the last step produced a parse error → re-run SymbolicSimplify
         on the raw problem expression.
      2. If SelfCritique flagged circular steps → reset current_expression
         to the problem statement and clear circular context.
      3. If EquationSolve failed → attempt to extract a simpler sub-equation
         and re-run.
      4. Generic fallback → resets current_expression to the problem and
         clears the failed flag so the pipeline can continue.

    After repair, clears ``state.failed`` and appends a repair note to the
    context.
    """

    name = "RepairChain"
    description = (
        "Diagnose the most recent failure and apply targeted repair, "
        "then continue the reasoning chain."
    )

    def apply(self, state: ReasoningState) -> ReasoningState:
        """Override BaseOperator.apply to execute even when state.failed == True."""
        import time as _time
        t0 = _time.perf_counter()
        try:
            new_state = self._run(state)
            duration = _time.perf_counter() - t0
            from mre.agents.state import StepRecord as _SR
            record = _SR(
                operator_name=self.name,
                input_snapshot=state.current_expression[:120],
                output_snapshot=new_state.current_expression[:120],
                success=not new_state.failed,
                error_msg=new_state.failure_reason if new_state.failed else "",
                duration_sec=round(duration, 4),
            )
            return new_state.add_step(record)
        except Exception as exc:
            duration = _time.perf_counter() - t0
            from mre.agents.state import StepRecord as _SR
            record = _SR(
                operator_name=self.name,
                input_snapshot=state.current_expression[:120],
                output_snapshot="",
                success=False,
                error_msg=str(exc),
                duration_sec=round(duration, 4),
            )
            return state.mark_failed(f"{self.name} raised: {exc}").add_step(record)

    def _run(self, state: ReasoningState) -> ReasoningState:
        ctx = {**state.context}
        repair_log: List[str] = list(ctx.get("repair_log", []))

        # Find the last failed step
        last_failed = next(
            (r for r in reversed(state.history) if not r.success),
            None,
        )

        if last_failed is None and not state.failed:
            ctx["repair_log"] = repair_log + ["RepairChain: nothing to repair."]
            return state.evolve(context=ctx)

        failed_op = last_failed.operator_name if last_failed else "unknown"
        error_msg = last_failed.error_msg if last_failed else state.failure_reason

        # ── Repair strategy selection ──────────────────────────────────────────

        repaired_expr = state.current_expression
        repair_action = "generic_reset"

        if "parse" in error_msg.lower() or "could not parse" in error_msg.lower():
            # Strategy 1: re-attempt with the original problem text
            repaired_expr = state.problem
            repair_action = "reset_to_problem"
            if _SYMPY_AVAILABLE:
                cleaned = re.sub(r"[^A-Za-z0-9_\+\-\*/\^\(\)\.\s=]", "", state.problem)
                repaired_expr = cleaned or state.problem

        elif ctx.get("critique_issues_found") and any(
            "circular" in iss.lower() for iss in ctx.get("critique", {}).get("issues", [])
        ):
            # Strategy 2: break cycle by resetting
            repaired_expr = state.problem
            repair_action = "break_circular_cycle"
            ctx.pop("deduced", None)

        elif failed_op == "EquationSolve":
            # Strategy 3: try to extract a simpler sub-equation
            sub_eq = _extract_equation(state.problem)
            if sub_eq:
                repaired_expr = sub_eq
                repair_action = "simplified_equation_extraction"
            else:
                repaired_expr = state.problem
                repair_action = "reset_to_problem"

        else:
            # Generic fallback
            repaired_expr = state.problem
            repair_action = "generic_reset"

        repair_log.append(
            f"Repaired after {failed_op!r} failure ({error_msg[:60]}): "
            f"action={repair_action!r}"
        )
        ctx["repair_log"] = repair_log
        ctx["last_repair_action"] = repair_action

        # Clear the failed flag so the pipeline can continue
        return state.evolve(
            current_expression=repaired_expr,
            context=ctx,
            failed=False,
            failure_reason="",
            confidence=max(0.1, state.confidence * 0.8),
        )

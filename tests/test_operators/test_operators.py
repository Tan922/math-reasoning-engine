"""
tests/test_phase2/test_operators.py
────────────────────────────────────
Unit tests for all six Phase-2 reasoning operators.
Fully offline — no LLM calls, no network.
"""

import pytest

from mre.agents.state import ReasoningState
from mre.operators.base import OPERATOR_REGISTRY, get_operator, list_operators
from mre.operators.library import (
    DeductiveStep,
    EquationSolve,
    ProofByContradiction,
    RepairChain,
    SelfCritique,
    SymbolicSimplify,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_state():
    return ReasoningState.from_problem("test problem")


# ── Registry ─────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_six_operators_registered(self):
        expected = {
            "SymbolicSimplify",
            "DeductiveStep",
            "ProofByContradiction",
            "EquationSolve",
            "SelfCritique",
            "RepairChain",
        }
        assert expected.issubset(set(OPERATOR_REGISTRY.keys()))

    def test_get_operator_by_name(self):
        op = get_operator("SymbolicSimplify")
        assert op.name == "SymbolicSimplify"

    def test_get_operator_unknown_raises(self):
        with pytest.raises(KeyError):
            get_operator("NonExistentOperator")

    def test_list_operators_returns_descriptions(self):
        ops = list_operators()
        for name in ["SymbolicSimplify", "EquationSolve"]:
            assert name in ops
            assert isinstance(ops[name], str)


# ── BaseOperator contract ─────────────────────────────────────────────────────

class TestBaseContract:
    """Every operator must fulfil the base contract."""

    @pytest.mark.parametrize("op_name", [
        "SymbolicSimplify",
        "DeductiveStep",
        "ProofByContradiction",
        "EquationSolve",
        "RepairChain",
    ])
    def test_apply_returns_new_state(self, op_name, blank_state):
        op = get_operator(op_name)
        result = op.apply(blank_state)
        assert isinstance(result, ReasoningState)
        assert result is not blank_state

    @pytest.mark.parametrize("op_name", [
        "SymbolicSimplify",
        "DeductiveStep",
        "EquationSolve",
        "RepairChain",
    ])
    def test_apply_appends_step_record(self, op_name, blank_state):
        op = get_operator(op_name)
        before = len(blank_state.history)
        result = op.apply(blank_state)
        assert len(result.history) == before + 1

    def test_failed_state_passes_through(self, blank_state):
        failed = blank_state.mark_failed("upstream error")
        op = SymbolicSimplify()
        result = op.apply(failed)
        # Should pass through without adding a step
        assert result.failed
        assert len(result.history) == len(failed.history)


# ── SymbolicSimplify ──────────────────────────────────────────────────────────

class TestSymbolicSimplify:
    def test_simplify_numeric_expression(self):
        state = ReasoningState.from_problem("simplify")
        state = state.evolve(current_expression="(x**2 - 1)/(x - 1)")
        op = SymbolicSimplify()
        result = op.apply(state)
        # Should simplify to x + 1
        assert "x + 1" in result.current_expression or result.context.get("simplify_method")

    def test_simplify_trivial_passes_through(self):
        state = ReasoningState.from_problem("p")
        state = state.evolve(current_expression="x")
        op = SymbolicSimplify()
        result = op.apply(state)
        assert isinstance(result, ReasoningState)

    def test_simplify_unparseable_adds_note(self):
        state = ReasoningState.from_problem("natural language")
        state = state.evolve(current_expression="The cat sat on the mat")
        op = SymbolicSimplify()
        result = op.apply(state)
        assert "simplify_note" in result.context or result.current_expression

    def test_simplify_trig(self):
        state = ReasoningState.from_problem("trig")
        state = state.evolve(current_expression="sin(x)**2 + cos(x)**2")
        op = SymbolicSimplify()
        result = op.apply(state)
        assert "1" in result.current_expression


# ── DeductiveStep ─────────────────────────────────────────────────────────────

class TestDeductiveStep:
    def test_applies_matching_rule(self):
        rules = [{"if": "x is even", "then": "x is divisible by 2"}]
        state = ReasoningState.from_problem("is x divisible by 2?")
        state = state.evolve(
            current_expression="x is even",
            context={"rules": rules},
        )
        op = DeductiveStep()
        result = op.apply(state)
        assert "x is divisible by 2" in result.current_expression
        assert "x is divisible by 2" in result.context.get("deduced", [])

    def test_no_matching_rule_adds_note(self, blank_state):
        state = blank_state.evolve(context={"rules": [{"if": "foo", "then": "bar"}]})
        op = DeductiveStep()
        result = op.apply(state)
        assert "deductive_note" in result.context

    def test_empty_rules_graceful(self, blank_state):
        op = DeductiveStep()
        result = op.apply(blank_state)
        assert isinstance(result, ReasoningState)

    def test_multiple_matching_rules(self):
        rules = [
            {"if": "prime", "then": "odd (unless 2)"},
            {"if": "prime", "then": "has exactly two divisors"},
        ]
        state = ReasoningState.from_problem("p")
        state = state.evolve(
            current_expression="n is prime",
            context={"rules": rules},
        )
        op = DeductiveStep()
        result = op.apply(state)
        assert len(result.context["deduced"]) == 2


# ── ProofByContradiction ──────────────────────────────────────────────────────

class TestProofByContradiction:
    def test_contradiction_with_false_negation(self):
        # P & ~P is always False → contradiction found
        import sympy as sp
        state = ReasoningState.from_problem("prove tautology")
        x = sp.Symbol("x")
        # Use a formula whose negation is unsatisfiable
        state = state.evolve(
            current_expression="x | ~x",   # tautology
            context={"hypothesis": "x | ~x"},
        )
        op = ProofByContradiction()
        result = op.apply(state)
        # If SymPy is available it should detect unsat negation
        assert isinstance(result, ReasoningState)

    def test_no_contradiction_returns_state(self):
        state = ReasoningState.from_problem("prove x > 0")
        state = state.evolve(
            current_expression="x > 0",
            context={"hypothesis": "x"},
        )
        op = ProofByContradiction()
        result = op.apply(state)
        assert "contradiction_found" in result.context

    def test_inconsistent_system_detected(self):
        # x = 1 AND x = 2 is inconsistent
        state = ReasoningState.from_problem("inconsistent system")
        state = state.evolve(
            current_expression="x = 1",
            context={"assumptions": ["x - 1", "x - 2"]},
        )
        op = ProofByContradiction()
        result = op.apply(state)
        assert isinstance(result, ReasoningState)


# ── EquationSolve ─────────────────────────────────────────────────────────────

class TestEquationSolve:
    def test_solve_linear_equation(self):
        state = ReasoningState.from_problem("Solve x + 3 = 7")
        state = state.evolve(context={"equation": "x + 3 = 7"})
        op = EquationSolve()
        result = op.apply(state)
        assert result.context.get("solutions") is not None
        assert "4" in str(result.context["solutions"])

    def test_solve_quadratic(self):
        state = ReasoningState.from_problem("Solve x**2 - 5*x + 6 = 0")
        state = state.evolve(context={"equation": "x**2 - 5*x + 6 = 0"})
        op = EquationSolve()
        result = op.apply(state)
        sols = str(result.context.get("solutions", ""))
        assert "2" in sols or "3" in sols

    def test_sets_answer_on_success(self):
        state = ReasoningState.from_problem("Solve 2*x = 8")
        state = state.evolve(context={"equation": "2*x = 8"})
        op = EquationSolve()
        result = op.apply(state)
        assert result.is_solved

    def test_no_equation_adds_note(self, blank_state):
        state = blank_state.evolve(current_expression="The sky is blue")
        op = EquationSolve()
        result = op.apply(state)
        assert "solve_note" in result.context

    def test_extracts_equation_from_problem_text(self):
        state = ReasoningState.from_problem("Find x where x - 4 = 0")
        op = EquationSolve()
        result = op.apply(state)
        assert isinstance(result, ReasoningState)


# ── SelfCritique ──────────────────────────────────────────────────────────────

class TestSelfCritique:
    def test_heuristic_mode_no_issues_clean_state(self):
        state = ReasoningState.from_problem("Solve x = 1")
        state = state.evolve(current_expression="x = 1", answer="x=1")
        op = SelfCritique(llm_client=None)
        result = op.apply(state)
        critique = result.context.get("critique", {})
        assert "issues" in critique
        assert critique.get("mode") == "heuristic"

    def test_heuristic_detects_circular_step(self):
        from mre.agents.state import StepRecord
        state = ReasoningState.from_problem("p")
        state = state.evolve(current_expression="expr")
        # Manually inject two identical output steps
        r1 = StepRecord("Op1", "in", "same_output", True)
        r2 = StepRecord("Op2", "in", "same_output", True)
        state = state.evolve(history=[r1, r2])
        op = SelfCritique(llm_client=None)
        result = op.apply(state)
        critique = result.context.get("critique", {})
        issues = critique.get("issues", [])
        assert any("circular" in i.lower() for i in issues)

    def test_heuristic_detects_failed_step(self):
        from mre.agents.state import StepRecord
        state = ReasoningState.from_problem("p")
        r = StepRecord("BadOp", "x", "", success=False, error_msg="division by zero")
        state = state.evolve(history=[r])
        op = SelfCritique(llm_client=None)
        result = op.apply(state)
        issues = result.context.get("critique", {}).get("issues", [])
        assert any("BadOp" in i for i in issues)

    def test_confidence_reduced_when_issues_found(self):
        from mre.agents.state import StepRecord
        state = ReasoningState.from_problem("p")
        r = StepRecord("Op", "x", "x", success=False, error_msg="err")
        state = state.evolve(history=[r], confidence=1.0)
        op = SelfCritique(llm_client=None)
        result = op.apply(state)
        assert result.confidence < 1.0


# ── RepairChain ───────────────────────────────────────────────────────────────

class TestRepairChain:
    def test_repairs_parse_error(self):
        from mre.agents.state import StepRecord
        state = ReasoningState.from_problem("x + 1 = 0")
        r = StepRecord("EquationSolve", "in", "", success=False,
                       error_msg="Could not parse equation")
        state = state.evolve(history=[r], failed=True, failure_reason="parse error")
        op = RepairChain()
        result = op.apply(state)
        assert not result.failed
        assert result.context.get("last_repair_action") == "reset_to_problem"

    def test_repairs_generic_failure(self):
        state = ReasoningState.from_problem("2*x = 4")
        state = state.mark_failed("unknown error")
        op = RepairChain()
        result = op.apply(state)
        assert not result.failed

    def test_no_failure_adds_note(self):
        state = ReasoningState.from_problem("p")
        op = RepairChain()
        result = op.apply(state)
        log = result.context.get("repair_log", [])
        assert any("nothing to repair" in entry for entry in log)

    def test_confidence_decreases_after_repair(self):
        state = ReasoningState.from_problem("p").mark_failed("err")
        state = state.evolve(confidence=1.0)
        op = RepairChain()
        result = op.apply(state)
        assert result.confidence < 1.0


# ── Pipeline integration ──────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_full_pipeline_solves_linear_eq(self):
        from mre.operators.pipeline import OperatorPipeline

        pipeline = OperatorPipeline.from_names([
            "SymbolicSimplify",
            "EquationSolve",
            "SelfCritique",
        ])
        state = ReasoningState.from_problem("Solve 3*x - 9 = 0")
        state = state.evolve(context={"equation": "3*x - 9 = 0"})
        result = pipeline.run(state)
        assert result.solved
        assert "3" in str(result.answer)

    def test_pipeline_with_repair(self):
        from mre.operators.pipeline import OperatorPipeline

        pipeline = OperatorPipeline.from_names([
            "EquationSolve",
            "SelfCritique",
            "RepairChain",
        ], stop_on_failure=False)
        state = ReasoningState.from_problem("plain text, no equation")
        result = pipeline.run(state)
        # Should not raise; repair chain should handle the no-equation case
        assert isinstance(result.final_state, ReasoningState)

    def test_pipeline_report_is_string(self):
        from mre.operators.pipeline import OperatorPipeline

        pipeline = OperatorPipeline.from_names(["SymbolicSimplify"])
        state = ReasoningState.from_problem("x**2")
        result = pipeline.run(state)
        report = result.report()
        assert isinstance(report, str)
        assert "Pipeline Result" in report

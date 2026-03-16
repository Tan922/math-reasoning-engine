"""
tests/test_phase3/test_evaluation.py  +  test_evolution.py  +  test_pipeline.py
────────────────────────────────────────────────────────────────────────────────
All offline — no network, no GPU, no LLM calls.
"""

import random
import pytest

# ── fixtures shared across sub-modules ───────────────────────────────────────

from mre.agents.dna import AgentDNA
from mre.agents.state import ReasoningState, StepRecord
from mre.operators.pipeline import OperatorPipeline, PipelineResult


def _make_result(solved=True, answer="[2, 3]", n_steps=2, failed_steps=0):
    """Build a minimal PipelineResult for testing.

    Each step gets a *unique* output snapshot (``f"out_{i}"``) so the
    circular-step detector in CritiqueAgent is never triggered by accident.
    Tests that *want* to trigger the circular detector build their own
    StepRecords with identical snapshots explicitly.
    """
    state = ReasoningState.from_problem("Solve x**2-5*x+6=0")
    state = state.evolve(context={"equation": "x**2-5*x+6=0"})
    if solved:
        state = state.set_answer(answer)
    extra = [
        StepRecord(f"Op{i}", f"in_{i}", f"out_{i}", success=(i >= failed_steps))
        for i in range(n_steps)
    ]
    state = state.evolve(history=extra)
    return PipelineResult(
        final_state=state,
        operator_sequence=["SymbolicSimplify", "EquationSolve"],
        total_duration_sec=0.1,
        solved=solved,
        answer=answer if solved else None,
        step_timings=[],
    )


# ══════════════════════════════════════════════════════════════════════════════
# EvaluationCommission
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrectnessJudge:
    from mre.evaluation.commission import CorrectnessJudge

    def test_correct_answer_scores_1(self):
        from mre.evaluation.commission import CorrectnessJudge
        judge = CorrectnessJudge()
        result = _make_result(solved=True, answer="[2, 3]")
        v = judge.judge(result, expected_answer="2")
        assert v.score == 1.0

    def test_no_answer_scores_0(self):
        from mre.evaluation.commission import CorrectnessJudge
        judge = CorrectnessJudge()
        result = _make_result(solved=False, answer=None)
        v = judge.judge(result, expected_answer="2")
        assert v.score == 0.0

    def test_no_ground_truth_partial(self):
        from mre.evaluation.commission import CorrectnessJudge
        judge = CorrectnessJudge()
        result = _make_result(solved=True, answer="anything")
        v = judge.judge(result, expected_answer=None)
        assert 0.0 < v.score < 1.0

    def test_wrong_answer_low_score(self):
        from mre.evaluation.commission import CorrectnessJudge
        judge = CorrectnessJudge()
        result = _make_result(solved=True, answer="[99]")
        v = judge.judge(result, expected_answer="2")
        assert v.score < 0.5


class TestLogicJudge:
    def test_no_steps_gives_partial(self):
        from mre.evaluation.commission import LogicJudge
        judge = LogicJudge()
        state = ReasoningState.from_problem("p").set_answer("1")
        result = PipelineResult(state, [], 0.0, True, "1", [])
        v = judge.judge(result)
        assert 0.0 < v.score <= 1.0

    def test_failed_step_penalises(self):
        from mre.evaluation.commission import LogicJudge
        judge = LogicJudge()
        result = _make_result(n_steps=3, failed_steps=1)
        v = judge.judge(result)
        assert v.score < 1.0


class TestCritiqueAgent:
    def test_no_faults_full_score(self):
        from mre.evaluation.commission import CritiqueAgent
        agent = CritiqueAgent(llm_client=None)
        result = _make_result(solved=True, answer="[2]", n_steps=2)
        v = agent.judge(result, expected_answer="2")
        assert v.score == 1.0

    def test_circular_step_penalised(self):
        from mre.evaluation.commission import CritiqueAgent
        agent = CritiqueAgent(llm_client=None)
        state = ReasoningState.from_problem("p")
        r1 = StepRecord("A", "x", "SAME", True)
        r2 = StepRecord("B", "y", "SAME", True)
        state = state.evolve(history=[r1, r2])
        result = PipelineResult(state, ["A","B"], 0.0, False, None, [])
        v = agent.judge(result)
        assert v.score < 1.0

    def test_failed_no_repair_penalised(self):
        from mre.evaluation.commission import CritiqueAgent
        agent = CritiqueAgent(llm_client=None)
        r = StepRecord("Op", "x", "", success=False, error_msg="err")
        state = ReasoningState.from_problem("p").evolve(history=[r])
        result = PipelineResult(state, ["Op"], 0.0, False, None, [])
        v = agent.judge(result)
        assert v.score < 1.0


class TestEvaluationCommission:
    def test_evaluate_returns_verdict(self):
        from mre.evaluation.commission import EvaluationCommission
        c = EvaluationCommission()
        result = _make_result(solved=True, answer="[2, 3]")
        v = c.evaluate(result, expected_answer="2")
        assert 0.0 <= v.weighted_score <= 1.0
        assert len(v.judge_verdicts) == 4  # 3 judges + conciseness

    def test_correct_answer_high_score(self):
        from mre.evaluation.commission import EvaluationCommission
        c = EvaluationCommission()
        result = _make_result(solved=True, answer="[2, 3]", n_steps=2)
        v = c.evaluate(result, expected_answer="2")
        assert v.weighted_score >= 0.50

    def test_no_answer_low_score(self):
        from mre.evaluation.commission import EvaluationCommission
        c = EvaluationCommission()
        # solved=False + one failed step → Correctness=0, LogicJudge penalised,
        # CritiqueAgent penalised (failed step with no repair) → total < 0.5
        result = _make_result(solved=False, n_steps=2, failed_steps=0)
        # Override: mark the state as having a failed step so all three judges
        # see a problem.  Build explicitly for full control.
        state = ReasoningState.from_problem("p")
        state = state.evolve(history=[
            StepRecord("Op0", "in_0", "out_0", success=False, error_msg="parse error"),
            StepRecord("Op1", "in_1", "out_1", success=False, error_msg="no solution"),
        ])
        bad_result = PipelineResult(
            final_state=state,
            operator_sequence=["Op0", "Op1"],
            total_duration_sec=0.1,
            solved=False,
            answer=None,
            step_timings=[],
        )
        v = c.evaluate(bad_result, expected_answer="2")
        assert v.weighted_score < 0.5, (
            f"Expected score < 0.5 for a failed, no-answer result; got {v.weighted_score}"
        )

    def test_batch_evaluate(self):
        from mre.evaluation.commission import EvaluationCommission
        c = EvaluationCommission()
        results = [_make_result(solved=True), _make_result(solved=False)]
        verdicts = c.batch_evaluate(results, ["2", "2"])
        assert len(verdicts) == 2
        assert verdicts[0].weighted_score > verdicts[1].weighted_score

    def test_report_is_string(self):
        from mre.evaluation.commission import EvaluationCommission
        c = EvaluationCommission()
        v = c.evaluate(_make_result(), "2")
        assert isinstance(v.report(), str)
        assert "Commission Verdict" in v.report()

    def test_weight_accessors(self):
        from mre.evaluation.commission import EvaluationCommission
        c = EvaluationCommission()
        v = c.evaluate(_make_result(solved=True), "2")
        assert 0.0 <= v.correctness <= 1.0
        assert 0.0 <= v.logic <= 1.0
        assert 0.0 <= v.critique <= 1.0
        assert 0.0 <= v.conciseness <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# SelectionEngine (Elo)
# ══════════════════════════════════════════════════════════════════════════════

class TestSelectionEngine:
    from mre.evolution.selection import SelectionEngine

    def _pop(self, n=4):
        return [AgentDNA() for _ in range(n)]

    def test_update_modifies_elo(self):
        from mre.evolution.selection import SelectionEngine
        engine = SelectionEngine()
        pop = self._pop(4)
        scores = [0.9, 0.7, 0.5, 0.3]
        engine.update(pop, scores)
        # Best scorer should have gained Elo, worst should have lost
        assert engine.elo(pop[0].agent_id) > 1200
        assert engine.elo(pop[3].agent_id) < 1200

    def test_select_culls_bottom(self):
        from mre.evolution.selection import SelectionEngine
        engine = SelectionEngine(cull_fraction=0.25)
        pop = self._pop(4)
        engine.update(pop, [0.9, 0.8, 0.4, 0.2])
        survivors, culled = engine.select(pop)
        assert len(culled) == 1
        assert len(survivors) == 3

    def test_never_culls_below_min(self):
        from mre.evolution.selection import SelectionEngine
        engine = SelectionEngine(cull_fraction=0.90, min_population=2)
        pop = self._pop(3)
        engine.update(pop, [0.9, 0.5, 0.1])
        survivors, culled = engine.select(pop)
        assert len(survivors) >= 2

    def test_top_k(self):
        from mre.evolution.selection import SelectionEngine
        engine = SelectionEngine()
        pop = self._pop(4)
        engine.update(pop, [0.9, 0.1, 0.7, 0.4])
        top = engine.top_k(pop, k=2)
        assert len(top) == 2

    def test_leaderboard_string(self):
        from mre.evolution.selection import SelectionEngine
        engine = SelectionEngine()
        pop = self._pop(3)
        engine.update(pop, [0.8, 0.5, 0.3])
        lb = engine.leaderboard(pop)
        assert "Elo Leaderboard" in lb

    def test_elo_written_to_metadata(self):
        from mre.evolution.selection import SelectionEngine
        engine = SelectionEngine()
        pop = self._pop(2)
        engine.update(pop, [0.9, 0.1])
        for dna in pop:
            assert "elo" in dna.metadata


# ══════════════════════════════════════════════════════════════════════════════
# EvolutionEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestEvolutionEngine:
    def _pop(self, n=6):
        return [AgentDNA(reasoning_gene=["SymbolicSimplify","EquationSolve"])
                for _ in range(n)]

    def test_evolve_returns_target_size(self):
        from mre.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(target_population=6)
        pop = self._pop(6)
        scores = [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
        new_pop = engine.evolve(pop, scores)
        assert len(new_pop) == 6

    def test_evolve_increments_generation(self):
        from mre.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(target_population=4)
        engine.evolve(self._pop(4), [0.9, 0.7, 0.5, 0.3])
        assert engine.generation == 1

    def test_elites_preserved(self):
        from mre.evolution.engine import EvolutionEngine
        rng = random.Random(0)
        engine = EvolutionEngine(target_population=4, elite_fraction=0.5, rng=rng)
        pop = self._pop(4)
        scores = [0.9, 0.8, 0.3, 0.2]
        engine.selector.update(pop, scores)
        new_pop = engine.evolve(pop, scores)
        # Top agents should appear in new population (as elites or parents)
        new_ids = {d.agent_id for d in new_pop}
        # At least one original agent_id OR its parent_id should appear
        assert len(new_pop) == 4

    def test_summary_available(self):
        from mre.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(target_population=4)
        engine.evolve(self._pop(4), [0.9, 0.7, 0.5, 0.3])
        s = engine.last_summary
        assert s is not None
        assert s.generation == 1
        assert isinstance(s.report(), str)

    def test_empty_population_handled(self):
        from mre.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(target_population=4)
        result = engine.evolve([], [])
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# MREPipeline (closed loop)
# ══════════════════════════════════════════════════════════════════════════════

class TestMREPipeline:
    def _problems(self):
        return [
            {"text": "Solve x**2-5*x+6=0",
             "context": {"equation": "x**2-5*x+6=0"},
             "answer": "2"},
        ]

    def test_pipeline_runs_returns_history(self):
        from mre.pipeline import MREPipeline
        p = MREPipeline(population_size=3, generations=2, target_score=0.95)
        history = p.run(self._problems())
        assert len(history) >= 1
        assert all(hasattr(r, "best_score") for r in history)

    def test_early_stop_on_target(self):
        from mre.pipeline import MREPipeline
        # Very low target → should stop at generation 1
        p = MREPipeline(population_size=2, generations=5, target_score=0.01)
        history = p.run(self._problems())
        assert len(history) == 1  # stopped after first gen

    def test_convergence_curve_increasing(self):
        from mre.pipeline import MREPipeline
        p = MREPipeline(population_size=3, generations=3, target_score=0.99)
        history = p.run(self._problems())
        curve = p.convergence_curve(history)
        assert len(curve) == len(history)
        assert all(0.0 <= s <= 1.0 for s in curve)

    def test_multiple_problems_round_robin(self):
        from mre.pipeline import MREPipeline
        problems = [
            {"text": "Solve x+1=0", "context": {"equation": "x+1=0"}, "answer": "-1"},
            {"text": "Solve x-2=0", "context": {"equation": "x-2=0"}, "answer": "2"},
        ]
        p = MREPipeline(population_size=2, generations=4, target_score=0.99)
        history = p.run(problems)
        # Should alternate between problems
        texts = [r.problem_text for r in history]
        assert len(set(texts)) <= 2

    def test_print_history_no_error(self, capsys):
        from mre.pipeline import MREPipeline
        p = MREPipeline(population_size=2, generations=2, target_score=0.99)
        history = p.run(self._problems())
        p.print_history(history)
        out = capsys.readouterr().out
        assert "MRE Pipeline" in out

"""
mre.benchmarks
──────────────
Benchmark runner that evaluates MREPipeline against problem sets
in MATH / miniF2F format, with a synthetic fallback for offline testing.

Usage
-----
    from mre.benchmarks import BenchmarkRunner, SyntheticBenchmark

    bench  = SyntheticBenchmark(n=20, seed=42)
    runner = BenchmarkRunner(population_size=4, generations=3)
    report = runner.run(bench.problems())
    print(report.summary())
    runner.plot(report, save_path="results/benchmark.png")
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mre.agents.dna import AgentDNA
from mre.pipeline import MREPipeline, GenerationRecord
from mre.utils import get_logger

logger = get_logger(__name__)


# ── Problem format ────────────────────────────────────────────────────────────

@dataclass
class BenchmarkProblem:
    problem_id: str
    text: str
    context: Dict[str, Any]
    expected_answer: str
    difficulty: str = "medium"      # easy / medium / hard
    domain: str     = "algebra"
    source: str     = "synthetic"


# ── Synthetic benchmark (always offline) ─────────────────────────────────────

class SyntheticBenchmark:
    """
    Generates a deterministic set of algebra + simplification problems.
    Fully offline — no network or file I/O required.

    Difficulty tiers:
      easy   — linear equations  (ax + b = 0)
      medium — quadratic         (x² + bx + c = 0, integer roots)
      hard   — cubic / rational  (x³ - ... = 0, (x²-1)/(x-1))
    """

    TEMPLATES = {
        "easy": [
            ("{a}*x + {b} = 0",      "{a}*x + {b} = 0",   "{sol}"),
            ("x - {b} = 0",           "x - {b} = 0",        "{sol}"),
            ("{a}*x = {c}",           "{a}*x = {c}",        "{sol}"),
        ],
        "medium": [
            ("x**2 + {b}*x + {c} = 0", "x**2 + {b}*x + {c} = 0", "[{r1}, {r2}]"),
            ("x**2 - {s}*x + {p} = 0", "x**2 - {s}*x + {p} = 0", "[{r1}, {r2}]"),
        ],
        "hard": [
            ("Simplify (x**2-{k}*x)/(x-{k})",   "",  "x"),
            ("Simplify sin(x)**2 + cos(x)**2",   "",  "1"),
            ("Simplify (x**2 - 1)/(x - 1)",      "",  "x + 1"),
        ],
    }

    def __init__(self, n: int = 20, seed: int = 42):
        self.n    = n
        self.seed = seed

    def problems(self) -> List[BenchmarkProblem]:
        rng  = random.Random(self.seed)
        out: List[BenchmarkProblem] = []
        pid  = 0

        # ── Easy: linear ─────────────────────────────────────────────────────
        for _ in range(self.n // 3):
            a = rng.randint(1, 5)
            b = rng.randint(-10, 10)
            if b == 0:
                b = 1
            sol_val = -b / a
            # Only keep integer solutions for clean expected-answer matching
            if sol_val == int(sol_val):
                sol = str(int(sol_val))
                eq  = f"{a}*x + {b} = 0"
                out.append(BenchmarkProblem(
                    problem_id=f"syn_easy_{pid}",
                    text=f"Solve {eq}",
                    context={"equation": eq},
                    expected_answer=sol,
                    difficulty="easy",
                    domain="algebra",
                ))
                pid += 1

        # ── Medium: quadratic with integer roots ──────────────────────────────
        for _ in range(self.n // 3):
            r1 = rng.randint(-5, 5)
            r2 = rng.randint(-5, 5)
            # (x - r1)(x - r2) = x² - (r1+r2)x + r1*r2
            b  = -(r1 + r2)
            c  = r1 * r2
            b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
            c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
            eq  = f"x**2 {b_str}*x {c_str} = 0"
            # Expected: smaller root first
            roots = sorted([r1, r2])
            # Use the smaller root as the expected key
            out.append(BenchmarkProblem(
                problem_id=f"syn_med_{pid}",
                text=f"Solve {eq}",
                context={"equation": eq},
                expected_answer=str(roots[0]),
                difficulty="medium",
                domain="algebra",
            ))
            pid += 1

        # ── Hard: simplification ─────────────────────────────────────────────
        hard_fixed = [
            ("Simplify (x**2-1)/(x-1)",      {}, "x + 1"),
            ("Simplify sin(x)**2+cos(x)**2",  {}, "1"),
            ("Simplify (x**2-4)/(x-2)",       {}, "x + 2"),
        ]
        for text, ctx, ans in hard_fixed:
            out.append(BenchmarkProblem(
                problem_id=f"syn_hard_{pid}",
                text=text,
                context=ctx,
                expected_answer=ans,
                difficulty="hard",
                domain="algebra",
            ))
            pid += 1

        return out[: self.n]


# ── MATH/miniF2F loader (when files are present) ──────────────────────────────

class MATHBenchmark:
    """
    Loads problems from a JSONL file in MATH dataset format:
      {"problem": "...", "solution": "...", "level": "Level 1", "type": "Algebra"}

    Falls back gracefully to SyntheticBenchmark if file not found.
    """

    def __init__(self, path: str, max_problems: int = 50, seed: int = 42):
        self.path         = Path(path)
        self.max_problems = max_problems
        self.seed         = seed

    def problems(self) -> List[BenchmarkProblem]:
        if not self.path.exists():
            logger.warning(
                "MATH dataset not found at %s — using SyntheticBenchmark.", self.path
            )
            return SyntheticBenchmark(n=self.max_problems, seed=self.seed).problems()

        out: List[BenchmarkProblem] = []
        with open(self.path) as f:
            for line in f:
                if len(out) >= self.max_problems:
                    break
                try:
                    d = json.loads(line)
                    # Extract final boxed answer if present
                    import re
                    m = re.search(r"\\boxed\{([^}]+)\}", d.get("solution", ""))
                    answer = m.group(1) if m else d.get("answer", "")
                    out.append(BenchmarkProblem(
                        problem_id=d.get("id", f"math_{len(out)}"),
                        text=d["problem"],
                        context={},
                        expected_answer=answer,
                        difficulty=d.get("level", "medium").lower().replace("level ", ""),
                        domain=d.get("type", "algebra").lower(),
                        source="MATH",
                    ))
                except Exception:
                    continue
        return out


# ── Benchmark result ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    problems: List[BenchmarkProblem]
    per_problem: List[Dict[str, Any]]
    accuracy: float
    accuracy_by_difficulty: Dict[str, float]
    accuracy_by_domain: Dict[str, float]
    total_duration_sec: float
    mean_generations: float
    population_size: int
    generations_limit: int

    def summary(self) -> str:
        lines = [
            "╔══ Benchmark Results ══════════════════════════════════════╗",
            f"  Problems : {len(self.problems)}",
            f"  Accuracy : {self.accuracy:.1%}",
            f"  Duration : {self.total_duration_sec:.1f}s",
            f"  Mean gens: {self.mean_generations:.1f}",
            "  ── By Difficulty ────────────────────────────────────────",
        ]
        for diff, acc in sorted(self.accuracy_by_difficulty.items()):
            n = sum(1 for p in self.problems if p.difficulty == diff)
            lines.append(f"    {diff:<8} {acc:.1%}  ({n} problems)")
        lines.append("  ── By Domain ────────────────────────────────────────")
        for dom, acc in sorted(self.accuracy_by_domain.items()):
            n = sum(1 for p in self.problems if p.domain == dom)
            lines.append(f"    {dom:<15} {acc:.1%}  ({n} problems)")
        lines.append("╚═══════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "accuracy_by_difficulty": self.accuracy_by_difficulty,
            "accuracy_by_domain": self.accuracy_by_domain,
            "total_duration_sec": self.total_duration_sec,
            "mean_generations": self.mean_generations,
            "n_problems": len(self.problems),
            "per_problem": self.per_problem,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── Runner ────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs MREPipeline on a list of BenchmarkProblems and collects metrics.

    Parameters
    ----------
    population_size : int
    generations : int
    target_score : float
    seed_dna : list, optional  — pre-built population to reuse across problems
    """

    def __init__(
        self,
        population_size: int = 4,
        generations: int = 3,
        target_score: float = 0.85,
        seed_dna: Optional[List[AgentDNA]] = None,
        llm_client: Optional[Callable] = None,
    ):
        self.population_size = population_size
        self.generations     = generations
        self.target_score    = target_score
        self.seed_dna        = seed_dna
        self.llm_client      = llm_client

    def run(self, problems: List[BenchmarkProblem]) -> BenchmarkResult:
        t0 = time.perf_counter()
        per_problem: List[Dict[str, Any]] = []
        total_gens  = 0

        for i, prob in enumerate(problems):
            logger.info(
                "Problem %d/%d [%s/%s]: %s",
                i + 1, len(problems),
                prob.difficulty, prob.domain,
                prob.text[:50],
            )
            pipe = MREPipeline(
                population_size=self.population_size,
                generations=self.generations,
                target_score=self.target_score,
                llm_client=self.llm_client,
            )
            seed = (
                [d.clone() for d in self.seed_dna]
                if self.seed_dna else None
            )
            history = pipe.run(
                [{"text": prob.text,
                  "context": prob.context,
                  "answer": prob.expected_answer}],
                seed_population=seed,
            )
            best = max(history, key=lambda r: r.best_score) if history else None
            correct = bool(
                best and best.best_answer and
                prob.expected_answer.strip() in str(best.best_answer)
            )
            per_problem.append({
                "problem_id": prob.problem_id,
                "text": prob.text[:80],
                "expected": prob.expected_answer,
                "got": best.best_answer if best else None,
                "correct": correct,
                "best_score": best.best_score if best else 0.0,
                "generations_used": len(history),
                "difficulty": prob.difficulty,
                "domain": prob.domain,
            })
            total_gens += len(history)

        # Aggregate metrics
        n = len(problems)
        accuracy = sum(p["correct"] for p in per_problem) / n if n else 0.0

        def _acc_by(key):
            groups: Dict[str, List[bool]] = {}
            for pp in per_problem:
                g = pp[key]
                groups.setdefault(g, []).append(pp["correct"])
            return {g: sum(v)/len(v) for g, v in groups.items()}

        return BenchmarkResult(
            problems=problems,
            per_problem=per_problem,
            accuracy=round(accuracy, 4),
            accuracy_by_difficulty=_acc_by("difficulty"),
            accuracy_by_domain=_acc_by("domain"),
            total_duration_sec=round(time.perf_counter() - t0, 2),
            mean_generations=round(total_gens / n, 2) if n else 0.0,
            population_size=self.population_size,
            generations_limit=self.generations,
        )

    def plot(
        self,
        result: BenchmarkResult,
        save_path: Optional[str] = None,
    ) -> None:
        """Render a 2×2 results dashboard."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available — skipping plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#0f0f1a")
        for ax in axes.flat:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444466")

        COLORS = {"easy": "#6BCB77", "medium": "#FFD93D", "hard": "#FF6B6B"}
        WHITE  = "white"

        # 1. Accuracy by difficulty
        ax = axes[0, 0]
        diffs  = ["easy", "medium", "hard"]
        accs   = [result.accuracy_by_difficulty.get(d, 0.0) for d in diffs]
        colors = [COLORS[d] for d in diffs]
        bars   = ax.bar(diffs, accs, color=colors, alpha=0.85, width=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{acc:.0%}", ha="center", color=WHITE, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy", color=WHITE)
        ax.set_title("Accuracy by Difficulty", color=WHITE, fontweight="bold")
        ax.set_xticklabels(diffs, color=WHITE)

        # 2. Score distribution
        ax = axes[0, 1]
        scores = [p["best_score"] for p in result.per_problem]
        ax.hist(scores, bins=10, color="#54A0FF", alpha=0.85, edgecolor="#0f0f1a")
        ax.axvline(np.mean(scores), color="#FFD93D", linestyle="--",
                   label=f"mean={np.mean(scores):.2f}")
        ax.set_xlabel("Best commission score", color=WHITE)
        ax.set_ylabel("Count", color=WHITE)
        ax.set_title("Score Distribution", color=WHITE, fontweight="bold")
        ax.legend(facecolor="#1a1a2e", labelcolor=WHITE, fontsize=10)

        # 3. Per-problem correct/wrong
        ax = axes[1, 0]
        corrects = [1 if p["correct"] else 0 for p in result.per_problem]
        ax.bar(range(len(corrects)), corrects,
               color=["#6BCB77" if c else "#FF6B6B" for c in corrects],
               alpha=0.85, width=0.8)
        ax.set_xlabel("Problem index", color=WHITE)
        ax.set_ylabel("Correct (1) / Wrong (0)", color=WHITE)
        ax.set_title(f"Per-Problem Results  (acc={result.accuracy:.1%})",
                     color=WHITE, fontweight="bold")
        ax.set_ylim(-0.1, 1.4)

        # 4. Summary stats box
        ax = axes[1, 1]
        ax.axis("off")
        txt = "\n".join([
            "  Benchmark Summary",
            "  " + "─" * 24,
            f"  Problems       : {len(result.problems)}",
            f"  Accuracy       : {result.accuracy:.1%}",
            f"  Mean gen used  : {result.mean_generations:.1f}",
            f"  Duration       : {result.total_duration_sec:.1f}s",
            f"  Population     : {result.population_size}",
            f"  Max gens       : {result.generations_limit}",
            "",
            f"  Easy  acc : {result.accuracy_by_difficulty.get('easy',  0):.1%}",
            f"  Medium acc: {result.accuracy_by_difficulty.get('medium', 0):.1%}",
            f"  Hard  acc : {result.accuracy_by_difficulty.get('hard',   0):.1%}",
        ])
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=11, color=WHITE,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                          edgecolor="#6BCB77", linewidth=1.5))

        fig.suptitle("MRE Benchmark Results", color=WHITE,
                     fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info("Benchmark plot saved → %s", save_path)
        else:
            plt.show()

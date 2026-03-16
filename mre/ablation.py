"""
mre.ablation
─────────────
AblationStudy — systematically ablates MRE components to measure each
component's contribution, as required by Phase-4.

Ablation conditions
-------------------
  full          MRE with evolution + commission scoring (full system)
  no_evolution  Fixed population, no crossover/mutation
  no_critique   Commission scores only correctness (40%) + logic (30%)
  no_repair     reasoning_gene excludes RepairChain
  single_agent  Population of 1 (no diversity)

Usage
-----
    from mre.ablation import AblationStudy
    from mre.benchmarks import SyntheticBenchmark

    study = AblationStudy(n_problems=12, generations=2)
    results = study.run()
    study.print_table(results)
    study.plot(results, save_path="results/ablation.png")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mre.agents.dna import AgentDNA
from mre.benchmarks import SyntheticBenchmark, BenchmarkProblem
from mre.pipeline import MREPipeline
from mre.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AblationCondition:
    name: str
    description: str
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed_dna_fn: Optional[Callable] = None  # () -> List[AgentDNA]


@dataclass
class AblationResult:
    condition: str
    accuracy: float
    mean_score: float
    mean_gens: float
    duration_sec: float
    per_problem: List[Dict[str, Any]] = field(default_factory=list)


class AblationStudy:
    """
    Runs the same problem set under multiple ablation conditions.

    Parameters
    ----------
    n_problems : int       — number of synthetic problems (default 12)
    generations : int      — max generations per condition (default 2)
    population_size : int  — agents per generation (default 4)
    seed : int             — random seed for reproducibility
    """

    CONDITIONS: List[AblationCondition] = [
        AblationCondition(
            name="full",
            description="Full MRE (evolution + commission)",
            pipeline_kwargs={"population_size": 4, "cull_fraction": 0.25},
        ),
        AblationCondition(
            name="no_evolution",
            description="Fixed population, no evolution",
            pipeline_kwargs={"population_size": 4, "cull_fraction": 0.0,
                             "mutation_rate": 0.0},
        ),
        AblationCondition(
            name="single_agent",
            description="Single agent (no diversity)",
            pipeline_kwargs={"population_size": 1, "cull_fraction": 0.0},
        ),
        AblationCondition(
            name="no_repair",
            description="No RepairChain operator",
            pipeline_kwargs={"population_size": 4},
            seed_dna_fn=lambda: [
                AgentDNA(reasoning_gene=["SymbolicSimplify", "EquationSolve",
                                         "DeductiveStep", "SelfCritique"])
                for _ in range(4)
            ],
        ),
        AblationCondition(
            name="no_simplify",
            description="No SymbolicSimplify (raw equation only)",
            pipeline_kwargs={"population_size": 4},
            seed_dna_fn=lambda: [
                AgentDNA(reasoning_gene=["EquationSolve", "SelfCritique",
                                         "RepairChain"])
                for _ in range(4)
            ],
        ),
    ]

    def __init__(
        self,
        n_problems: int = 12,
        generations: int = 2,
        population_size: int = 4,
        seed: int = 42,
        conditions: Optional[List[str]] = None,
    ):
        self.n_problems      = n_problems
        self.generations     = generations
        self.population_size = population_size
        self.seed            = seed
        self._selected_conds = conditions  # None = all

    def _get_conditions(self) -> List[AblationCondition]:
        if self._selected_conds is None:
            return self.CONDITIONS
        return [c for c in self.CONDITIONS if c.name in self._selected_conds]

    def run(self) -> List[AblationResult]:
        bench    = SyntheticBenchmark(n=self.n_problems, seed=self.seed)
        problems = bench.problems()
        results: List[AblationResult] = []

        for cond in self._get_conditions():
            logger.info("Ablation: %s — %s", cond.name, cond.description)
            t0 = time.perf_counter()

            kwargs = {
                "generations":     self.generations,
                "population_size": self.population_size,
                "target_score":    0.99,   # don't early-stop — run all gens
                **cond.pipeline_kwargs,
            }

            per_problem: List[Dict[str, Any]] = []
            total_gens  = 0

            for prob in problems:
                pipe = MREPipeline(**kwargs)
                seed_dna = cond.seed_dna_fn() if cond.seed_dna_fn else None
                history  = pipe.run(
                    [{"text": prob.text,
                      "context": prob.context,
                      "answer": prob.expected_answer}],
                    seed_population=seed_dna,
                )
                best = max(history, key=lambda r: r.best_score) if history else None
                correct = bool(
                    best and best.best_answer and
                    prob.expected_answer.strip() in str(best.best_answer)
                )
                per_problem.append({
                    "correct": correct,
                    "best_score": best.best_score if best else 0.0,
                    "gens": len(history),
                })
                total_gens += len(history)

            n      = len(problems)
            acc    = sum(p["correct"] for p in per_problem) / n if n else 0.0
            mscore = sum(p["best_score"] for p in per_problem) / n if n else 0.0
            mgens  = total_gens / n if n else 0.0

            results.append(AblationResult(
                condition=cond.name,
                accuracy=round(acc, 4),
                mean_score=round(mscore, 4),
                mean_gens=round(mgens, 2),
                duration_sec=round(time.perf_counter() - t0, 2),
                per_problem=per_problem,
            ))
            logger.info(
                "  %s → acc=%.1f%%  mean_score=%.4f  t=%.1fs",
                cond.name, acc * 100, mscore,
                time.perf_counter() - t0,
            )

        return results

    def print_table(self, results: List[AblationResult]) -> None:
        print("╔══ Ablation Study ════════════════════════════════════════════╗")
        print("  {:<18} {:>8} {:>10} {:>8} {:>10}".format(
            "Condition", "Acc", "MeanScore", "MeanGen", "Time(s)"
        ))
        print("  " + "─" * 56)

        # Sort by accuracy descending
        ranked = sorted(results, key=lambda r: r.accuracy, reverse=True)
        best_acc = ranked[0].accuracy if ranked else 0.0

        for r in ranked:
            delta = f"  (Δ{r.accuracy - best_acc:+.1%})" if r.accuracy < best_acc else "  ← best"
            print("  {:<18} {:>7.1%} {:>10.4f} {:>8.1f} {:>10.2f}{}".format(
                r.condition, r.accuracy, r.mean_score,
                r.mean_gens, r.duration_sec, delta,
            ))
        print("╚══════════════════════════════════════════════════════════════╝")

    def plot(
        self,
        results: List[AblationResult],
        save_path: Optional[str] = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available — skipping plot.")
            return

        names  = [r.condition.replace("_", "\n") for r in results]
        accs   = [r.accuracy for r in results]
        scores = [r.mean_score for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0f0f1a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444466")

        x      = np.arange(len(names))
        colors = ["#6BCB77" if a == max(accs) else "#54A0FF" for a in accs]

        # Accuracy bars
        bars = ax1.bar(x, accs, color=colors, alpha=0.85, width=0.6)
        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{acc:.0%}", ha="center", color="white", fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, color="white")
        ax1.set_ylim(0, 1.2)
        ax1.set_ylabel("Accuracy", color="white")
        ax1.set_title("Ablation: Accuracy by Condition",
                      color="white", fontweight="bold")
        ax1.axhline(max(accs), color="#FFD93D", linestyle="--",
                    alpha=0.5, linewidth=1)

        # Mean score bars
        score_colors = ["#6BCB77" if s == max(scores) else "#FF9FF3" for s in scores]
        bars2 = ax2.bar(x, scores, color=score_colors, alpha=0.85, width=0.6)
        for bar, sc in zip(bars2, scores):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{sc:.3f}", ha="center", color="white", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, color="white")
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Mean Commission Score", color="white")
        ax2.set_title("Ablation: Commission Score by Condition",
                      color="white", fontweight="bold")

        fig.suptitle("MRE Ablation Study — Component Contributions",
                     color="white", fontsize=15, fontweight="bold")
        plt.tight_layout()

        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info("Ablation plot saved → %s", save_path)
        else:
            plt.show()

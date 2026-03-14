"""
mre.knowledge_graph.experiments
─────────────────────────────────
Composition hypothesis experiment.

Strategies
----------
from_scratch      Random init + fine-tune on k examples.
compose_fixed     f_r2 ∘ f_r1 directly, no fine-tuning.
compose_finetune  Compositional init (distillation) + fine-tune on k examples.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple, Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from mre.knowledge_graph.nro_model import NROModel
from mre.knowledge_graph.training import hits_at_k, train_epoch
from mre.utils import get_logger

logger = get_logger(__name__)

KShot   = Union[int, str]
Edge    = Tuple[int, int]
Results = Dict[str, Dict[KShot, Tuple[float, float]]]

STRATEGIES = ("from_scratch", "compose_fixed", "compose_finetune")


def _run_one_trial(
    strategy: str,
    k_shot: KShot,
    base_model: NROModel,
    comp_relation: str,
    comp_chain: List[str],
    train_pool: List[Edge],
    test_edges: List[Edge],
    ft_epochs: int,
    ft_lr: float,
    hits_k: int,
) -> float:
    model = copy.deepcopy(base_model)

    k = len(train_pool) if k_shot == "full" else int(k_shot)
    k = min(k, len(train_pool))
    few_train = random.sample(train_pool, k) if k < len(train_pool) else list(train_pool)

    if strategy == "compose_fixed":
        return hits_at_k(model, test_edges, comp_chain, k=hits_k)
    elif strategy == "from_scratch":
        model.add_operator(comp_relation)
    elif strategy == "compose_finetune":
        model.compose_init(comp_relation, comp_chain)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    for name, op in model.operators.items():
        for p in op.parameters():
            p.requires_grad = (name == comp_relation)
    model.entity_emb.weight.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return hits_at_k(model, test_edges, [comp_relation], k=hits_k)

    opt = torch.optim.AdamW(trainable, lr=ft_lr, weight_decay=1e-4)
    if k >= 4:
        for _ in range(ft_epochs):
            train_epoch(model, few_train, [comp_relation], opt, batch_size=min(32, k))

    return hits_at_k(model, test_edges, [comp_relation], k=hits_k)


def run_composition_experiment(
    base_model: NROModel,
    comp_relation: str,
    comp_chain: List[str],
    train_pool: List[Edge],
    val_edges: List[Edge],
    test_edges: List[Edge],
    k_shots: List[KShot],
    n_trials: int = 5,
    ft_epochs: int = 40,
    ft_lr: float = 5e-4,
    hits_k: int = 10,
) -> Results:
    """Run full composition experiment; return results[strategy][k] = (mean, std)."""
    results: Results = {s: {} for s in STRATEGIES}
    logger.info("Composition experiment: %s = %s", comp_relation, " ∘ ".join(comp_chain))

    for k_shot in k_shots:
        logger.info("  k = %s", k_shot)
        for strategy in STRATEGIES:
            scores = [
                _run_one_trial(
                    strategy, k_shot, base_model,
                    comp_relation, comp_chain,
                    train_pool, test_edges,
                    ft_epochs, ft_lr, hits_k,
                )
                for _ in range(n_trials)
            ]
            mean_s, std_s = float(np.mean(scores)), float(np.std(scores))
            results[strategy][k_shot] = (mean_s, std_s)
            logger.info("    %-22s  %.3f ± %.3f", strategy, mean_s, std_s)

    return results


def print_results_table(results: Results, k_shots: List[KShot]) -> None:
    col_w = 16
    header = f"{'k':>6}  " + "  ".join(f"{s:{col_w}}" for s in STRATEGIES)
    sep = "=" * len(header)
    print(sep)
    print(f"{'Hits@10 by strategy and k-shot':^{len(header)}}")
    print(sep)
    print(header)
    print("-" * len(header))
    for k in k_shots:
        row = f"{str(k):>6}  "
        for s in STRATEGIES:
            mean, std = results[s].get(k, (0.0, 0.0))
            row += f"{f'{mean:.3f} ± {std:.3f}':{col_w}}  "
        print(row)
    print(sep)

    if 5 in results["compose_finetune"] and "full" in results["from_scratch"]:
        cf5      = results["compose_finetune"][5][0]
        sc5      = results["from_scratch"][5][0]
        sc_full  = results["from_scratch"]["full"][0] or 1.0
        print(f"\nKey finding (k=5):")
        print(f"  Compose+FT = {cf5:.3f}   From scratch = {sc5:.3f}")
        print(f"  Compose+FT@5 achieves {cf5/sc_full*100:.1f}% of full-data from-scratch")
        verdict = "✅ HYPOTHESIS SUPPORTED" if cf5 > sc5 * 1.5 else "⚠  MIXED RESULTS"
        print(f"  {verdict}")


def plot_results(results: Results, history: dict, k_shots: List[KShot]):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    COLORS = {
        "from_scratch":     "#FF6B6B",
        "compose_fixed":    "#FFD93D",
        "compose_finetune": "#6BCB77",
    }
    LABELS = {
        "from_scratch":     "From Scratch",
        "compose_fixed":    "Compose (fixed)",
        "compose_finetune": "Compose + Finetune  ★",
    }

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    x = np.arange(len(k_shots))

    # Plot 1: few-shot learning curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#1a1a2e")
    for strategy in STRATEGIES:
        means = [results[strategy][k][0] for k in k_shots]
        stds  = [results[strategy][k][1] for k in k_shots]
        ax1.plot(x, means, "-o", color=COLORS[strategy], label=LABELS[strategy],
                 linewidth=2.5, markersize=8, zorder=3)
        ax1.fill_between(x,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.15, color=COLORS[strategy])
    if 5 in k_shots:
        k5_idx = list(k_shots).index(5)
        ax1.axvline(x=k5_idx, color="white", linestyle="--", alpha=0.3, linewidth=1)
        for s in ("from_scratch", "compose_finetune"):
            val = results[s].get(5, (0, 0))[0]
            ax1.annotate(f"{val:.2f}", xy=(k5_idx, val),
                         xytext=(k5_idx + 0.15, val + 0.03),
                         color=COLORS[s], fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in k_shots], color="white", fontsize=11)
    ax1.set_xlabel("Training examples (k)", color="white", fontsize=12)
    ax1.set_ylabel("Hits@10", color="white", fontsize=12)
    ax1.set_title("Few-shot learning: gen_dep = generalizes ∘ depends_on",
                  color="white", fontsize=14, fontweight="bold", pad=12)
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#444466")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=11,
               loc="lower right", framealpha=0.8)
    ax1.grid(True, alpha=0.15, color="white")

    # Plot 2: training loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#1a1a2e")
    ax2.plot(history.get("loss", []), color="#4FC3F7", linewidth=2)
    ax2.set_title("Base Model Training Loss", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Loss", color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#444466")
    ax2.grid(True, alpha=0.15, color="white")

    # Plot 3: val hits per relation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1a1a2e")
    rel_colors = ["#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3"]
    for rel, col in zip(history.get("val_hits", {}).keys(), rel_colors):
        vals = history["val_hits"][rel]
        if vals:
            ax3.plot([(i+1)*10 for i in range(len(vals))], vals, "-o",
                     color=col, label=rel.replace("_", " "), linewidth=2, markersize=5)
    ax3.set_title("Validation Hits@10 (base)", color="white", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", color="white"); ax3.set_ylabel("Hits@10", color="white")
    ax3.tick_params(colors="white")
    for sp in ax3.spines.values(): sp.set_edgecolor("#444466")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax3.grid(True, alpha=0.15, color="white")

    # Plot 4: efficiency bar
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1a1a2e")
    full_scratch = results["from_scratch"].get("full", (1.0, 0))[0] or 1.0
    threshold = 0.80 * full_scratch
    bar_data = {}
    for strategy in ("from_scratch", "compose_finetune"):
        for k in k_shots:
            if k == "full": continue
            if results[strategy].get(k, (0,))[0] >= threshold:
                bar_data[strategy] = k; break
        else:
            bar_data[strategy] = "full"
    bar_vals = [bar_data.get("from_scratch", 100), bar_data.get("compose_finetune", 5)]
    bar_num  = [v if isinstance(v, int) else 100 for v in bar_vals]
    bars = ax4.bar(["From Scratch", "Compose+FT"], bar_num,
                   color=[COLORS["from_scratch"], COLORS["compose_finetune"]],
                   alpha=0.85, width=0.5)
    for bar, val in zip(bars, bar_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"k={val}", ha="center", color="white", fontsize=12, fontweight="bold")
    ax4.set_title("Examples to reach 80%\nof full-data performance",
                  color="white", fontsize=11, fontweight="bold")
    ax4.set_ylabel("k examples", color="white")
    ax4.tick_params(colors="white")
    for sp in ax4.spines.values(): sp.set_edgecolor("#444466")
    ax4.grid(True, alpha=0.15, color="white", axis="y")

    fig.suptitle("NRO: Compositional Init Enables Few-Shot Relation Learning",
                 color="white", fontsize=16, fontweight="bold", y=0.98)
    return fig

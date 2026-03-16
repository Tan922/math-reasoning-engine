"""
mre.evolution.selection
────────────────────────
SelectionEngine — Elo-based ranking + survival selection for the agent pool.

Elo rules
---------
* After each evaluation round, every agent's score is compared pair-wise
  against the population mean.  Win → Elo goes up, loss → Elo goes down.
* K-factor = 32 (standard chess).
* Agents with Elo below a survival threshold are marked for replacement.
* The bottom ``cull_fraction`` of the population is culled each generation.

Usage
-----
    from mre.evolution.selection import SelectionEngine
    from mre.agents.dna import AgentDNA

    engine = SelectionEngine(cull_fraction=0.20)
    engine.update(population, scores)           # updates Elo ratings in-place
    survivors, culled = engine.select(population)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mre.agents.dna import AgentDNA
from mre.utils import get_logger

logger = get_logger(__name__)

_K = 32.0   # Elo K-factor


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _new_elo(rating: float, score: float, expected: float) -> float:
    """Update Elo given actual score (1=win, 0.5=draw, 0=loss)."""
    return rating + _K * (score - expected)


# ── Rating store ──────────────────────────────────────────────────────────────

@dataclass
class AgentRating:
    agent_id: str
    elo: float = 1200.0
    games: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


# ── Selection engine ──────────────────────────────────────────────────────────

class SelectionEngine:
    """
    Elo-based selection engine.

    Parameters
    ----------
    cull_fraction : float
        Fraction of the population to cull each generation (default 0.20 = 20%).
    initial_elo : float
        Starting Elo for new agents (default 1200).
    min_population : int
        Never cull below this many agents (safety floor).
    """

    def __init__(
        self,
        cull_fraction: float = 0.20,
        initial_elo: float = 1200.0,
        min_population: int = 2,
    ):
        self.cull_fraction  = cull_fraction
        self.initial_elo    = initial_elo
        self.min_population = min_population
        self._ratings: Dict[str, AgentRating] = {}
        self._generation: int = 0

    # ── Rating access ─────────────────────────────────────────────────────────

    def _get_rating(self, agent_id: str) -> AgentRating:
        if agent_id not in self._ratings:
            self._ratings[agent_id] = AgentRating(
                agent_id=agent_id, elo=self.initial_elo
            )
        return self._ratings[agent_id]

    def elo(self, agent_id: str) -> float:
        return self._get_rating(agent_id).elo

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        population: List[AgentDNA],
        scores: List[float],
    ) -> None:
        """
        Update Elo ratings for each agent in *population* given *scores*.

        Strategy: each agent plays a "virtual match" against the population
        mean.  Score > mean_score → win; score < mean_score → loss;
        score == mean_score → draw.
        """
        if not population or not scores:
            return

        mean_score = sum(scores) / len(scores)
        mean_elo   = sum(self.elo(d.agent_id) for d in population) / len(population)

        for dna, score in zip(population, scores):
            rating = self._get_rating(dna.agent_id)
            expected = _expected_score(rating.elo, mean_elo)

            # Convert raw score to win/draw/loss
            if score > mean_score + 1e-6:
                actual = 1.0
                rating.wins += 1
            elif score < mean_score - 1e-6:
                actual = 0.0
                rating.losses += 1
            else:
                actual = 0.5  # draw

            rating.elo = _new_elo(rating.elo, actual, expected)
            rating.games += 1

            # Write Elo back into DNA metadata for downstream use
            dna.fitness_score = score
            dna.metadata["elo"] = round(rating.elo, 2)
            dna.metadata["elo_games"] = rating.games

        self._generation += 1
        logger.info(
            "Elo updated (gen %d): mean_elo=%.1f  mean_score=%.4f",
            self._generation, mean_elo, mean_score,
        )

    # ── Selection ─────────────────────────────────────────────────────────────

    def select(
        self,
        population: List[AgentDNA],
    ) -> Tuple[List[AgentDNA], List[AgentDNA]]:
        """
        Partition population into (survivors, culled).

        Culled = bottom ``cull_fraction`` by Elo, but never fewer than
        ``min_population`` survivors.
        """
        if len(population) <= self.min_population:
            return list(population), []

        ranked = sorted(
            population,
            key=lambda d: self.elo(d.agent_id),
            reverse=True,
        )

        n_cull = max(0, min(
            int(len(population) * self.cull_fraction),
            len(population) - self.min_population,
        ))

        survivors = ranked[: len(ranked) - n_cull]
        culled    = ranked[len(ranked) - n_cull :]

        logger.info(
            "Selection: %d survivors, %d culled (bottom %.0f%%)",
            len(survivors), len(culled), self.cull_fraction * 100,
        )
        return survivors, culled

    # ── Leaderboard ───────────────────────────────────────────────────────────

    def leaderboard(self, population: List[AgentDNA], top_k: int = 10) -> str:
        rated = sorted(
            [(d, self._get_rating(d.agent_id)) for d in population],
            key=lambda x: x[1].elo,
            reverse=True,
        )[:top_k]

        lines = [
            "╔══ Elo Leaderboard ════════════════════════════════════════╗",
            f"  Generation: {self._generation}",
            "  {:<10} {:>7} {:>6} {:>7} {:>7}".format(
                "Agent ID", "Elo", "Games", "Win%", "Fitness"
            ),
            "  " + "─" * 48,
        ]
        for dna, r in rated:
            lines.append(
                "  {:<10} {:>7.1f} {:>6} {:>6.1f}% {:>7.4f}".format(
                    dna.agent_id, r.elo, r.games,
                    r.win_rate * 100,
                    dna.fitness_score or 0.0,
                )
            )
        lines.append("╚═══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ── Top-K helpers ─────────────────────────────────────────────────────────

    def top_k(self, population: List[AgentDNA], k: int = 2) -> List[AgentDNA]:
        return sorted(
            population,
            key=lambda d: self.elo(d.agent_id),
            reverse=True,
        )[:k]

    def __repr__(self) -> str:
        return (
            f"SelectionEngine(agents={len(self._ratings)}, "
            f"gen={self._generation}, "
            f"cull={self.cull_fraction:.0%})"
        )

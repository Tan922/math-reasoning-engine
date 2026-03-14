"""
mre.knowledge_graph.synthetic_kg
──────────────────────────────────
Synthetic knowledge graph with known ground-truth transformations.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

    class _FakeNN:
        class Module:
            def __init__(self, *a, **kw): pass
        class Sequential:
            def __init__(self, *a, **kw): pass
            def __call__(self, x): return x
        Linear = Sequential
        Tanh   = Sequential

    nn = _FakeNN()  # type: ignore[assignment]

from mre.utils import get_logger

logger = get_logger(__name__)

Edge = Tuple[int, int]


@dataclass
class SyntheticKG:
    """Synthetic math-like knowledge graph with known ground-truth transforms."""

    num_entities: int = 500
    embed_dim: int = 64
    noise_std: float = 0.05
    relations: List[str] = field(default_factory=lambda: [
        "depends_on", "generalizes", "equivalent_to", "applied_in"
    ])
    seed: int = 42

    def __post_init__(self) -> None:
        d = self.embed_dim
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        self.true_entity_emb = F.normalize(
            torch.randn(self.num_entities, d, generator=rng) * 0.5, dim=-1
        )

        self.true_transforms: Dict[str, nn.Module] = {}
        gt_rng = torch.Generator()
        gt_rng.manual_seed(0)
        for rel in self.relations:
            net = nn.Sequential(
                nn.Linear(d, d * 2),
                nn.Tanh(),
                nn.Linear(d * 2, d),
            )
            nn.init.orthogonal_(net[0].weight)
            nn.init.orthogonal_(net[2].weight)
            self.true_transforms[rel] = net

        self.composite_relations: Dict[str, List[str]] = {
            "gen_dep":      ["generalizes", "depends_on"],
            "indirect_dep": ["depends_on",  "depends_on"],
        }

        self.edges: Dict[str, List[Edge]] = defaultdict(list)
        self._generate_edges()
        logger.info(
            "SyntheticKG — %d entities, %d base + %d composite relations",
            self.num_entities, len(self.relations), len(self.composite_relations),
        )

    def _apply_transform(self, h_idx: int, relation: str) -> "torch.Tensor":
        x = self.true_entity_emb[h_idx].unsqueeze(0)
        with torch.no_grad():
            return self.true_transforms[relation](x).squeeze(0)

    def _apply_composite(self, h_idx: int, chain: List[str]) -> "torch.Tensor":
        x = self.true_entity_emb[h_idx].unsqueeze(0)
        with torch.no_grad():
            for rel in chain:
                x = self.true_transforms[rel](x)
        return x.squeeze(0)

    def _nearest_entity(self, target_emb: "torch.Tensor", exclude: int) -> int:
        dists = ((self.true_entity_emb - target_emb.unsqueeze(0)) ** 2).sum(-1)
        dists[exclude] = float("inf")
        return int(dists.argmin().item())

    def _generate_edges(self, edges_per_relation: int = 400) -> None:
        n = min(edges_per_relation, self.num_entities)
        heads = random.sample(range(self.num_entities), n)

        for rel in self.relations:
            for h in heads:
                target = self._apply_transform(h, rel)
                noisy = target + torch.randn_like(target) * self.noise_std
                t = self._nearest_entity(noisy, exclude=h)
                self.edges[rel].append((h, t))

        for comp_rel, chain in self.composite_relations.items():
            for h in heads:
                target = self._apply_composite(h, chain)
                noisy = target + torch.randn_like(target) * self.noise_std
                t = self._nearest_entity(noisy, exclude=h)
                self.edges[comp_rel].append((h, t))

    def get_split(
        self,
        relation: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[Edge], List[Edge], List[Edge]]:
        edges = list(self.edges[relation])
        random.shuffle(edges)
        n = len(edges)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return edges[:n_train], edges[n_train: n_train + n_val], edges[n_train + n_val:]

    def summary(self) -> str:
        lines = [f"SyntheticKG (entities={self.num_entities}, dim={self.embed_dim})"]
        for rel, edges in self.edges.items():
            lines.append(f"  {rel:25s}: {len(edges)} edges")
        return "\n".join(lines)

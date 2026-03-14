"""
mre.knowledge_graph.mathkg_loader
───────────────────────────────────
Load MathKG from disk. Drop-in replacement for SyntheticKG.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from mre.utils import get_logger

logger = get_logger(__name__)

Edge = Tuple[int, int]


class MathKGLoader:
    """Load MathKG from disk; same interface as SyntheticKG."""

    def __init__(self, data_dir: str = "data/mathkg/"):
        import pandas as pd
        self.data_dir = Path(data_dir)

        if not (self.data_dir / "entities.tsv").exists():
            logger.warning("MathKG not found at '%s'. Run 02_MathKG_builder first.", data_dir)
            self._init_empty()
            return

        ent_df = pd.read_csv(self.data_dir / "entities.tsv", sep="\t")
        self.num_entities   = len(ent_df)
        self.entity_names   = ent_df["name"].tolist()
        self.entity_types   = ent_df["type"].tolist()
        self.name_to_id     = {n: i for i, n in enumerate(self.entity_names)}

        stats_path = self.data_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as fh:
                self.stats = json.load(fh)
            self.relations = self.stats["relations"]
        else:
            self.relations = ["depends_on", "generalizes", "equivalent_to", "applied_in"]
            self.stats = {}

        rel_df = pd.read_csv(self.data_dir / "relations.tsv", sep="\t")
        self.edges: Dict[str, List[Edge]] = {rel: [] for rel in self.relations}
        for rel in self.relations:
            sub = rel_df[rel_df["relation"] == rel]
            self.edges[rel] = list(zip(sub["head_id"].tolist(), sub["tail_id"].tolist()))

        emb_path = self.data_dir / "entity_embeddings.npy"
        if emb_path.exists():
            arr = np.load(str(emb_path))
            self.pretrained_embeddings: Optional["torch.Tensor"] = \
                torch.tensor(arr, dtype=torch.float32) if _TORCH_AVAILABLE else None
            self.embed_dim = arr.shape[1]
        else:
            self.pretrained_embeddings = None
            self.embed_dim = 64

        logger.info("MathKGLoader — %d entities, %d relations", self.num_entities, len(self.relations))

    def _init_empty(self) -> None:
        self.num_entities = 0
        self.entity_names = []
        self.entity_types = []
        self.name_to_id   = {}
        self.relations    = []
        self.edges        = {}
        self.stats        = {}
        self.pretrained_embeddings = None
        self.embed_dim    = 64

    def get_split(
        self,
        relation: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[List[Edge], List[Edge], List[Edge]]:
        import pandas as pd
        split_dir = self.data_dir / "splits"

        def _safe(path: Path) -> List[Edge]:
            if not path.exists():
                return []
            try:
                df = pd.read_csv(path, sep="\t")
                if df.empty or "head_id" not in df.columns:
                    return []
                return list(zip(df["head_id"].tolist(), df["tail_id"].tolist()))
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)
                return []

        train_p = split_dir / f"{relation}_train.tsv"
        if train_p.exists():
            return (
                _safe(train_p),
                _safe(split_dir / f"{relation}_val.tsv"),
                _safe(split_dir / f"{relation}_test.tsv"),
            )

        edges = list(self.edges.get(relation, []))
        random.shuffle(edges)
        n = len(edges)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return edges[:n_train], edges[n_train: n_train + n_val], edges[n_train + n_val:]

    def usable_relations(self, min_train: int = 4) -> List[str]:
        usable = []
        for rel in self.relations:
            train, _, _ = self.get_split(rel)
            if len(train) >= min_train:
                usable.append(rel)
            else:
                logger.info("Skipping '%s': only %d train examples", rel, len(train))
        return usable

    def get_pretrained_embedding_init(self, target_dim: int = 64):
        if not _TORCH_AVAILABLE or self.pretrained_embeddings is None or self.num_entities == 0:
            return None
        emb = self.pretrained_embeddings
        if emb.shape[1] == target_dim:
            return emb.clone()
        centred = emb - emb.mean(0)
        _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
        return F.normalize(centred @ Vt[:target_dim].T, dim=-1)

    def summary(self) -> str:
        lines = [f"MathKGLoader (entities={self.num_entities}, dir={self.data_dir})"]
        for rel in self.relations:
            lines.append(f"  {rel:25s}: {len(self.edges.get(rel, []))} triples")
        return "\n".join(lines)

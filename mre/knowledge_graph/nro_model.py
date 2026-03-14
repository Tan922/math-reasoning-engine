"""
mre.knowledge_graph.nro_model
──────────────────────────────
Neural Relation Operators (NRO) model.

Each relation is represented as a learnable operator  f_r : ℝ^d → ℝ^d.
Multi-hop reasoning = operator composition:
    f_r2∘f_r1(h)  →  predicted tail embedding

Key capability: compose_init()
    Initialise a NEW operator for an unseen composite relation by distilling
    the function  f_r2∘f_r1  onto the new operator's weights.
"""

from __future__ import annotations

from typing import List, Optional

# torch imported lazily — package stays importable in numpy-only envs
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

    class _FakeModule:
        def __init__(self, *a, **kw): pass
        def parameters(self): return iter([])
        def to(self, *a, **kw): return self
        def eval(self): return self
        def train(self, mode=True): return self
        def state_dict(self): return {}
        def load_state_dict(self, *a, **kw): pass

    class _FakeNN:
        Module = _FakeModule
        ModuleDict = dict
        Embedding = _FakeModule
        Linear = _FakeModule
        Sequential = _FakeModule
        LayerNorm = _FakeModule
        GELU = _FakeModule

    nn = _FakeNN()  # type: ignore[assignment]

from mre.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class NeuralRelationOperator(nn.Module):
    """
    One learnable operator per relation type.

        f_r(x)  =  MLP_r(x)  +  W_shared · x
                   ↑ relation-specific    ↑ shared residual (identity init)
    """

    def __init__(self, embed_dim: int, hidden_dim: int, shared_residual):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.residual = shared_residual

    def forward(self, x):
        return self.mlp(x) + self.residual(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class NROModel(nn.Module):
    """
    Full NRO model:
      • Learnable entity embeddings
      • One NeuralRelationOperator per relation
      • Compositional reasoning = operator composition
    """

    def __init__(
        self,
        num_entities: int,
        relation_names: List[str],
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.relation_names = list(relation_names)

        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        nn.init.normal_(self.entity_emb.weight, std=0.1)

        self.shared_residual = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.shared_residual.weight)

        self.operators = nn.ModuleDict({
            name: NeuralRelationOperator(embed_dim, hidden_dim, self.shared_residual)
            for name in relation_names
        })

    # ── Inference ─────────────────────────────────────────────────────────────

    def reason(self, head_ids, relation_chain: List[str]):
        """Multi-hop reasoning: apply operators in sequence."""
        x = self.entity_emb(head_ids)
        for rel in relation_chain:
            if rel not in self.operators:
                raise KeyError(
                    f"Unknown relation '{rel}'. Known: {list(self.operators.keys())}"
                )
            x = self.operators[rel](x)
        return x

    def score(self, head_ids, relation_chain: List[str], tail_ids):
        """Cosine similarity between predicted tail and actual tail embedding."""
        pred = self.reason(head_ids, relation_chain)
        tail = self.entity_emb(tail_ids)
        return F.cosine_similarity(pred, tail, dim=-1)

    # ── Dynamic operator management ───────────────────────────────────────────

    def add_operator(self, name: str) -> None:
        """Add a new operator with random init."""
        if name in self.operators:
            logger.warning("Operator '%s' already exists — skipping.", name)
            return
        hidden_dim = self.operators[self.relation_names[0]].mlp[0].out_features
        self.operators[name] = NeuralRelationOperator(
            self.embed_dim, hidden_dim, self.shared_residual
        ).to(next(self.parameters()).device)
        logger.debug("Added operator '%s' (random init).", name)

    def compose_init(self, new_name: str, chain: List[str]) -> None:
        """
        Initialise a new operator as a functional approximation of the
        composition  f_chain[-1] ∘ … ∘ f_chain[0]  via MSE distillation.

        This is the key method enabling few-shot learning of new relations.
        """
        self.add_operator(new_name)
        device = next(self.parameters()).device
        new_op = self.operators[new_name]

        with torch.no_grad():
            all_ids = torch.arange(self.entity_emb.num_embeddings, device=device)
            base_emb = self.entity_emb(all_ids).detach()

            idx_a = torch.randint(0, len(all_ids), (1000,), device=device)
            idx_b = torch.randint(0, len(all_ids), (1000,), device=device)
            alpha = torch.rand(1000, 1, device=device)
            interp = alpha * self.entity_emb(idx_a) + (1 - alpha) * self.entity_emb(idx_b)
            calibration = torch.cat([base_emb, interp.detach()], dim=0)

            target = calibration.clone()
            for rel in chain:
                target = self.operators[rel](target)

        optimizer = torch.optim.Adam(new_op.mlp.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

        for _ in range(500):
            idx = torch.randint(0, len(calibration), (256,), device=device)
            optimizer.zero_grad()
            loss = F.mse_loss(new_op(calibration[idx]), target[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()

        logger.info("compose_init '%s': distillation loss = %.6f", new_name, loss.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "relation_names": self.relation_names,
            "embed_dim": self.embed_dim,
            "num_entities": self.entity_emb.num_embeddings,
            "hidden_dim": self.operators[self.relation_names[0]].mlp[0].out_features,
        }, path)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NROModel":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            num_entities=ckpt["num_entities"],
            relation_names=ckpt["relation_names"],
            embed_dim=ckpt["embed_dim"],
            hidden_dim=ckpt["hidden_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        logger.info("Model loaded ← %s", path)
        return model

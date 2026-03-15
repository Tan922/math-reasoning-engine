"""
mre.knowledge_graph.nro_model
──────────────────────────────
Neural Relation Operators (NRO) model.

Architecture decisions
──────────────────────
1. Entity embeddings are L2-normalised at forward time so all vectors live
   on the unit sphere — cosine similarity is a stable, scale-free distance.

2. Each operator: f_r(x) = residual(x) + mlp_scale * (MLP(x) normalised to
   residual magnitude).  The MLP learns the *direction* of the transformation;
   its magnitude is bounded to at most `mlp_scale` × residual magnitude.
   This prevents the MLP from overwriting the residual skip connection
   (observed symptom: residual_ratio >> 1, cos(input, output) ≈ 0).

3. compose_init() distills f_r2∘f_r1 onto a new operator — the key method
   enabling few-shot learning of composite relations.
"""

from __future__ import annotations

from typing import List

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
        GELU   = Sequential
        LayerNorm = Sequential

        @staticmethod
        def init(): pass

    nn = _FakeNN()  # type: ignore[assignment]

from mre.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Building block
# ─────────────────────────────────────────────────────────────────────────────

class NeuralRelationOperator(nn.Module):
    """
    One learnable operator per relation type.

        out = residual(x)  +  MLP(x) * (‖residual(x)‖ / ‖MLP(x)‖) * mlp_scale

    The residual is a shared linear transform (identity init).
    The MLP contributes a direction-only transformation whose magnitude
    is bounded relative to the residual, preventing explosion.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        shared_residual: "nn.Linear",
        mlp_scale: float = 0.5,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.residual  = shared_residual
        self.mlp_scale = mlp_scale

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual_out = self.residual(x)                                    # (B, d)
        mlp_out      = self.mlp(x)                                         # (B, d)

        # Normalise MLP magnitude to at most mlp_scale × residual magnitude.
        res_norm = residual_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mlp_norm = mlp_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mlp_scaled = mlp_out * (res_norm / mlp_norm) * self.mlp_scale

        return residual_out + mlp_scaled


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class NROModel(nn.Module):
    """
    Full NRO model:
      • Learnable entity embeddings (L2-normalised at forward time)
      • One NeuralRelationOperator per relation
      • Compositional reasoning = operator composition
    """

    def __init__(
        self,
        num_entities:   int,
        relation_names: List[str],
        embed_dim:      int   = 64,
        hidden_dim:     int   = 128,
        mlp_scale:      float = 0.5,
    ):
        super().__init__()
        self.embed_dim      = embed_dim
        self.hidden_dim     = hidden_dim
        self.mlp_scale      = mlp_scale
        self.relation_names = list(relation_names)

        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        nn.init.normal_(self.entity_emb.weight, std=0.1)

        self.shared_residual = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.shared_residual.weight)

        self.operators = nn.ModuleDict({
            name: NeuralRelationOperator(
                embed_dim, hidden_dim, self.shared_residual, mlp_scale
            )
            for name in relation_names
        })

    # ── Inference ─────────────────────────────────────────────────────────────

    def _norm_emb(self, ids: "torch.Tensor") -> "torch.Tensor":
        """Return L2-normalised entity embeddings."""
        return F.normalize(self.entity_emb(ids), dim=-1)

    def reason(
        self,
        head_ids: "torch.Tensor",
        relation_chain: List[str],
    ) -> "torch.Tensor":
        """
        Multi-hop reasoning via operator composition.
        Input and each intermediate result are L2-normalised so that cosine
        similarity remains stable across the chain.
        """
        x = self._norm_emb(head_ids)
        for rel in relation_chain:
            if rel not in self.operators:
                raise KeyError(
                    f"Unknown relation '{rel}'. Known: {list(self.operators.keys())}"
                )
            x = F.normalize(self.operators[rel](x), dim=-1)
        return x

    def score(
        self,
        head_ids: "torch.Tensor",
        relation_chain: List[str],
        tail_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """Cosine similarity between predicted tail and actual tail embedding."""
        pred = self.reason(head_ids, relation_chain)
        tail = self._norm_emb(tail_ids)
        return F.cosine_similarity(pred, tail, dim=-1)

    # ── Dynamic operator management ───────────────────────────────────────────

    def add_operator(self, name: str) -> None:
        """Add a new operator with random init, inheriting mlp_scale."""
        if name in self.operators:
            logger.warning("Operator '%s' already exists — skipping.", name)
            return
        self.operators[name] = NeuralRelationOperator(
            self.embed_dim, self.hidden_dim,
            self.shared_residual, self.mlp_scale
        ).to(next(self.parameters()).device)
        logger.debug("Added operator '%s' (mlp_scale=%.2f).", name, self.mlp_scale)

    def compose_init(self, new_name: str, chain: List[str]) -> None:
        """
        Initialise a new operator as a functional approximation of
        f_chain[-1] ∘ … ∘ f_chain[0]  via MSE distillation over the actual
        entity embedding distribution + convex interpolations.
        """
        self.add_operator(new_name)
        device = next(self.parameters()).device
        new_op = self.operators[new_name]

        with torch.no_grad():
            all_ids     = torch.arange(self.entity_emb.num_embeddings, device=device)
            base_emb    = self._norm_emb(all_ids).detach()

            idx_a  = torch.randint(0, len(all_ids), (1000,), device=device)
            idx_b  = torch.randint(0, len(all_ids), (1000,), device=device)
            alpha  = torch.rand(1000, 1, device=device)
            interp = F.normalize(
                alpha * self.entity_emb(idx_a) + (1 - alpha) * self.entity_emb(idx_b),
                dim=-1,
            )
            calibration = torch.cat([base_emb, interp.detach()], dim=0)

            target = calibration.clone()
            for rel in chain:
                target = F.normalize(self.operators[rel](target), dim=-1)

        optimizer = torch.optim.Adam(new_op.mlp.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

        for _ in range(500):
            idx  = torch.randint(0, len(calibration), (256,), device=device)
            optimizer.zero_grad()
            pred = F.normalize(new_op(calibration[idx]), dim=-1)
            loss = F.mse_loss(pred, target[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()

        logger.info("compose_init '%s': distillation loss = %.6f", new_name, loss.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "state_dict":    self.state_dict(),
            "relation_names": self.relation_names,
            "embed_dim":     self.embed_dim,
            "hidden_dim":    self.hidden_dim,
            "mlp_scale":     self.mlp_scale,
            "num_entities":  self.entity_emb.num_embeddings,
        }, path)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NROModel":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            num_entities   = ckpt["num_entities"],
            relation_names = ckpt["relation_names"],
            embed_dim      = ckpt["embed_dim"],
            hidden_dim     = ckpt["hidden_dim"],
            mlp_scale      = ckpt.get("mlp_scale", 0.5),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        logger.info("Model loaded ← %s", path)
        return model

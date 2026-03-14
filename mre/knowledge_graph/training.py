"""
mre.knowledge_graph.training
─────────────────────────────
Loss functions, training loop, and evaluation metrics for NROModel.
"""

from __future__ import annotations

import random
from typing import List, Tuple

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
    def _no_grad(fn):
        return torch.no_grad()(fn)
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    def _no_grad(fn):  # type: ignore[misc]
        return fn

from mre.knowledge_graph.nro_model import NROModel
from mre.utils import get_logger

logger = get_logger(__name__)

Edge = Tuple[int, int]


def margin_ranking_loss(
    model: NROModel,
    head_ids,
    relation_chain: List[str],
    tail_ids,
    num_negatives: int = 32,
    margin: float = 0.5,
):
    """Self-adversarial margin loss."""
    B = head_ids.size(0)
    n_ents = model.entity_emb.num_embeddings

    pos_scores = model.score(head_ids, relation_chain, tail_ids)

    neg_tail_ids = torch.randint(0, n_ents, (B, num_negatives), device=head_ids.device)
    head_exp     = head_ids.unsqueeze(1).expand(B, num_negatives).reshape(-1)
    neg_scores   = model.score(
        head_exp, relation_chain, neg_tail_ids.reshape(-1)
    ).view(B, num_negatives).mean(dim=1)

    return F.relu(margin - pos_scores + neg_scores).mean()


def hits_at_k(
    model: NROModel,
    test_edges: List[Edge],
    relation_chain: List[str],
    k: int = 10,
    batch_size: int = 64,
) -> float:
    """Filtered Hits@k over all entities."""
    if not test_edges:
        return 0.0

    model.eval()
    device   = next(model.parameters()).device
    n_ents   = model.entity_emb.num_embeddings

    with torch.no_grad():
        all_ent_emb = model.entity_emb(torch.arange(n_ents, device=device))
        hits = 0
        for i in range(0, len(test_edges), batch_size):
            batch  = test_edges[i: i + batch_size]
            heads  = torch.tensor([h for h, _ in batch], device=device)
            tails  = torch.tensor([t for _, t in batch], device=device)
            pred   = model.reason(heads, relation_chain)
            scores = F.cosine_similarity(
                pred.unsqueeze(1), all_ent_emb.unsqueeze(0), dim=-1
            )
            true_s = scores[torch.arange(len(batch), device=device), tails]
            rank   = (scores > true_s.unsqueeze(1)).sum(dim=1) + 1
            hits  += (rank <= k).sum().item()

    return hits / len(test_edges)


def train_epoch(
    model: NROModel,
    edges: List[Edge],
    relation_chain: List[str],
    optimizer,
    batch_size: int = 128,
) -> float:
    """Run one epoch; return mean loss."""
    model.train()
    device = next(model.parameters()).device
    random.shuffle(edges)

    total_loss, n_batches = 0.0, 0
    for i in range(0, len(edges), batch_size):
        batch = edges[i: i + batch_size]
        if len(batch) < 4:
            continue
        heads = torch.tensor([h for h, _ in batch], device=device)
        tails = torch.tensor([t for _, t in batch], device=device)

        optimizer.zero_grad()
        loss = margin_ranking_loss(model, heads, relation_chain, tails)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def train_base_model(
    model: NROModel,
    splits: dict,
    base_relations: List[str],
    epochs: int = 60,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    eval_every: int = 10,
    hits_k: int = 10,
) -> dict:
    """Train NROModel jointly on all base relations. Returns history dict."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: dict = {"loss": [], "val_hits": {rel: [] for rel in base_relations}}

    logger.info("Training base model — %d epochs, %d relations", epochs, len(base_relations))
    for epoch in range(epochs):
        epoch_loss = sum(
            train_epoch(model, splits[rel]["train"], [rel], optimizer, batch_size)
            for rel in base_relations
        )
        scheduler.step()
        avg_loss = epoch_loss / len(base_relations)
        history["loss"].append(avg_loss)

        if (epoch + 1) % eval_every == 0:
            val_hits = {
                rel: hits_at_k(model, splits[rel]["val"], [rel], k=hits_k)
                for rel in base_relations
            }
            for rel, h in val_hits.items():
                history["val_hits"][rel].append(h)
            avg_h = sum(val_hits.values()) / len(val_hits)
            logger.info("Epoch %3d/%d  loss=%.4f  val_H@%d=%.3f",
                        epoch + 1, epochs, avg_loss, hits_k, avg_h)

    return history

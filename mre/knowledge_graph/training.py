"""
mre.knowledge_graph.training
─────────────────────────────
Loss functions, training loop, and evaluation metrics for NROModel.

Key fixes vs original:
  1. train_base_model uses a single mixed-batch forward pass across all
     relations per step — every operator receives gradients every update.
  2. Hard negative sampling (max neg instead of mean) — much stronger signal.
  3. Filtered Hits@k excludes known positive edges from the ranking
     denominator so the metric is not penalised for correctly scoring them.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from mre.knowledge_graph.nro_model import NROModel
from mre.utils import get_logger

logger = get_logger(__name__)

Edge = Tuple[int, int]


# ── Loss ──────────────────────────────────────────────────────────────────────

def margin_ranking_loss(
    model: NROModel,
    head_ids,
    relation_chain: List[str],
    tail_ids,
    num_negatives: int = 64,
    margin: float = 0.3,
) -> "torch.Tensor":
    """
    Self-adversarial margin loss with hard negative mining.

    Uses the *hardest* negative per sample (max scoring corrupted tail)
    instead of the mean.  This produces much stronger gradient signal,
    especially early in training when pos/neg scores are close.

        L = mean( relu(margin − score_pos + score_neg_hard) )
    """
    B      = head_ids.size(0)
    n_ents = model.entity_emb.num_embeddings

    pos_scores = model.score(head_ids, relation_chain, tail_ids)   # (B,)

    # Sample negatives and pick the hardest one per example
    neg_tail_ids = torch.randint(
        0, n_ents, (B, num_negatives), device=head_ids.device
    )
    head_exp  = head_ids.unsqueeze(1).expand(B, num_negatives).reshape(-1)
    neg_scores = model.score(
        head_exp, relation_chain, neg_tail_ids.reshape(-1)
    ).view(B, num_negatives)

    hard_neg_scores, _ = neg_scores.max(dim=1)                    # (B,) hardest

    return F.relu(margin - pos_scores + hard_neg_scores).mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def hits_at_k(
    model: NROModel,
    test_edges: List[Edge],
    relation_chain: List[str],
    k: int = 10,
    batch_size: int = 64,
    filter_edges: List[Edge] | None = None,
) -> float:
    """
    Filtered Hits@k.

    For each (h, t) in test_edges, rank all entities by score.
    If *filter_edges* is provided (typically train + val edges for the same
    relation), those known-positive (h, t') pairs are masked out of the
    ranking so the metric is not penalised for scoring them highly.
    """
    if not test_edges:
        return 0.0

    model.eval()
    device  = next(model.parameters()).device
    n_ents  = model.entity_emb.num_embeddings

    # Build per-head positive set for filtering
    filter_set: Dict[int, Set[int]] = {}
    if filter_edges:
        for h, t in filter_edges:
            filter_set.setdefault(h, set()).add(t)

    with torch.no_grad():
        all_ent_emb = F.normalize(model.entity_emb(torch.arange(n_ents, device=device)), dim=-1)
        hits = 0

        for i in range(0, len(test_edges), batch_size):
            batch = test_edges[i: i + batch_size]
            heads = torch.tensor([h for h, _ in batch], device=device)
            tails = torch.tensor([t for _, t in batch], device=device)

            pred   = model.reason(heads, relation_chain)           # (B, d)
            scores = F.cosine_similarity(
                pred.unsqueeze(1), all_ent_emb.unsqueeze(0), dim=-1
            )                                                      # (B, N)

            # Mask known positives (set their score to -inf so they don't
            # push the true test tail down in the ranking)
            if filter_set:
                for bi, (h, t) in enumerate(batch):
                    known = filter_set.get(int(h), set()) - {int(t)}
                    if known:
                        idx = torch.tensor(list(known), device=device)
                        scores[bi, idx] = -1e9

            true_s = scores[torch.arange(len(batch), device=device), tails]
            rank   = (scores > true_s.unsqueeze(1)).sum(dim=1) + 1
            hits  += (rank <= k).sum().item()

    return hits / len(test_edges)


# ── Per-relation epoch (used for fine-tuning a single relation) ───────────────

def train_epoch(
    model: NROModel,
    edges: List[Edge],
    relation_chain: List[str],
    optimizer,
    batch_size: int = 128,
) -> float:
    """
    Run one epoch over *edges* for a single relation chain.
    Used for fine-tuning a new composite operator (Phase 2 experiments).
    Each batch does its own zero_grad → backward → step.
    """
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


# ── Joint multi-relation training ─────────────────────────────────────────────

def train_base_model(
    model: NROModel,
    splits: dict,
    base_relations: List[str],
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    eval_every: int = 10,
    hits_k: int = 10,
) -> dict:
    """
    Train NROModel jointly on all base relations.

    Each optimiser step processes one batch that contains edges from ALL
    relations interleaved.  A single loss.backward() call therefore sends
    gradients through every operator simultaneously — fixing the original
    bug where operators for relations not in the current batch received
    NO GRAD for that step.

    Returns a history dict with keys 'loss' and 'val_hits'.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    history: dict = {"loss": [], "val_hits": {rel: [] for rel in base_relations}}
    device = next(model.parameters()).device

    logger.info(
        "Training base model — %d epochs, %d relations, joint batches",
        epochs, len(base_relations),
    )

    for epoch in range(epochs):
        model.train()

        # Build one interleaved pool of (head, tail, relation) for this epoch
        pool: List[Tuple[int, int, str]] = []
        for rel in base_relations:
            pool.extend((h, t, rel) for h, t in splits[rel]["train"])
        random.shuffle(pool)

        total_loss, n_batches = 0.0, 0

        for i in range(0, len(pool), batch_size):
            raw_batch = pool[i: i + batch_size]
            if len(raw_batch) < 4:
                continue

            # Group by relation so we can call model.score per relation
            rel_groups: Dict[str, Tuple[List[int], List[int]]] = {}
            for h, t, rel in raw_batch:
                if rel not in rel_groups:
                    rel_groups[rel] = ([], [])
                rel_groups[rel][0].append(h)
                rel_groups[rel][1].append(t)

            optimizer.zero_grad()

            # Sum loss across all relations in this batch —
            # one backward() → gradients flow through every operator present
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for rel, (hs, ts) in rel_groups.items():
                h_t = torch.tensor(hs, device=device)
                t_t = torch.tensor(ts, device=device)
                batch_loss = batch_loss + margin_ranking_loss(
                    model, h_t, [rel], t_t
                )

            batch_loss = batch_loss / len(rel_groups)   # normalise by n_relations
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)

        if (epoch + 1) % eval_every == 0:
            # Build filter sets (train + val positives) for each relation
            filter_edges = {
                rel: splits[rel]["train"] + splits[rel]["val"]
                for rel in base_relations
            }
            val_hits = {
                rel: hits_at_k(
                    model, splits[rel]["val"], [rel],
                    k=hits_k,
                    filter_edges=filter_edges[rel],
                )
                for rel in base_relations
            }
            for rel, h in val_hits.items():
                history["val_hits"][rel].append(h)
            avg_h = sum(val_hits.values()) / len(val_hits)
            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_H@%d=%.3f",
                epoch + 1, epochs, avg_loss, hits_k, avg_h,
            )

    return history

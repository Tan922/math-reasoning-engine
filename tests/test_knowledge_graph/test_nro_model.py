"""
Unit tests for NROModel and NeuralRelationOperator.
"""

import copy
import pytest
import torch

from mre.knowledge_graph.nro_model import NROModel


RELATIONS = ["depends_on", "generalizes", "equivalent_to", "applied_in"]
N_ENTITIES = 50
EMBED_DIM = 16
HIDDEN_DIM = 32


@pytest.fixture
def model():
    m = NROModel(
        num_entities=N_ENTITIES,
        relation_names=RELATIONS,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    return m


def test_reason_single_hop(model):
    heads = torch.tensor([0, 1, 2])
    out = model.reason(heads, ["depends_on"])
    assert out.shape == (3, EMBED_DIM)


def test_reason_multi_hop(model):
    heads = torch.tensor([0, 1])
    out = model.reason(heads, ["generalizes", "depends_on"])
    assert out.shape == (2, EMBED_DIM)


def test_score_range(model):
    heads = torch.tensor([0, 1, 2])
    tails = torch.tensor([3, 4, 5])
    scores = model.score(heads, ["depends_on"], tails)
    assert scores.shape == (3,)
    # Cosine similarity must be in [-1, 1]
    assert (scores >= -1.0 - 1e-5).all()
    assert (scores <= 1.0 + 1e-5).all()


def test_add_operator(model):
    model.add_operator("new_rel")
    assert "new_rel" in model.operators
    heads = torch.tensor([0, 1])
    out = model.reason(heads, ["new_rel"])
    assert out.shape == (2, EMBED_DIM)


def test_add_operator_idempotent(model):
    model.add_operator("dup_rel")
    model.add_operator("dup_rel")   # should not raise
    assert list(model.operators.keys()).count("dup_rel") == 1


def test_compose_init(model):
    chain = ["generalizes", "depends_on"]
    model.compose_init("gen_dep", chain)
    assert "gen_dep" in model.operators
    heads = torch.tensor([0, 1, 2])
    out = model.reason(heads, ["gen_dep"])
    assert out.shape == (3, EMBED_DIM)


def test_compose_init_better_than_random():
    """
    After distillation, gen_dep output should be closer to the
    composed chain output than a randomly-initialised operator.
    """
    torch.manual_seed(0)
    model = NROModel(N_ENTITIES, RELATIONS, EMBED_DIM, HIDDEN_DIM)
    chain = ["generalizes", "depends_on"]

    # Ground truth: composed output
    heads = torch.tensor(list(range(10)))
    with torch.no_grad():
        gt = model.reason(heads, chain)

    # Random init baseline
    random_model = copy.deepcopy(model)
    random_model.add_operator("gen_dep")
    with torch.no_grad():
        random_out = random_model.reason(heads, ["gen_dep"])

    # Compose-init model
    composed_model = copy.deepcopy(model)
    composed_model.compose_init("gen_dep", chain)
    with torch.no_grad():
        composed_out = composed_model.reason(heads, ["gen_dep"])

    mse_random   = ((random_out   - gt) ** 2).mean().item()
    mse_composed = ((composed_out - gt) ** 2).mean().item()
    assert mse_composed < mse_random, (
        f"Compose-init MSE ({mse_composed:.4f}) should be less than "
        f"random-init MSE ({mse_random:.4f})"
    )


def test_unknown_relation_raises(model):
    heads = torch.tensor([0])
    with pytest.raises(KeyError):
        model.reason(heads, ["nonexistent_relation"])


def test_save_load(tmp_path, model):
    path = str(tmp_path / "nro.pt")
    model.save(path)
    loaded = NROModel.load(path, device="cpu")
    heads = torch.tensor([0, 1])
    orig  = model.reason(heads, ["depends_on"])
    reloaded = loaded.reason(heads, ["depends_on"])
    assert torch.allclose(orig, reloaded, atol=1e-5)

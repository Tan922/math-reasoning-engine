"""
Unit tests for SyntheticKG.
"""

import pytest

from mre.knowledge_graph.synthetic_kg import SyntheticKG


@pytest.fixture(scope="module")
def kg():
    return SyntheticKG(num_entities=100, embed_dim=16, seed=42)


def test_edge_counts(kg):
    for rel in kg.relations:
        assert len(kg.edges[rel]) > 0, f"No edges for relation '{rel}'"


def test_composite_edges(kg):
    for rel in kg.composite_relations:
        assert len(kg.edges[rel]) > 0, f"No composite edges for '{rel}'"


def test_no_self_loops(kg):
    for rel, edges in kg.edges.items():
        for h, t in edges:
            assert h != t, f"Self-loop found in '{rel}': ({h}, {h})"


def test_valid_entity_ids(kg):
    for rel, edges in kg.edges.items():
        for h, t in edges:
            assert 0 <= h < kg.num_entities
            assert 0 <= t < kg.num_entities


def test_get_split_sizes(kg):
    train, val, test = kg.get_split("depends_on")
    total = len(train) + len(val) + len(test)
    assert total == len(kg.edges["depends_on"])
    assert len(train) > len(val)
    assert len(train) > len(test)


def test_get_split_no_overlap(kg):
    train, val, test = kg.get_split("generalizes")
    train_set = set(train)
    val_set   = set(val)
    test_set  = set(test)
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


def test_summary_string(kg):
    s = kg.summary()
    assert "SyntheticKG" in s
    for rel in kg.relations:
        assert rel in s

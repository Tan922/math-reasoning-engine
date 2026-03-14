"""
Unit tests for MathKGBuilder and relation extractor.
"""

import pytest

from mre.knowledge_graph.mathkg_builder import (
    MathKGBuilder,
    extract_relations_from_wikitext,
    _generate_synthetic_entities,
    _generate_synthetic_triples,
    TARGET_RELATIONS,
)

# ─── Relation extractor ───────────────────────────────────────────────────────

SAMPLE_WIKITEXT = """
== Theorem ==
Let $f$ be a function. This is a special case of [[Generalised Theorem]].

== Proof ==
By [[Cauchy's Theorem]] and [[Definition:Continuous Function]], we have...
From [[Intermediate Value Theorem]] it follows that...

== Corollaries ==
* [[MyTheorem/Corollary 1]]
* [[MyTheorem/Corollary 2]]

== Also see ==
* [[Related Theorem A]]

== Generalizations ==
* [[Big General Theorem]]

== Equivalent statements ==
* [[Other Form of Theorem]]
"""


def test_extractor_finds_corollaries():
    triples = extract_relations_from_wikitext("MyTheorem", SAMPLE_WIKITEXT)
    rels = {t.relation for t in triples}
    assert "corollary_of" in rels
    assert "generalizes" in rels


def test_extractor_finds_proof_deps():
    triples = extract_relations_from_wikitext("MyTheorem", SAMPLE_WIKITEXT)
    dep_tails = {t.tail for t in triples if t.relation == "depends_on"}
    assert "Cauchy's Theorem" in dep_tails
    assert "Intermediate Value Theorem" in dep_tails


def test_extractor_finds_definition_applied():
    triples = extract_relations_from_wikitext("MyTheorem", SAMPLE_WIKITEXT)
    applied = [(t.head, t.tail) for t in triples if t.relation == "applied_in"]
    heads = [h for h, _ in applied]
    assert "Definition:Continuous Function" in heads


def test_extractor_finds_equivalent():
    triples = extract_relations_from_wikitext("MyTheorem", SAMPLE_WIKITEXT)
    equiv_tails = {t.tail for t in triples if t.relation == "equivalent_to"}
    assert "Other Form of Theorem" in equiv_tails


def test_extractor_no_self_loops():
    triples = extract_relations_from_wikitext("MyTheorem", SAMPLE_WIKITEXT)
    for t in triples:
        assert t.head != t.tail


def test_extractor_deduplicates():
    # Same wikitext twice should not double the triples
    triples1 = extract_relations_from_wikitext("X", SAMPLE_WIKITEXT)
    triples2 = extract_relations_from_wikitext("X", SAMPLE_WIKITEXT)
    assert len(triples1) == len(triples2)


def test_extractor_empty_wikitext():
    triples = extract_relations_from_wikitext("X", "")
    assert triples == []


def test_extractor_confidence_range():
    triples = extract_relations_from_wikitext("MyTheorem", SAMPLE_WIKITEXT)
    for t in triples:
        assert 0.0 <= t.confidence <= 1.0


# ─── Synthetic data generators ────────────────────────────────────────────────

def test_synthetic_entities_count():
    entities, wmap = _generate_synthetic_entities(50, seed=42)
    assert len(entities) == 50
    assert len(wmap) == 50


def test_synthetic_entity_ids_unique():
    entities, _ = _generate_synthetic_entities(30, seed=0)
    ids = [e.entity_id for e in entities]
    assert len(ids) == len(set(ids))


def test_synthetic_triples_no_self_loops():
    entities, _ = _generate_synthetic_entities(50, seed=42)
    triples = _generate_synthetic_triples(entities, seed=42)
    for t in triples:
        assert t.head != t.tail


def test_synthetic_triples_valid_relations():
    entities, _ = _generate_synthetic_entities(50, seed=42)
    triples = _generate_synthetic_triples(entities, seed=42)
    for t in triples:
        assert t.relation in TARGET_RELATIONS


# ─── MathKGBuilder (offline / synthetic mode) ────────────────────────────────

@pytest.fixture
def builder(tmp_path):
    b = MathKGBuilder(data_dir=str(tmp_path / "mathkg"), max_entities=60, seed=42)
    b.crawl_entities()      # will fall back to synthetic
    b.extract_relations()
    b.filter_and_clean()
    b.make_splits()
    return b


def test_builder_has_entities(builder):
    assert len(builder.clean_entities) > 0


def test_builder_has_triples(builder):
    assert len(builder.clean_triples) > 0


def test_builder_no_self_loops(builder):
    id_map = {e.name: e.entity_id for e in builder.clean_entities}
    for t in builder.clean_triples:
        assert id_map[t.head] != id_map[t.tail]


def test_builder_splits_cover_all(builder):
    for rel in TARGET_RELATIONS:
        sp = builder.splits_data.get(rel, {})
        total_split = (len(sp.get("train",[])) + len(sp.get("val",[])) + len(sp.get("test",[])))
        total_rel   = sum(1 for t in builder.clean_triples if t.relation == rel)
        assert total_split == total_rel


def test_builder_save_creates_files(builder):
    builder.save()
    data_dir = builder.data_dir
    assert (data_dir / "entities.tsv").exists()
    assert (data_dir / "relations.tsv").exists()
    assert (data_dir / "stats.json").exists()
    for rel in TARGET_RELATIONS:
        for split in ("train", "val", "test"):
            assert (data_dir / "splits" / f"{rel}_{split}.tsv").exists()


def test_builder_saved_entities_loadable(builder, tmp_path):
    builder.save()
    import pandas as pd
    df = pd.read_csv(builder.data_dir / "entities.tsv", sep="\t")
    assert len(df) == len(builder.clean_entities)
    assert "entity_id" in df.columns
    assert "name" in df.columns
    assert "type" in df.columns


def test_builder_summary_string(builder):
    s = builder.summary()
    assert "MathKGBuilder" in s
    for rel in TARGET_RELATIONS:
        assert rel in s

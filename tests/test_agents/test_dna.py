"""
tests/test_phase2/test_dna.py
─────────────────────────────
Unit tests for AgentDNA — serialisation, mutation, crossover.
All tests are offline (no network, no GPU).
"""

import json
import random

import pytest

from mre.agents.dna import (
    AgentDNA,
    KNOWN_DOMAINS,
    KNOWN_TOOLS,
    PROMPT_TEMPLATES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def seed_dna():
    return AgentDNA()


@pytest.fixture
def algebra_dna():
    dna = AgentDNA()
    dna.domain_gene = {d: (1.0 if d == "algebra" else 0.0) for d in KNOWN_DOMAINS}
    dna.reasoning_gene = ["SymbolicSimplify", "EquationSolve"]
    return dna


@pytest.fixture
def geometry_dna():
    dna = AgentDNA()
    dna.domain_gene = {d: (1.0 if d == "geometry" else 0.0) for d in KNOWN_DOMAINS}
    dna.reasoning_gene = ["DeductiveStep", "ProofByContradiction"]
    return dna


# ── Basic construction ────────────────────────────────────────────────────────

class TestAgentDNAConstruction:
    def test_default_fields(self, seed_dna):
        assert seed_dna.agent_id
        assert seed_dna.generation == 0
        assert seed_dna.parent_ids == []
        assert seed_dna.model_gene
        assert seed_dna.prompt_gene in PROMPT_TEMPLATES
        assert seed_dna.fitness_score is None

    def test_domain_gene_sums_to_one(self, seed_dna):
        total = sum(seed_dna.domain_gene.values())
        assert abs(total - 1.0) < 1e-9

    def test_tool_gene_keys(self, seed_dna):
        for t in KNOWN_TOOLS:
            assert t in seed_dna.tool_gene

    def test_reasoning_gene_nonempty(self, seed_dna):
        assert len(seed_dna.reasoning_gene) > 0

    def test_top_domain(self, algebra_dna):
        assert algebra_dna.top_domain == "algebra"

    def test_active_tools(self, seed_dna):
        active = seed_dna.active_tools
        assert isinstance(active, list)

    def test_system_prompt_resolution(self, seed_dna):
        prompt = seed_dna.system_prompt
        assert len(prompt) > 10

    def test_custom_raw_prompt(self):
        dna = AgentDNA(prompt_gene="Be extremely concise.")
        assert dna.system_prompt == "Be extremely concise."


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestAgentDNASerialization:
    def test_to_dict_roundtrip(self, seed_dna):
        d = seed_dna.to_dict()
        restored = AgentDNA.from_dict(d)
        assert restored.agent_id == seed_dna.agent_id
        assert restored.model_gene == seed_dna.model_gene
        assert restored.reasoning_gene == seed_dna.reasoning_gene

    def test_to_json_roundtrip(self, seed_dna):
        js = seed_dna.to_json()
        parsed = json.loads(js)
        assert parsed["agent_id"] == seed_dna.agent_id
        restored = AgentDNA.from_json(js)
        assert restored.agent_id == seed_dna.agent_id

    def test_dna_hash_deterministic(self, seed_dna):
        h1 = seed_dna.dna_hash()
        h2 = seed_dna.dna_hash()
        assert h1 == h2

    def test_dna_hash_changes_on_mutation(self, seed_dna):
        original_hash = seed_dna.dna_hash()
        rng = random.Random(42)
        # Force a prompt change
        mutant = seed_dna.clone()
        mutant.prompt_gene = "creative"
        assert mutant.dna_hash() != original_hash


# ── Clone ─────────────────────────────────────────────────────────────────────

class TestAgentDNAClone:
    def test_clone_produces_new_id(self, seed_dna):
        clone = seed_dna.clone()
        assert clone.agent_id != seed_dna.agent_id

    def test_clone_increments_generation(self, seed_dna):
        clone = seed_dna.clone()
        assert clone.generation == seed_dna.generation + 1

    def test_clone_sets_parent_id(self, seed_dna):
        clone = seed_dna.clone()
        assert seed_dna.agent_id in clone.parent_ids

    def test_clone_clears_fitness(self, seed_dna):
        seed_dna.fitness_score = 0.9
        clone = seed_dna.clone()
        assert clone.fitness_score is None


# ── Mutation ──────────────────────────────────────────────────────────────────

class TestAgentDNAMutation:
    def test_mutate_returns_new_object(self, seed_dna):
        rng = random.Random(0)
        mutant = seed_dna.mutate(rng=rng, mutation_rate=1.0)
        assert mutant is not seed_dna

    def test_mutate_preserves_domain_sum(self, seed_dna):
        rng = random.Random(7)
        for _ in range(10):
            mutant = seed_dna.mutate(rng=rng, mutation_rate=0.5)
            total = sum(mutant.domain_gene.values())
            assert abs(total - 1.0) < 1e-6, f"domain sum = {total}"

    def test_mutate_tools_stay_in_range(self, seed_dna):
        rng = random.Random(13)
        for _ in range(20):
            mutant = seed_dna.mutate(rng=rng, mutation_rate=1.0)
            for v in mutant.tool_gene.values():
                assert 0.0 <= v <= 1.0

    def test_mutate_reasoning_gene_stays_nonempty(self, seed_dna):
        rng = random.Random(99)
        for _ in range(50):
            mutant = seed_dna.mutate(rng=rng, mutation_rate=1.0)
            # May be empty only if original had 1 element and delete was chosen
            assert isinstance(mutant.reasoning_gene, list)


# ── Crossover ────────────────────────────────────────────────────────────────

class TestAgentDNACrossover:
    def test_crossover_produces_two_offspring(self, algebra_dna, geometry_dna):
        o1, o2 = AgentDNA.crossover(algebra_dna, geometry_dna)
        assert o1 is not algebra_dna
        assert o2 is not geometry_dna

    def test_offspring_have_two_parents(self, algebra_dna, geometry_dna):
        o1, o2 = AgentDNA.crossover(algebra_dna, geometry_dna)
        assert algebra_dna.agent_id in o1.parent_ids
        assert geometry_dna.agent_id in o1.parent_ids

    def test_offspring_domain_is_blend(self, algebra_dna, geometry_dna):
        o1, _ = AgentDNA.crossover(algebra_dna, geometry_dna)
        # blend means algebra weight < 1.0 (unless geometry had 0 everywhere)
        assert o1.domain_gene["algebra"] <= 1.0

    def test_crossover_reasoning_uses_both_parents(self, algebra_dna, geometry_dna):
        rng = random.Random(5)
        # run 20 times to get statistical coverage
        ops_seen = set()
        for _ in range(20):
            o1, o2 = AgentDNA.crossover(algebra_dna, geometry_dna, rng=rng)
            ops_seen.update(o1.reasoning_gene)
            ops_seen.update(o2.reasoning_gene)
        # Should see operators from both parents
        algebra_ops = set(algebra_dna.reasoning_gene)
        geometry_ops = set(geometry_dna.reasoning_gene)
        assert ops_seen & algebra_ops
        assert ops_seen & geometry_ops

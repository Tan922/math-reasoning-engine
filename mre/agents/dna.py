"""
mre.agents.dna
──────────────
AgentDNA — the heritable blueprint of a reasoning agent.

Five gene fields control every aspect of agent behaviour:

  model_gene    – which LLM backend to use
  prompt_gene   – thinking-style instruction template
  domain_gene   – weighted domain expertise vector
  tool_gene     – tool access preferences
  reasoning_gene– ordered operator sequence (the "operator graph")

The class is intentionally a plain dataclass so it can be serialised,
deep-copied, crossed-over, and mutated by the evolution engine (Phase 3).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Domain catalogue ─────────────────────────────────────────────────────────

KNOWN_DOMAINS: List[str] = [
    "algebra",
    "number_theory",
    "geometry",
    "calculus",
    "combinatorics",
    "probability",
    "linear_algebra",
    "topology",
    "logic",
    "set_theory",
]

# ── Tool catalogue ────────────────────────────────────────────────────────────

KNOWN_TOOLS: List[str] = [
    "sympy",
    "python_exec",
    "theorem_db",
    "numerical_solver",
    "latex_renderer",
]

# ── Operator catalogue (populated by the operator library) ───────────────────

DEFAULT_OPERATOR_SEQUENCE: List[str] = [
    "SymbolicSimplify",
    "DeductiveStep",
    "EquationSolve",
    "SelfCritique",
    "RepairChain",
]

# ── PromptGene templates ──────────────────────────────────────────────────────

PROMPT_TEMPLATES: Dict[str, str] = {
    "rigorous": (
        "You are a mathematically rigorous assistant. "
        "Every step must be formally justified. "
        "Prefer symbolic proofs over heuristic arguments."
    ),
    "creative": (
        "You are a creative mathematical explorer. "
        "Try unconventional approaches, consider analogies, "
        "and do not hesitate to conjecture before proving."
    ),
    "concise": (
        "You are a concise problem-solver. "
        "Minimise words; maximise precision. "
        "Omit routine steps and jump to key insights."
    ),
    "pedagogical": (
        "You are a patient mathematics tutor. "
        "Explain each step clearly, define every term, "
        "and check for common mistakes."
    ),
    "adversarial": (
        "You are a devil's advocate. "
        "After each reasoning step, actively look for counter-examples "
        "or logical gaps before proceeding."
    ),
}


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class AgentDNA:
    """
    Complete genetic blueprint for one reasoning agent.

    Parameters
    ----------
    agent_id : str
        Unique identifier (auto-generated if not provided).
    model_gene : str
        LLM backend identifier, e.g. ``"claude-sonnet-4-20250514"`` or
        ``"gpt-4o"``.  The engine resolves this to an actual API client.
    prompt_gene : str
        Key into ``PROMPT_TEMPLATES`` **or** a raw system-prompt string.
    domain_gene : Dict[str, float]
        Mapping ``domain_name -> weight`` (values should sum to ~1.0).
        Agents with higher weight in a domain are preferentially selected
        for problems in that domain.
    tool_gene : Dict[str, float]
        Mapping ``tool_name -> preference_score`` ∈ [0, 1].
        Score 0 = disabled, 1 = strongly preferred.
    reasoning_gene : List[str]
        Ordered list of operator class names that form the default
        reasoning pipeline.  The agent executes them in sequence unless
        an operator signals early termination.
    generation : int
        Evolutionary generation counter.
    parent_ids : List[str]
        IDs of parent agents (empty for seed agents).
    fitness_score : Optional[float]
        Latest fitness score assigned by the evaluation commission.
    metadata : Dict[str, Any]
        Arbitrary extra data (e.g., Elo rating, win/loss record).
    """

    # Core identity
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    # ── Five genes ────────────────────────────────────────────────────────────

    model_gene: str = "claude-sonnet-4-20250514"

    prompt_gene: str = "rigorous"   # key in PROMPT_TEMPLATES or raw string

    domain_gene: Dict[str, float] = field(
        default_factory=lambda: {d: 1.0 / len(KNOWN_DOMAINS) for d in KNOWN_DOMAINS}
    )

    tool_gene: Dict[str, float] = field(
        default_factory=lambda: {t: 0.5 for t in KNOWN_TOOLS}
    )

    reasoning_gene: List[str] = field(
        default_factory=lambda: list(DEFAULT_OPERATOR_SEQUENCE)
    )

    # ── Fitness / meta ────────────────────────────────────────────────────────

    fitness_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        """Resolve prompt_gene to the actual system-prompt string."""
        return PROMPT_TEMPLATES.get(self.prompt_gene, self.prompt_gene)

    @property
    def top_domain(self) -> str:
        """Return the domain with the highest weight."""
        return max(self.domain_gene, key=lambda k: self.domain_gene[k])

    @property
    def active_tools(self) -> List[str]:
        """Return tools with preference score > 0.3."""
        return [t for t, s in self.tool_gene.items() if s > 0.3]

    def dna_hash(self) -> str:
        """Deterministic 8-hex fingerprint of this agent's genes."""
        payload = json.dumps(
            {
                "model": self.model_gene,
                "prompt": self.prompt_gene,
                "domain": sorted(self.domain_gene.items()),
                "tools": sorted(self.tool_gene.items()),
                "reasoning": self.reasoning_gene,
            },
            sort_keys=True,
        ).encode()
        return hashlib.md5(payload).hexdigest()[:8]

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "model_gene": self.model_gene,
            "prompt_gene": self.prompt_gene,
            "domain_gene": self.domain_gene,
            "tool_gene": self.tool_gene,
            "reasoning_gene": self.reasoning_gene,
            "fitness_score": self.fitness_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentDNA":
        return cls(
            agent_id=d.get("agent_id", str(uuid.uuid4())[:8]),
            generation=d.get("generation", 0),
            parent_ids=d.get("parent_ids", []),
            model_gene=d.get("model_gene", "claude-sonnet-4-20250514"),
            prompt_gene=d.get("prompt_gene", "rigorous"),
            domain_gene=d.get("domain_gene", {}),
            tool_gene=d.get("tool_gene", {}),
            reasoning_gene=d.get("reasoning_gene", []),
            fitness_score=d.get("fitness_score"),
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "AgentDNA":
        return cls.from_dict(json.loads(s))

    # ── Evolution primitives (used by Phase-3 engine) ─────────────────────────

    def clone(self) -> "AgentDNA":
        new = copy.deepcopy(self)
        new.agent_id = str(uuid.uuid4())[:8]
        new.parent_ids = [self.agent_id]
        new.generation = self.generation + 1
        new.fitness_score = None
        return new

    def mutate(
        self,
        rng: Optional[random.Random] = None,
        mutation_rate: float = 0.2,
    ) -> "AgentDNA":
        """
        Return a *new* AgentDNA with small random perturbations.

        Operations chosen at random:
          - swap prompt style
          - shift domain weights
          - flip a tool preference
          - insert / delete / replace one operator in reasoning_gene
        """
        rng = rng or random.Random()
        child = self.clone()

        # Prompt mutation
        if rng.random() < mutation_rate:
            child.prompt_gene = rng.choice(list(PROMPT_TEMPLATES.keys()))

        # Domain weight mutation (Gaussian jitter, then renormalise)
        if rng.random() < mutation_rate:
            domains = list(child.domain_gene.keys())
            d = rng.choice(domains)
            child.domain_gene[d] = max(0.0, child.domain_gene[d] + rng.gauss(0, 0.1))
            total = sum(child.domain_gene.values()) or 1.0
            child.domain_gene = {k: v / total for k, v in child.domain_gene.items()}

        # Tool preference mutation
        if rng.random() < mutation_rate:
            tools = list(child.tool_gene.keys())
            t = rng.choice(tools)
            child.tool_gene[t] = round(min(1.0, max(0.0, child.tool_gene[t] + rng.gauss(0, 0.2))), 3)

        # Reasoning gene mutation
        if rng.random() < mutation_rate and child.reasoning_gene:
            op = rng.randint(0, 2)
            idx = rng.randrange(len(child.reasoning_gene))
            all_ops = DEFAULT_OPERATOR_SEQUENCE  # base catalogue
            if op == 0 and len(child.reasoning_gene) > 1:  # delete
                child.reasoning_gene.pop(idx)
            elif op == 1:  # replace
                child.reasoning_gene[idx] = rng.choice(all_ops)
            else:  # insert
                child.reasoning_gene.insert(idx, rng.choice(all_ops))

        return child

    @staticmethod
    def crossover(
        parent_a: "AgentDNA",
        parent_b: "AgentDNA",
        rng: Optional[random.Random] = None,
    ) -> Tuple["AgentDNA", "AgentDNA"]:
        """
        Single-point crossover on the reasoning_gene;
        gene-swap for scalar genes.

        Returns two offspring DNA objects.
        """
        rng = rng or random.Random()

        def make_child(p1: "AgentDNA", p2: "AgentDNA") -> "AgentDNA":
            child = p1.clone()
            child.parent_ids = [p1.agent_id, p2.agent_id]

            # Blend domain weights (arithmetic mean)
            child.domain_gene = {
                d: (p1.domain_gene.get(d, 0) + p2.domain_gene.get(d, 0)) / 2
                for d in set(p1.domain_gene) | set(p2.domain_gene)
            }

            # Tool preference — pick from either parent
            child.tool_gene = {
                t: rng.choice([p1.tool_gene.get(t, 0.5), p2.tool_gene.get(t, 0.5)])
                for t in set(p1.tool_gene) | set(p2.tool_gene)
            }

            # Reasoning gene — single-point crossover
            seq_a = p1.reasoning_gene
            seq_b = p2.reasoning_gene
            if seq_a and seq_b:
                pt_a = rng.randint(1, len(seq_a))
                pt_b = rng.randint(1, len(seq_b))
                child.reasoning_gene = seq_a[:pt_a] + seq_b[pt_b:]

            # Prompt — inherit from the fitter parent
            if (p1.fitness_score or 0) >= (p2.fitness_score or 0):
                child.prompt_gene = p1.prompt_gene
            else:
                child.prompt_gene = p2.prompt_gene

            return child

        offspring_1 = make_child(parent_a, parent_b)
        offspring_2 = make_child(parent_b, parent_a)
        return offspring_1, offspring_2

    # ── Pretty display ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AgentDNA(id={self.agent_id!r}, gen={self.generation}, "
            f"model={self.model_gene!r}, top_domain={self.top_domain!r}, "
            f"ops={self.reasoning_gene}, fitness={self.fitness_score})"
        )

    def pretty(self) -> str:
        lines = [
            "╔══ AgentDNA ══════════════════════════════════════╗",
            f"  ID          : {self.agent_id}  (hash {self.dna_hash()})",
            f"  Generation  : {self.generation}",
            f"  Parents     : {self.parent_ids or 'seed'}",
            f"  Fitness     : {self.fitness_score}",
            "  ── Genes ──────────────────────────────────────",
            f"  model_gene  : {self.model_gene}",
            f"  prompt_gene : {self.prompt_gene!r}",
            f"  domain_gene : {self.top_domain} (top)",
            f"  tool_gene   : {self.active_tools}",
            f"  reason_gene : {' → '.join(self.reasoning_gene)}",
            "╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

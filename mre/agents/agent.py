"""
mre.agents.agent
────────────────
ReasoningAgent — ties together AgentDNA, OperatorPipeline, and
an optional LLM client into a runnable agent.

Usage
-----
    from mre.agents.agent import ReasoningAgent
    from mre.agents.dna import AgentDNA
    from mre.agents.state import ReasoningState

    agent = ReasoningAgent(dna=AgentDNA())
    result = agent.solve("Solve x**2 - 5*x + 6 = 0",
                         context={"equation": "x**2 - 5*x + 6 = 0"})
    print(result.report())
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from mre.agents.dna import AgentDNA
from mre.agents.state import ReasoningState
from mre.operators.pipeline import OperatorPipeline, PipelineResult
from mre.utils import get_logger

logger = get_logger(__name__)


class ReasoningAgent:
    """
    A single reasoning agent instantiated from an AgentDNA blueprint.

    Parameters
    ----------
    dna : AgentDNA
        The agent's genetic blueprint.
    llm_client : callable, optional
        ``(prompt: str) -> str`` function for LLM-powered operators.
    pipeline_kwargs : dict, optional
        Extra kwargs forwarded to OperatorPipeline.from_dna().
    """

    def __init__(
        self,
        dna: Optional[AgentDNA] = None,
        llm_client: Optional[Callable] = None,
        **pipeline_kwargs,
    ):
        self.dna = dna or AgentDNA()
        self.llm_client = llm_client
        self._pipeline_kwargs = pipeline_kwargs

    # ── Pipeline (lazy construction) ──────────────────────────────────────────

    @property
    def pipeline(self) -> OperatorPipeline:
        if not hasattr(self, "_pipeline"):
            self._pipeline = OperatorPipeline.from_dna(
                self.dna,
                llm_client=self.llm_client,
                **self._pipeline_kwargs,
            )
        return self._pipeline

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Run the agent's operator pipeline on a problem.

        Parameters
        ----------
        problem : str
            The mathematical problem statement.
        context : dict, optional
            Pre-populated context (e.g., ``{"equation": "x+1=3"}``).

        Returns
        -------
        PipelineResult
        """
        state = ReasoningState.from_problem(problem)
        if context:
            state = state.evolve(context={**state.context, **context})

        logger.info(
            "Agent %s solving: %s",
            self.dna.agent_id,
            problem[:60],
        )
        result = self.pipeline.run(state)
        logger.info(
            "Agent %s finished: solved=%s, answer=%s",
            self.dna.agent_id,
            result.solved,
            result.answer,
        )
        return result

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ReasoningAgent(id={self.dna.agent_id!r}, "
            f"pipeline={self.pipeline})"
        )


# ── Pool ─────────────────────────────────────────────────────────────────────

class ReasoningAgentPool:
    """
    Manages a population of ReasoningAgents.

    Provides:
      - ``solve_parallel()``  — run all agents on the same problem
      - ``best_result()``     — return the highest-confidence solution
      - ``update_fitness()``  — write back fitness scores to each DNA
    """

    def __init__(self, agents: Optional[list] = None):
        self.agents: list[ReasoningAgent] = agents or []

    @classmethod
    def from_dna_list(
        cls,
        dna_list: list,
        llm_client: Optional[Callable] = None,
    ) -> "ReasoningAgentPool":
        agents = [ReasoningAgent(dna=d, llm_client=llm_client) for d in dna_list]
        return cls(agents=agents)

    def solve_all(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> list[PipelineResult]:
        """Run every agent on the same problem (sequential)."""
        results = []
        for agent in self.agents:
            try:
                results.append(agent.solve(problem, context=context))
            except Exception as exc:
                logger.warning("Agent %s raised: %s", agent.dna.agent_id, exc)
        return results

    def best_result(
        self,
        results: list[PipelineResult],
    ) -> Optional[PipelineResult]:
        """Return the solved result with the highest confidence."""
        solved = [r for r in results if r.solved]
        if not solved:
            return None
        return max(solved, key=lambda r: r.final_state.confidence)

    def update_fitness(self, results: list[PipelineResult]) -> None:
        """Write back fitness scores (confidence) to each DNA object."""
        for agent, result in zip(self.agents, results):
            agent.dna.fitness_score = round(result.final_state.confidence, 4)

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        return f"ReasoningAgentPool(n={len(self.agents)})"

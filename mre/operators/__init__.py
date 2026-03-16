"""mre.operators — Reasoning operator library."""

# Import library module first to trigger the metaclass auto-registration
# of all six operators into OPERATOR_REGISTRY.
import mre.operators.library  # noqa: F401

from mre.operators.base import (
    BaseOperator,
    OPERATOR_REGISTRY,
    get_operator,
    list_operators,
)
from mre.operators.library import (
    SymbolicSimplify,
    DeductiveStep,
    ProofByContradiction,
    EquationSolve,
    SelfCritique,
    RepairChain,
)
from mre.operators.pipeline import OperatorPipeline, PipelineResult

__all__ = [
    # Base utilities
    "BaseOperator",
    "OPERATOR_REGISTRY",
    "get_operator",
    "list_operators",
    # Six core operators
    "SymbolicSimplify",
    "DeductiveStep",
    "ProofByContradiction",
    "EquationSolve",
    "SelfCritique",
    "RepairChain",
    # Pipeline
    "OperatorPipeline",
    "PipelineResult",
]
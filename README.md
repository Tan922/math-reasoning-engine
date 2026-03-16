# MRE — Evolvable Mathematical Reasoning Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/)
[![Phases](https://img.shields.io/badge/phases-1--4%20complete-brightgreen)](https://github.com/your-org/math-reasoning-engine)

A research system for **self-evolving mathematical reasoning** that combines:
- **Neural Relation Operators (NRO)** for few-shot knowledge-graph learning
- **Agent DNA** — a heritable, evolvable 5-gene blueprint for reasoning agents
- **Reasoning Operator Library** — 6 composable symbolic + meta operators
- **EvaluationCommission** — 4-dimensional scoring (correctness, logic, critique, conciseness)
- **EvolutionEngine** — Elo-based selection + crossover + mutation

---

## Quick start

```bash
git clone https://github.com/your-org/math-reasoning-engine
cd math-reasoning-engine
pip install -e ".[dev]"
pytest                          # all tests offline
```

### One-liner: run the closed-loop pipeline

```python
from mre.pipeline import MREPipeline

pipe = MREPipeline(population_size=4, generations=5, target_score=0.90)
history = pipe.run([
    {"text": "Solve x**2 - 5*x + 6 = 0",
     "context": {"equation": "x**2 - 5*x + 6 = 0"},
     "answer": "2"},
])
pipe.print_history(history)
```

### Run the benchmark

```python
from mre.benchmarks import SyntheticBenchmark, BenchmarkRunner

bench  = SyntheticBenchmark(n=20, seed=42)
runner = BenchmarkRunner(population_size=4, generations=3)
result = runner.run(bench.problems())
print(result.summary())        # accuracy by difficulty / domain
runner.plot(result)            # 2×2 dashboard
```

### Run the ablation study

```python
from mre.ablation import AblationStudy

study = AblationStudy(n_problems=12, generations=2)
results = study.run()
study.print_table(results)     # full vs no_evolution vs single_agent …
study.plot(results)            # bar chart: component contributions
```

---

## Repository layout

```
math-reasoning-engine/
├── mre/
│   ├── knowledge_graph/        # Phase 1: NRO model, MathKG, training
│   │   ├── nro_model.py
│   │   ├── training.py
│   │   ├── synthetic_kg.py
│   │   ├── mathkg_builder.py
│   │   ├── mathkg_loader.py
│   │   └── experiments.py
│   ├── agents/                 # Phase 2: DNA, state, agent pool, task manager
│   │   ├── state.py            #   ReasoningState (immutable, copy-on-write)
│   │   ├── dna.py              #   AgentDNA (5 genes + clone/mutate/crossover)
│   │   ├── agent.py            #   ReasoningAgent, ReasoningAgentPool
│   │   └── task_manager.py     #   TaskManager (multi-round orchestration)
│   ├── operators/              # Phase 2: reasoning operator library
│   │   ├── base.py             #   BaseOperator + auto-registry
│   │   ├── library.py          #   6 operators (Simplify, Deduce, Contradict …)
│   │   ├── pipeline.py         #   OperatorPipeline, PipelineResult
│   │   └── stats.py            #   OperatorStats (leaderboard, macro-ops)
│   ├── evaluation/             # Phase 3: commission + three judges
│   │   └── commission.py       #   EvaluationCommission (4-dim weighted score)
│   ├── evolution/              # Phase 3: Elo selection + evolution
│   │   ├── selection.py        #   SelectionEngine (Elo, cull, top-K)
│   │   └── engine.py           #   EvolutionEngine (elite + crossover + mutate)
│   ├── pipeline.py             # Phase 3: MREPipeline (closed-loop main entry)
│   ├── benchmarks.py           # Phase 4: BenchmarkRunner, SyntheticBenchmark
│   ├── ablation.py             # Phase 4: AblationStudy (5 conditions)
│   └── utils/
│       └── __init__.py         #   get_logger
├── notebooks/
│   ├── 01_NRO_toy_experiment.ipynb
│   ├── 02_MathKG_builder.ipynb
│   ├── 03_Phase2_AgentDNA_Operators.ipynb
│   └── 04_Phase3_4_Pipeline_Benchmark_Ablation.ipynb
├── tests/
│   ├── test_knowledge_graph/   # Phase 1 tests (24 cases)
│   ├── test_phase2/            # Phase 2 tests (75+ cases)
│   └── test_phase3/            # Phase 3 tests (36 cases)
├── configs/
│   └── kaggle_config.yaml
└── requirements.txt
```

---

## Core concepts

### Agent DNA (5 genes)

```python
from mre.agents.dna import AgentDNA

dna = AgentDNA(
    model_gene     = "claude-sonnet-4-20250514",   # LLM backend
    prompt_gene    = "rigorous",                   # thinking style
    domain_gene    = {"algebra": 0.7, ...},        # expertise weights
    tool_gene      = {"sympy": 0.9, ...},          # tool preferences
    reasoning_gene = ["SymbolicSimplify",          # operator pipeline
                      "EquationSolve",
                      "SelfCritique"],
)

# Evolution primitives
child   = dna.clone()
mutant  = dna.mutate(mutation_rate=0.25)
o1, o2  = AgentDNA.crossover(dna_a, dna_b)
```

### Reasoning Operator Library

| Operator | Category | Description |
|----------|----------|-------------|
| `SymbolicSimplify` | Symbolic | SymPy simplify → cancel → trigsimp → expand |
| `DeductiveStep` | Logic | Rule-based modus-ponens from `context['rules']` |
| `ProofByContradiction` | Logic | SymPy satisfiable + system inconsistency |
| `EquationSolve` | Algebraic | `sp.solve` with auto-extraction of equation |
| `SelfCritique` | Meta | Circular-step detection + LLM critique (optional) |
| `RepairChain` | Meta | 3-strategy targeted repair after failure |

### Evaluation Commission

```
weighted_score = 0.40 × CorrectnessJudge   (SymPy equivalence check)
              + 0.30 × LogicJudge           (step-by-step validity)
              + 0.20 × CritiqueAgent        (fault localisation)
              + 0.10 × ConcisenessScore     (fewer steps = bonus)
```

### Evolution loop

```
Round N:
  ReasoningAgentPool.solve_all()   → PipelineResult × pop_size
  EvaluationCommission.evaluate()  → weighted_score × pop_size
  SelectionEngine.update()         → update Elo ratings
  SelectionEngine.select()         → survivors, culled (bottom 20%)
  EvolutionEngine.evolve():
    elite_fraction → pass through unmutated
    crossover(top-K parents) → offspring
    mutate(non-elites)
    replenish → target_population
Round N+1 with new population
```

---

## Phase deliverables

| Phase | Module | Status | Key items |
|-------|--------|--------|-----------|
| 1 | `mre/knowledge_graph/` | ✅ Done | NROModel, SyntheticKG, MathKGBuilder, 24 tests |
| 2 | `mre/agents/`, `mre/operators/` | ✅ Done | AgentDNA (5 genes), 6 operators, TaskManager, 75+ tests |
| 3 | `mre/evaluation/`, `mre/evolution/`, `mre/pipeline.py` | ✅ Done | EvaluationCommission, Elo selection, EvolutionEngine, 36 tests |
| 4 | `mre/benchmarks.py`, `mre/ablation.py` | ✅ Done | BenchmarkRunner (100% on synthetic), AblationStudy (5 conditions) |

---

## Kaggle usage

```python
from mre.utils import get_logger
# All modules auto-detect CPU/GPU; no config changes needed for Kaggle T4
```

| Environment | Recommended `population_size` | Recommended `generations` |
|-------------|------------------------------|--------------------------|
| Local dev   | 2–4                          | 2–3                      |
| Kaggle T4   | 6–8                          | 5–10                     |
| Kaggle P100 | 8–12                         | 10–20                    |

---

## Roadmap

| Phase | Module | Status |
|-------|--------|--------|
| 1 | KG foundation (NRO, MathKG) | **✅ Done** |
| 2 | Agent DNA, reasoning operator library | **✅ Done** |
| 3 | Evaluation commission, evolution engine | **✅ Done** |
| 4 | Benchmarks, ablation study | **✅ Done** |
| 5 | Automatic theorem discovery (MCTS + novelty reward) | Planned |

---

## Citation

```bibtex
@software{mre2025,
  title  = {MRE: Evolvable Mathematical Reasoning Engine},
  year   = {2025},
  url    = {https://github.com/your-org/math-reasoning-engine}
}
```

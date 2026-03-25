from pathlib import Path

from mae.lib import KGBuilder, KnowledgeGraph, TaskSpace, ToolLibrary


def test_tool_library_defaults_are_10():
    lib = ToolLibrary.with_defaults()
    assert len(lib.tools) == 10
    assert lib.estimate_cost(["tool_001", "tool_010"]) > 0


def test_kg_builder_and_graph(tmp_path: Path):
    builder = KGBuilder()
    p1 = builder.parse_proofwiki_markdown(
        "# AM-GM\n## Statement\nFor positive reals ...\n## Proof\nUse Jensen. [[Convex Function]]",
        "k1",
    )
    p2 = builder.parse_proofwiki_markdown("# Convex Function\n## Statement\n...", "k2")
    p1["links"] = [{"relation": "depends_on", "target_id": "k2", "target_name": "Convex Function"}]

    k_path = tmp_path / "knowledge.csv"
    r_path = tmp_path / "relations.csv"
    builder.build_from_proofwiki([p1, p2], k_path, r_path)

    kg = KnowledgeGraph.from_csv(k_path, r_path)
    nbs = kg.neighbors("k1")
    assert len(nbs) == 1


def test_task_space(tmp_path: Path):
    rows = [
        {"id": "t1", "name": "IMO 1", "difficulty": 3, "bonus": 100},
        {"id": "t2", "name": "IMO 6", "difficulty": 9, "bonus": 500},
    ]
    out = tmp_path / "tasks.csv"

    builder = KGBuilder()
    builder.build_tasks_from_olympiad(rows, out)

    space = TaskSpace.from_csv(out)
    hard = space.by_difficulty(8, 10)
    assert len(hard) == 1
    assert hard[0].id == "t2"

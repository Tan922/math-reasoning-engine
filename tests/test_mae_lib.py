from pathlib import Path

from mae.lib import KGBuilder, KnowledgeGraph, TaskSpace, ToolLibrary
from mae.lib.schemas import KnowledgeFile


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


def test_load_from_api_proofwiki(monkeypatch):
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(_url, params=None, timeout=0):
        if params and params.get("list") == "categorymembers":
            return DummyResponse({"query": {"categorymembers": [{"title": "AM-GM Inequality"}]}})
        return DummyResponse(
            {
                "query": {
                    "pages": {
                        "1": {
                            "revisions": [
                                {"slots": {"main": {"*": "== Statement == x [[Convex Function]]\n== Proof == y"}}}
                            ]
                        }
                    }
                }
            }
        )

    monkeypatch.setattr("mae.lib.initializer.requests.get", fake_get)

    rows = KGBuilder()._load_from_api("proofwiki", limit=1)
    assert len(rows) == 1
    assert rows[0]["name"] == "AM-GM Inequality"
    assert rows[0]["links"][0]["target_name"] == "Convex Function"


def test_save_wrapper(tmp_path: Path):
    builder = KGBuilder()
    rows = [
        KnowledgeFile(
            id="k1",
            name="n",
            type="theorem",
            author="a",
            description="d",
            reasoning_chain="r",
            evaluator="e",
            usage_fee=0.0,
            url="u",
        )
    ]
    out = tmp_path / "knowledge.csv"
    builder.save(knowledge_rows=rows, knowledge_out=out)
    assert out.exists()


def test_load_from_api_olympiad_hf_rows(monkeypatch):
    called = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "rows": [
                    {"row": {"uid": "u1", "problem": "Prove X", "difficulty": 7}},
                ]
            }

    def fake_get(_url, params=None, timeout=0):
        called["params"] = params or {}
        called["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("mae.lib.initializer.requests.get", fake_get)
    rows = KGBuilder()._load_from_api(
        "olympiadbench",
        dataset="my/olympiad",
        config="zh",
        split="test",
        limit=1,
        timeout=9,
    )
    assert called["params"]["dataset"] == "my/olympiad"
    assert called["params"]["config"] == "zh"
    assert called["params"]["split"] == "test"
    assert called["timeout"] == 9
    assert rows[0]["name"] == "Prove X"


def test_generate_csv_files(tmp_path: Path):
    from mae.lib.initializer import generate_csv_files

    generate_csv_files(output_dir=tmp_path)

    assert (tmp_path / "knowledges.csv").exists()
    assert (tmp_path / "relations.csv").exists()
    assert (tmp_path / "tasks.csv").exists()
    assert (tmp_path / "tools.csv").exists()

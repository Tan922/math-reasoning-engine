from pathlib import Path
from urllib.error import HTTPError

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
    def fake_fetch_json(self, _url, *, params=None, timeout=0):
        if params and params.get("list") == "categorymembers":
            return {"query": {"categorymembers": [{"title": "AM-GM Inequality"}]}}
        return (
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

    monkeypatch.setattr(KGBuilder, "_fetch_json", fake_fetch_json)

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

    def fake_fetch_json(self, _url, *, params=None, timeout=0):
        called["params"] = params or {}
        called["timeout"] = timeout
        return {
            "rows": [
                {"row": {"uid": "u1", "problem": "Prove X", "difficulty": 7}},
            ]
        }

    monkeypatch.setattr(KGBuilder, "_fetch_json", fake_fetch_json)
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


def test_load_from_api_olympiad_hf_rows_fallback_on_422(monkeypatch):
    calls = []

    def fake_fetch_json(self, url, *, params=None, timeout=0):
        calls.append((url, dict(params or {}), timeout))
        if url.endswith("/rows") and params and params.get("config") == "default":
            raise HTTPError(url, 422, "Unprocessable Entity", hdrs=None, fp=None)
        if url.endswith("/splits"):
            return {
                "splits": [
                    {"config": "main", "split": "train"},
                    {"config": "main", "split": "test"},
                ]
            }
        return {"rows": [{"row": {"uid": "u2", "problem": "Fallback row"}}]}

    monkeypatch.setattr(KGBuilder, "_fetch_json", fake_fetch_json)
    rows = KGBuilder()._load_from_api(
        "olympiadbench",
        dataset="my/olympiad",
        config="default",
        split="train",
        limit=1,
        timeout=11,
    )

    assert rows[0]["name"] == "Fallback row"
    assert calls[0][0].endswith("/rows")
    assert calls[0][1]["config"] == "default"
    assert calls[1][0].endswith("/splits")
    assert calls[2][0].endswith("/rows")
    assert calls[2][1]["config"] == "main"


def test_generate_csv_files(tmp_path: Path, monkeypatch):
    from mae.lib.initializer import generate_csv_files
    from mae.lib.initializer import KGBuilder

    def fake_load_from_api(self, source: str, **kwargs):
        if source == "proofwiki":
            return [
                {
                    "id": "1",
                    "name": "Theorem: Sample",
                    "type": "theorem",
                    "description": "Sample theorem description",
                    "url": "https://proofwiki.org/wiki/Theorem:Sample",
                    "links": [{"target_id": "2", "target_name": "Definition: SampleDef", "relation": "depends_on"}],
                },
                {
                    "id": "2",
                    "name": "Definition: SampleDef",
                    "type": "definition",
                    "description": "Sample definition description",
                    "url": "https://proofwiki.org/wiki/Definition:SampleDef",
                    "links": [],
                }
            ]
        if source == "olympiadbench":
            return [
                {
                    "id": "t1",
                    "name": "Sample Olympiad Problem",
                    "description": "Prove a simple identity.",
                    "difficulty": 5.0,
                }
            ]
        raise AssertionError(f"unexpected source: {source}")

    monkeypatch.setattr(KGBuilder, "_load_from_api", fake_load_from_api)

    generate_csv_files(output_dir=tmp_path)

    assert (tmp_path / "knowledges.csv").exists()
    assert (tmp_path / "relations.csv").exists()
    assert (tmp_path / "tasks.csv").exists()
    assert (tmp_path / "tools.csv").exists()

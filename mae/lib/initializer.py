from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence
import csv
import re
import sys

import requests

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from mae.lib.schemas import KnowledgeFile, RelationFile, TaskFile, save_records
    from mae.lib.tools import ToolLibrary
else:
    from .schemas import KnowledgeFile, RelationFile, TaskFile, save_records
    from .tools import ToolLibrary

PROOFWIKI_API = "https://proofwiki.org/w/api.php"
OLYMPIAD_DATA_SOURCES = {
    "hf_rows": "https://datasets-server.huggingface.co/rows"
}


class KGBuilder:
    """从 ProofWiki 与奥林匹克数据源生成结构化文件。"""

    def build_from_proofwiki(
        self,
        proofwiki_rows: Sequence[dict],
        knowledge_out: str | Path,
        relation_out: str | Path,
        default_author: str = "ProofWiki",
        default_evaluator: str = "MAE-Auto-Eval",
        default_usage_fee: float = 0.0,
    ) -> tuple[List[KnowledgeFile], List[RelationFile]]:
        """根据 ProofWiki 结构化行生成知识文件与关系文件。

        每条行数据推荐字段：
        - id/name/type/description/url/author/reasoning_chain
        - links: list[dict], 元素包含 relation + target_id + target_name
        """
        knowledge_rows: List[KnowledgeFile] = []
        relation_rows: List[RelationFile] = []

        for row in proofwiki_rows:
            k = KnowledgeFile(
                id=str(row["id"]),
                name=row["name"],
                type=row.get("type", "theorem"),
                author=row.get("author", default_author),
                description=row.get("description", ""),
                reasoning_chain=row.get("reasoning_chain", ""),
                evaluator=row.get("evaluator", default_evaluator),
                usage_fee=float(row.get("usage_fee", default_usage_fee)),
                url=row.get("url", "https://proofwiki.org"),
            )
            knowledge_rows.append(k)

            for link in row.get("links", []):
                relation_rows.append(
                    RelationFile(
                        head_id=k.id,
                        head_name=k.name,
                        relation=link.get("relation", "related_to"),
                        tail_id=str(link["target_id"]),
                        tail_name=link["target_name"],
                    )
                )

        save_records(knowledge_rows, knowledge_out)
        save_records(relation_rows, relation_out)
        return knowledge_rows, relation_rows

    def build_tasks_from_olympiad(
        self,
        olympiad_rows: Sequence[dict],
        task_out: str | Path,
        default_author: str = "OlympiadDataset",
        default_bonus: float = 100.0,
    ) -> List[TaskFile]:
        """从奥林匹克题目数据集行构建任务文件。"""
        tasks: List[TaskFile] = []
        for row in olympiad_rows:
            tasks.append(
                TaskFile(
                    id=str(row["id"]),
                    name=row["name"],
                    type=row.get("type", "olympiad_problem"),
                    author=row.get("author", default_author),
                    description=row.get("description", ""),
                    knowledge_amount=float(
                        row.get("knowledge_amount", row.get("difficulty", 1.0))
                    ),
                    bonus=float(row.get("bonus", default_bonus)),
                )
            )
        save_records(tasks, task_out)
        return tasks

    def parse_proofwiki_markdown(self, markdown_text: str, page_id: str) -> dict:
        """将简化版 ProofWiki markdown 解析为结构化条目。"""
        title_match = re.search(r"^#\s+(.+)$", markdown_text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else f"ProofWiki Entry {page_id}"

        desc_match = re.search(r"##\s+Statement\s*(.+?)(?:\n##|\Z)", markdown_text, re.S)
        description = _clean(desc_match.group(1)) if desc_match else ""

        proof_match = re.search(r"##\s+Proof\s*(.+?)(?:\n##|\Z)", markdown_text, re.S)
        reasoning = _clean(proof_match.group(1)) if proof_match else ""

        links = []
        for m in re.finditer(r"\[\[(.+?)\]\]", markdown_text):
            target = m.group(1).strip()
            links.append(
                {
                    "relation": "depends_on",
                    "target_id": _slug(target),
                    "target_name": target,
                }
            )

        return {
            "id": page_id,
            "name": title,
            "type": "theorem",
            "description": description,
            "reasoning_chain": reasoning,
            "url": f"https://proofwiki.org/wiki/{_slug(title)}",
            "links": links,
        }

    def load_rows_from_csv(self, path: str | Path) -> List[dict]:
        source = Path(path)
        with source.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _load_from_api(
        self,
        source: str,
        *,
        limit: int = 50,
        timeout: int = 20,
        **kwargs: object,
    ) -> List[dict]:
        """从 API 加载 ProofWiki 或奥林匹克题目数据。"""
        normalized = source.strip().lower()
        if normalized == "proofwiki":
            category = str(kwargs.get("category", "Proven Results"))
            return self._load_proofwiki_rows(limit=limit, category=category, timeout=timeout)
        if normalized in {"olympiadbench", "olympiad", "hf_rows"}:
            dataset = str(kwargs.get("dataset", "Hothan/OlympiadBench"))
            config = str(kwargs.get("config", "default"))
            split = str(kwargs.get("split", "train"))
            return self._load_hf_rows(dataset=dataset, config=config, split=split, limit=limit, timeout=timeout)
        raise ValueError(f"不支持的数据源: {source}")

    def save(
        self,
        *,
        knowledge_rows: Sequence[KnowledgeFile] | None = None,
        relation_rows: Sequence[RelationFile] | None = None,
        task_rows: Sequence[TaskFile] | None = None,
        knowledge_out: str | Path | None = None,
        relation_out: str | Path | None = None,
        task_out: str | Path | None = None,
    ) -> None:
        """统一保存构建结果；至少保存一类记录。"""
        written = 0
        if knowledge_rows is not None:
            if knowledge_out is None:
                raise ValueError("knowledge_rows 已提供时必须给 knowledge_out")
            save_records(knowledge_rows, knowledge_out)
            written += 1
        if relation_rows is not None:
            if relation_out is None:
                raise ValueError("relation_rows 已提供时必须给 relation_out")
            save_records(relation_rows, relation_out)
            written += 1
        if task_rows is not None:
            if task_out is None:
                raise ValueError("task_rows 已提供时必须给 task_out")
            save_records(task_rows, task_out)
            written += 1
        if written == 0:
            raise ValueError("至少需要提供一类可保存的数据")

    def _load_proofwiki_rows(self, *, limit: int, category: str, timeout: int) -> List[dict]:
        rows: List[dict] = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 0,
            "cmlimit": min(limit, 50),
            "format": "json",
        }
        response = requests.get(PROOFWIKI_API, params=params, timeout=timeout)
        response.raise_for_status()
        members = response.json().get("query", {}).get("categorymembers", [])

        for page in members[:limit]:
            title = page.get("title", "")
            if not title:
                continue
            detail = requests.get(
                PROOFWIKI_API,
                params={
                    "action": "query",
                    "titles": title,
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "format": "json",
                },
                timeout=timeout,
            )
            detail.raise_for_status()
            wikitext = ""
            pages = detail.json().get("query", {}).get("pages", {})
            for entry in pages.values():
                rev = (entry.get("revisions") or [{}])[0]
                wikitext = rev.get("slots", {}).get("main", {}).get("*", "")
            rows.append(self._proofwiki_row_from_wikitext(title=title, wikitext=wikitext))
        return rows

    def _proofwiki_row_from_wikitext(self, *, title: str, wikitext: str) -> dict:
        statement = ""
        proof = ""
        st = re.search(r"==\s*(?:Theorem|Statement|Definition)\s*==(.+?)(?:\n==|\Z)", wikitext, re.S)
        if st:
            statement = _clean(st.group(1))
        pf = re.search(r"==\s*Proof\s*==(.+?)(?:\n==|\Z)", wikitext, re.S)
        if pf:
            proof = _clean(pf.group(1))

        links = []
        for m in re.finditer(r"\[\[([^\]|#\n]+)", wikitext):
            target = m.group(1).strip()
            if target and target != title and not target.startswith(("File:", "Category:", "Template:")):
                links.append(
                    {
                        "relation": "depends_on",
                        "target_id": _slug(target),
                        "target_name": target,
                    }
                )

        return {
            "id": _slug(title),
            "name": title,
            "type": "theorem",
            "description": statement,
            "reasoning_chain": proof,
            "url": f"https://proofwiki.org/wiki/{title.replace(' ', '_')}",
            "links": links,
        }

    def _load_hf_rows(
        self,
        *,
        dataset: str,
        config: str,
        split: str,
        limit: int,
        timeout: int,
    ) -> List[dict]:
        response = requests.get(
            OLYMPIAD_DATA_SOURCES["hf_rows"],
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": 0,
                "length": limit,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json().get("rows", [])
        rows: List[dict] = []
        for i, item in enumerate(data):
            row = item.get("row", {})
            prompt = (
                row.get("question")
                or row.get("problem")
                or row.get("prompt")
                or row.get("题目")
                or ""
            )
            rows.append(
                {
                    "id": str(row.get("id", row.get("uid", i))),
                    "name": prompt or f"Olympiad Problem {i + 1}",
                    "type": "olympiad_problem",
                    "description": prompt,
                    "difficulty": row.get("level", row.get("difficulty", 1)),
                }
            )
        return rows



def _clean(text: str) -> str:
    return " ".join(text.split())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def generate_csv_files(output_dir: str | Path = ".") -> None:
    """生成 knowledges/relations/tasks/tools 四个 CSV 初始化文件。"""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    builder = KGBuilder()
    proof_rows = [
        builder.parse_proofwiki_markdown(
            "# AM-GM Inequality\n## Statement\nFor positive reals, arithmetic mean >= geometric mean.\n## Proof\nApply convexity/Jensen. [[Convex Function]]",
            "k1",
        ),
        builder.parse_proofwiki_markdown(
            "# Convex Function\n## Statement\nA function is convex if its secants lie above the graph.",
            "k2",
        ),
    ]
    proof_rows[0]["links"] = [
        {
            "relation": "depends_on",
            "target_id": "k2",
            "target_name": "Convex Function",
        }
    ]

    builder.build_from_proofwiki(
        proof_rows,
        knowledge_out=base / "knowledges.csv",
        relation_out=base / "relations.csv",
    )

    olympiad_rows = [
        {"id": "t1", "name": "IMO Example 1", "difficulty": 3, "bonus": 100},
        {"id": "t2", "name": "IMO Example 2", "difficulty": 8, "bonus": 300},
    ]
    builder.build_tasks_from_olympiad(olympiad_rows, task_out=base / "tasks.csv")
    ToolLibrary.with_defaults().save(base / "tools.csv")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize MAE CSV files: knowledges.csv, relations.csv, tasks.csv, tools.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated csv files (default: current directory).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    generate_csv_files(args.output_dir)


if __name__ == "__main__":
    main()

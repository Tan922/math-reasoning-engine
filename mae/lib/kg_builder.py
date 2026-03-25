from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence
import csv
import re

from .schemas import KnowledgeFile, RelationFile, TaskFile, save_records


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



def _clean(text: str) -> str:
    return " ".join(text.split())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")

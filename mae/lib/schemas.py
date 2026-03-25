from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Type, TypeVar
import csv

T = TypeVar("T")


@dataclass
class TaskFile:
    """任务文件。

    字段：ID，名称，类型，作者，描述，知识量（难度指数），奖金
    """

    id: str
    name: str
    type: str
    author: str
    description: str
    knowledge_amount: float
    bonus: float


@dataclass
class KnowledgeFile:
    """知识文件。

    字段：ID，名称，类型，作者，描述，推理链，评估者，使用费，URL
    """

    id: str
    name: str
    type: str
    author: str
    description: str
    reasoning_chain: str
    evaluator: str
    usage_fee: float
    url: str


@dataclass
class RelationFile:
    """关系文件。字段：head_id, head_name, relation, tail_id, tail_name"""

    head_id: str
    head_name: str
    relation: str
    tail_id: str
    tail_name: str


@dataclass
class ToolFile:
    """工具文件。字段：ID，名称，类型，作者，描述，使用费"""

    id: str
    name: str
    type: str
    author: str
    description: str
    usage_fee: float


def save_records(records: Sequence[T], path: str | Path) -> None:
    """将 dataclass 记录保存为 CSV。"""
    if not records:
        raise ValueError("records 不能为空")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(x) for x in records]
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_records(path: str | Path, cls: Type[T]) -> List[T]:
    """从 CSV 加载到 dataclass 列表。"""
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as f:
        rows: Iterable[Dict[str, str]] = csv.DictReader(f)
        parsed: List[T] = []
        for row in rows:
            parsed.append(cls(**_coerce_row_types(row, cls)))
    return parsed


def _coerce_row_types(row: Dict[str, str], cls: Type[T]) -> Dict[str, object]:
    fields = getattr(cls, "__dataclass_fields__")
    out: Dict[str, object] = {}
    for key, value in row.items():
        tp = fields[key].type
        if tp in (float, "float"):
            out[key] = float(value)
        elif tp in (int, "int"):
            out[key] = int(value)
        else:
            out[key] = value
    return out

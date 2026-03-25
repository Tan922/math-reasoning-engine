from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set

from .schemas import KnowledgeFile, RelationFile, load_records


class KnowledgeGraph:
    """基于知识文件和关系文件的轻量知识图谱模块。"""

    def __init__(self, knowledge: List[KnowledgeFile], relations: List[RelationFile]) -> None:
        self.knowledge = {k.id: k for k in knowledge}
        self.relations = relations
        self.out_edges: Dict[str, List[RelationFile]] = defaultdict(list)
        self.in_edges: Dict[str, List[RelationFile]] = defaultdict(list)
        for rel in relations:
            self.out_edges[rel.head_id].append(rel)
            self.in_edges[rel.tail_id].append(rel)

    @classmethod
    def from_csv(cls, knowledge_path: str | Path, relation_path: str | Path) -> "KnowledgeGraph":
        knowledge = load_records(knowledge_path, KnowledgeFile)
        relations = load_records(relation_path, RelationFile)
        return cls(knowledge, relations)

    def neighbors(self, node_id: str, relation: str | None = None) -> List[KnowledgeFile]:
        edges = self.out_edges.get(node_id, [])
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return [self.knowledge[e.tail_id] for e in edges if e.tail_id in self.knowledge]

    def prerequisites(self, node_id: str) -> List[KnowledgeFile]:
        return [self.knowledge[e.head_id] for e in self.in_edges.get(node_id, []) if e.head_id in self.knowledge]

    def shortest_path(self, start_id: str, end_id: str) -> List[str]:
        if start_id == end_id:
            return [start_id]

        q = deque([start_id])
        prev: Dict[str, str] = {}
        seen: Set[str] = {start_id}

        while q:
            cur = q.popleft()
            for edge in self.out_edges.get(cur, []):
                nxt = edge.tail_id
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = cur
                if nxt == end_id:
                    return _trace(prev, start_id, end_id)
                q.append(nxt)
        return []


def _trace(prev: Dict[str, str], start: str, end: str) -> List[str]:
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return list(reversed(path))

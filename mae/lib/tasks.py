from __future__ import annotations

from pathlib import Path
from random import Random
from typing import List

from .schemas import TaskFile, load_records


class TaskSpace:
    """基于任务文件的任务空间模块。"""

    def __init__(self, tasks: List[TaskFile]) -> None:
        self.tasks = tasks

    @classmethod
    def from_csv(cls, task_path: str | Path) -> "TaskSpace":
        return cls(load_records(task_path, TaskFile))

    def by_type(self, task_type: str) -> List[TaskFile]:
        return [t for t in self.tasks if t.type == task_type]

    def by_difficulty(self, min_value: float = 0.0, max_value: float = 10.0) -> List[TaskFile]:
        return [t for t in self.tasks if min_value <= t.knowledge_amount <= max_value]

    def top_bonus(self, n: int = 5) -> List[TaskFile]:
        return sorted(self.tasks, key=lambda t: t.bonus, reverse=True)[:n]

    def sample(self, n: int = 1, seed: int = 42) -> List[TaskFile]:
        rng = Random(seed)
        if n >= len(self.tasks):
            return list(self.tasks)
        return rng.sample(self.tasks, n)

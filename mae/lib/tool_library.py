from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .schemas import ToolFile, load_records, save_records


class ToolLibrary:
    """基于工具文件的工具库模块。"""

    def __init__(self, tools: List[ToolFile]) -> None:
        self.tools = tools

    @classmethod
    def with_defaults(cls) -> "ToolLibrary":
        """内置 10 个常用数学推理工具。"""
        tools = [
            ToolFile("tool_001", "SymPy", "symbolic", "MAE", "符号化简与求解", 0.02),
            ToolFile("tool_002", "NumPy", "numeric", "MAE", "数值计算和线性代数", 0.01),
            ToolFile("tool_003", "SciPy", "optimization", "MAE", "优化与科学计算", 0.03),
            ToolFile("tool_004", "NetworkX", "graph", "MAE", "图结构与路径分析", 0.02),
            ToolFile("tool_005", "Pandas", "data", "MAE", "表格数据清洗处理", 0.01),
            ToolFile("tool_006", "Matplotlib", "visualization", "MAE", "结果可视化", 0.01),
            ToolFile("tool_007", "Z3", "solver", "MAE", "约束求解与验证", 0.05),
            ToolFile("tool_008", "PyTorch", "ml", "MAE", "神经模型训练", 0.04),
            ToolFile("tool_009", "WolframAPI", "knowledge", "MAE", "外部知识检索", 0.08),
            ToolFile("tool_010", "LeanChecker", "proof", "MAE", "形式化证明检查", 0.06),
        ]
        return cls(tools)

    @classmethod
    def from_csv(cls, tool_path: str | Path) -> "ToolLibrary":
        return cls(load_records(tool_path, ToolFile))

    def save(self, path: str | Path) -> None:
        save_records(self.tools, path)

    def add_tools(self, new_tools: Iterable[ToolFile]) -> None:
        self.tools.extend(new_tools)

    def get_by_type(self, tool_type: str) -> List[ToolFile]:
        return [t for t in self.tools if t.type == tool_type]

    def estimate_cost(self, tool_ids: Iterable[str]) -> float:
        wanted = set(tool_ids)
        return sum(tool.usage_fee for tool in self.tools if tool.id in wanted)

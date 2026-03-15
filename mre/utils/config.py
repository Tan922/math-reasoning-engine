"""
mre.utils.config
────────────────
Load and access YAML configuration.  Supports dot-notation access,
automatic device resolution, and automatic path resolution.

Path resolution
───────────────
Any config value whose key ends with ``_dir`` and whose value is a relative
path is resolved to an absolute path at load time, anchored at the project
root.  The project root is the nearest ancestor directory that contains a
``configs/`` subdirectory.

This means notebooks and scripts always get the same absolute path regardless
of their working directory (``notebooks/``, ``/kaggle/working``, repo root, …).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# ── Project-root detection ────────────────────────────────────────────────────

def _find_project_root(start: Path) -> Path:
    """
    Walk upward from *start* until a directory containing ``configs/`` is found.
    Falls back to *start* if nothing is found.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "configs").is_dir():
            return candidate
    return start


def _resolve_dirs(data: dict, project_root: Path) -> dict:
    """
    Recursively walk *data* and resolve every value whose key ends with
    ``_dir`` from a relative path to an absolute path under *project_root*.
    Absolute paths and non-string values are left untouched.
    """
    resolved = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_dirs(value, project_root)
        elif key.endswith("_dir") and isinstance(value, str):
            p = Path(value)
            resolved[key] = str(project_root / p) if not p.is_absolute() else value
        else:
            resolved[key] = value
    return resolved


# ── Config wrapper ────────────────────────────────────────────────────────────

class Config:
    """Thin wrapper around a nested dict that supports dot-notation access."""

    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def to_dict(self) -> dict:
        return self._data


# ── Public API ────────────────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> Config:
    """
    Load configuration from *path*, resolve all ``*_dir`` values to absolute
    paths, and return a :class:`Config` object.

    Search order when *path* is ``None``:
      1. ``MRE_CONFIG`` environment variable
      2. ``configs/kaggle_config.yaml`` located by walking up from this file
      3. Built-in defaults

    After loading, every relative ``*_dir`` value is resolved relative to the
    project root (the directory that contains ``configs/``), so callers never
    need to think about the current working directory.
    """
    config_path: Path | None = None

    if path is None:
        env_path = os.environ.get("MRE_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            here = Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "configs" / "kaggle_config.yaml"
                if candidate.exists():
                    path = candidate
                    break

    if path is not None and Path(path).exists():
        config_path = Path(path).resolve()
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        data = _defaults()

    # Determine project root:
    #   • if we loaded a file, root = directory two levels up from configs/file
    #     (i.e. the directory that *contains* configs/)
    #   • otherwise search upward from CWD
    if config_path is not None:
        project_root = config_path.parent.parent   # .../configs/kaggle_config.yaml → ...
    else:
        project_root = _find_project_root(Path.cwd())

    data = _resolve_dirs(data, project_root)
    return Config(data)


def resolve_device(cfg: Config) -> str:
    """Return the concrete device string (``'cuda'`` or ``'cpu'``)."""
    try:
        import torch
        requested = getattr(cfg.hardware, "device", "auto")
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested
    except ImportError:
        return "cpu"


# ── Built-in defaults ─────────────────────────────────────────────────────────

def _defaults() -> dict:
    return {
        "env": "local",
        "hardware": {"device": "auto", "fp16": False, "dataloader_workers": 2},
        "knowledge_graph": {
            "max_entities": 500,
            "embed_dim": 64,
            "hidden_dim": 128,
            "noise_std": 0.05,
            "data_dir": "data/mathkg",
            "confidence_threshold": 0.65,
            "min_entity_degree": 1,
        },
        "nro": {
            "embed_dim": 64,
            "hidden_dim": 128,
            "base_relations": ["depends_on", "generalizes", "equivalent_to", "applied_in"],
            "train_epochs": 60,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "compose_init_steps": 500,
            "compose_init_lr": 1e-3,
            "finetune_epochs": 40,
            "finetune_lr": 5e-4,
        },
        "experiment": {
            "seed": 42,
            "n_trials": 5,
            "k_shots": [1, 5, 10, 25, 50, 100, "full"],
            "composite_relation": "gen_dep",
            "composite_chain": ["generalizes", "depends_on"],
            "hits_k": 10,
        },
        "logging": {
            "level": "INFO",
            "save_checkpoints": True,
            "checkpoint_dir": "data/checkpoints",
            "results_dir": "data/results",
        },
    }

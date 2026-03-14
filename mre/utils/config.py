"""
mre.utils.config
────────────────
Load and access YAML configuration.  Supports dot-notation access and
automatic device resolution (auto → cuda / cpu).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a nested dict that supports attribute access."""

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


def load_config(path: str | Path | None = None) -> Config:
    """
    Load configuration from *path*.

    Search order when *path* is None:
      1. ``MRE_CONFIG`` environment variable
      2. ``configs/kaggle_config.yaml`` relative to the project root
      3. Built-in defaults
    """
    if path is None:
        env_path = os.environ.get("MRE_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            # Walk up from this file to find the project root
            here = Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "configs" / "kaggle_config.yaml"
                if candidate.exists():
                    path = candidate
                    break

    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        data = _defaults()

    return Config(data)


def resolve_device(cfg: Config) -> str:
    """Return the concrete device string ('cuda' or 'cpu')."""
    import torch

    requested = getattr(cfg.hardware, "device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


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

"""
Unit tests for config and seed utilities.
"""

import pytest

from mre.utils.config import Config, load_config, _defaults
from mre.utils.seed import set_seed, get_logger


def test_load_config_defaults():
    cfg = load_config(path=None)
    assert hasattr(cfg, "hardware")
    assert hasattr(cfg, "nro")
    assert hasattr(cfg, "knowledge_graph")
    assert hasattr(cfg, "experiment")


def test_config_dot_access():
    cfg = load_config()
    # Nested access
    assert cfg.nro.embed_dim == 64
    assert cfg.experiment.seed == 42


def test_config_from_yaml(tmp_path):
    yaml_text = """
env: test
hardware:
  device: cpu
  fp16: false
  dataloader_workers: 1
nro:
  embed_dim: 32
  hidden_dim: 64
  base_relations: [depends_on]
  train_epochs: 5
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 16
  compose_init_steps: 10
  compose_init_lr: 0.001
  finetune_epochs: 5
  finetune_lr: 0.001
knowledge_graph:
  max_entities: 100
  embed_dim: 32
  hidden_dim: 64
  noise_std: 0.05
  data_dir: data/mathkg
  confidence_threshold: 0.65
  min_entity_degree: 1
experiment:
  seed: 7
  n_trials: 2
  k_shots: [1, 5]
  composite_relation: gen_dep
  composite_chain: [generalizes, depends_on]
  hits_k: 10
logging:
  level: DEBUG
  save_checkpoints: false
  checkpoint_dir: /tmp
  results_dir: /tmp
"""
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml_text)
    cfg = load_config(path=str(p))
    assert cfg.env == "test"
    assert cfg.nro.embed_dim == 32
    assert cfg.experiment.seed == 7


def test_set_seed_reproducible():
    import random
    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b


def test_get_logger_returns_logger():
    import logging
    logger = get_logger("test_module", level="WARNING")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.WARNING

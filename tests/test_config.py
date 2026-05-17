"""Tests for :mod:`chuck_dreamer.config` — primarily ``warmup_source``
normalization, alias merging, and override precedence."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config


def _write_cfg(tmp_path: Path, overrides: dict) -> Path:
  """Build a minimal config file by patching the default with ``overrides``."""
  base = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
  patched = OmegaConf.merge(base, OmegaConf.create(overrides))
  out = tmp_path / "cfg.yaml"
  out.write_text(OmegaConf.to_yaml(patched))
  return out


# ---------------------------------------------------------------------------
# warmup_source normalization
# ---------------------------------------------------------------------------


def test_warmup_source_string_expands_to_single_entry(tmp_path):
  cfg_path = _write_cfg(tmp_path, {
    "training": {"data": {
      "warmup_source": "data/foo/",
      "warmup_format": "hdf5",
      "warmup_num_episodes": 42,
    }},
  })
  cfg = load_config(str(cfg_path))
  src = list(cfg.training.data.warmup_source)
  assert src == [{"path": "data/foo/", "format": "hdf5", "num_episodes": 42}]


def test_warmup_source_list_inherits_defaults_per_entry(tmp_path):
  cfg_path = _write_cfg(tmp_path, {
    "training": {"data": {
      "warmup_source": [
        "data/sim/",
        {"path": "data/real/", "num_episodes": None},
        {"path": "data/extra/", "format": "hdf5"},
      ],
      "warmup_format": "rerun",
      "warmup_num_episodes": 100,
    }},
  })
  cfg = load_config(str(cfg_path))
  src = [dict(e) for e in cfg.training.data.warmup_source]
  assert src == [
    {"path": "data/sim/",   "format": "rerun", "num_episodes": 100},
    {"path": "data/real/",  "format": "rerun", "num_episodes": None},
    {"path": "data/extra/", "format": "hdf5",  "num_episodes": 100},
  ]


def test_warmup_source_dict_form_inherits_defaults(tmp_path):
  cfg_path = _write_cfg(tmp_path, {
    "training": {"data": {
      "warmup_source": {"path": "data/only/"},
      "warmup_format": "hdf5",
      "warmup_num_episodes": 7,
    }},
  })
  cfg = load_config(str(cfg_path))
  src = list(cfg.training.data.warmup_source)
  assert src == [{"path": "data/only/", "format": "hdf5", "num_episodes": 7}]


def test_warmup_source_normalization_strips_top_level_defaults(tmp_path):
  cfg_path = _write_cfg(tmp_path, {
    "training": {"data": {
      "warmup_source": "data/foo/",
      "warmup_format": "rerun",
      "warmup_num_episodes": 1,
    }},
  })
  cfg = load_config(str(cfg_path))
  # After normalization the top-level defaults are gone so no consumer
  # accidentally reads a stale value.
  assert "warmup_format"       not in cfg.training.data
  assert "warmup_num_episodes" not in cfg.training.data


# ---------------------------------------------------------------------------
# Override + alias plumbing
# ---------------------------------------------------------------------------


def test_string_override_re_normalizes(tmp_path):
  cfg_path = _write_cfg(tmp_path, {
    "training": {"data": {"warmup_source": "data/foo/"}},
  })
  cfg = load_config(str(cfg_path), overrides=("training.data.warmup_source=data/bar/",))
  src = list(cfg.training.data.warmup_source)
  assert len(src) == 1
  assert src[0]["path"] == "data/bar/"


def test_list_override_via_yaml_literal_re_normalizes(tmp_path):
  cfg_path = _write_cfg(tmp_path, {
    "training": {"data": {"warmup_source": "data/foo/"}},
  })
  cfg = load_config(
    str(cfg_path),
    overrides=("training.data.warmup_source=[data/a/,data/b/]",),
  )
  paths = [e["path"] for e in cfg.training.data.warmup_source]
  assert paths == ["data/a/", "data/b/"]


def test_aliases_apply_before_overrides(tmp_path):
  cfg_path = _write_cfg(tmp_path, {})
  # Same key set via alias and override — override must win.
  cfg = load_config(
    str(cfg_path),
    overrides=("seed=999",),
    aliases={"seed": 42},
  )
  assert int(cfg.seed) == 999


def test_aliases_skip_none_values(tmp_path):
  cfg_path = _write_cfg(tmp_path, {"seed": 7})
  cfg = load_config(str(cfg_path), aliases={"seed": None})
  # None aliases are dropped, so the YAML seed wins.
  assert int(cfg.seed) == 7


def test_aliases_set_nested_keys(tmp_path):
  cfg_path = _write_cfg(tmp_path, {})
  cfg = load_config(
    str(cfg_path),
    aliases={"logging.experiment_name": "run-1"},
  )
  assert cfg.logging.experiment_name == "run-1"


def test_seed_randomized_when_unset(tmp_path):
  cfg_path = _write_cfg(tmp_path, {"seed": None})
  cfg = load_config(str(cfg_path))
  assert isinstance(int(cfg.seed), int)
  assert int(cfg.seed) >= 0

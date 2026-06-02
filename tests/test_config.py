"""Tests for :mod:`chuck_dreamer.config` — primarily ``warmup_source``
normalization, alias merging, override precedence, and ``lookup_device``."""

from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from chuck_dreamer.config import load_config, lookup_device


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


# ---------------------------------------------------------------------------
# lookup_device
# ---------------------------------------------------------------------------


def _mock_backends(monkeypatch, *, mlx: bool, cuda: bool, mps: bool) -> None:
  """Pin every backend-availability probe so ``"auto"`` resolution is
  deterministic regardless of the host the tests run on (this dev box has
  mlx + mps; CI may have cuda or none)."""
  import chuck_dreamer.config as config_mod
  import torch

  monkeypatch.setattr(config_mod, "_mlx_available", lambda: mlx)
  monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
  # ``torch.backends.mps`` may be absent on some builds; the resolver
  # guards with getattr, so only patch is_available when it exists.
  if getattr(torch.backends, "mps", None) is not None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def _mock_backends_no_mlx_no_torch(monkeypatch) -> None:
  """Simulate a host with no mlx and no importable torch: mlx probe is
  False and ``import torch`` inside the resolver raises ImportError."""
  import chuck_dreamer.config as config_mod

  monkeypatch.setattr(config_mod, "_mlx_available", lambda: False)
  real_import = builtins.__import__

  def no_torch(name, *args, **kwargs):
    if name == "torch":
      raise ImportError("torch not installed")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", no_torch)


@pytest.mark.parametrize("value", ["cuda", "cuda:1", "mps", "cpu"])
def test_lookup_device_explicit_value_passes_through(value):
  cfg = OmegaConf.create({"hardware": {"device": value}})
  # An explicit device is returned verbatim — no discovery, even if the
  # host has a "better" backend available.
  assert lookup_device(cfg, "hardware.device") == value


def test_lookup_device_lowercases():
  cfg = OmegaConf.create({"hardware": {"device": "CUDA:1"}})
  assert lookup_device(cfg, "hardware.device") == "cuda:1"


def test_lookup_device_reads_the_named_dotted_path():
  # Two device fields live in one config; the path picks which one.
  cfg = OmegaConf.create({
    "hardware": {"device": "cpu"},
    "object_localization": {"device": "cuda:0"},
  })
  assert lookup_device(cfg, "hardware.device") == "cpu"
  assert lookup_device(cfg, "object_localization.device") == "cuda:0"


def test_lookup_device_auto_prefers_mlx_when_present(monkeypatch):
  # mlx wins discovery whenever importable, even with cuda + mps available.
  _mock_backends(monkeypatch, mlx=True, cuda=True, mps=True)
  cfg = OmegaConf.create({"hardware": {"device": "auto"}})
  assert lookup_device(cfg, "hardware.device") == "mlx"


def test_lookup_device_auto_skips_mlx_under_pytorch_only(monkeypatch):
  # A torch-only call site must never get "mlx" from discovery, even when
  # mlx is importable — it falls through to the torch backends.
  _mock_backends(monkeypatch, mlx=True, cuda=True, mps=True)
  cfg = OmegaConf.create({"hardware": {"device": "auto"}})
  assert lookup_device(cfg, "hardware.device", pytorch_only=True) == "cuda"


def test_lookup_device_auto_prefers_cuda_when_no_mlx(monkeypatch):
  _mock_backends(monkeypatch, mlx=False, cuda=True, mps=True)
  cfg = OmegaConf.create({"hardware": {"device": "auto"}})
  assert lookup_device(cfg, "hardware.device") == "cuda"


def test_lookup_device_auto_picks_mps_and_sets_fallback_env(monkeypatch):
  import torch
  if getattr(torch.backends, "mps", None) is None:
    pytest.skip("torch build has no mps backend to probe")
  _mock_backends(monkeypatch, mlx=False, cuda=False, mps=True)
  monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
  cfg = OmegaConf.create({"hardware": {"device": "auto"}})
  assert lookup_device(cfg, "hardware.device") == "mps"
  # mps lacks some kernels; the resolver opts into CPU fallback so those
  # ops don't hard-error mid-run.
  assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_lookup_device_auto_falls_back_to_cpu(monkeypatch):
  _mock_backends(monkeypatch, mlx=False, cuda=False, mps=False)
  cfg = OmegaConf.create({"hardware": {"device": "auto"}})
  assert lookup_device(cfg, "hardware.device") == "cpu"


def test_lookup_device_missing_key_resolves_like_auto(monkeypatch):
  _mock_backends(monkeypatch, mlx=False, cuda=True, mps=False)
  cfg = OmegaConf.create({"hardware": {}})
  # No device key at all → discovery, same as "auto".
  assert lookup_device(cfg, "hardware.device") == "cuda"


def test_lookup_device_none_resolves_like_auto(monkeypatch):
  _mock_backends(monkeypatch, mlx=False, cuda=True, mps=False)
  cfg = OmegaConf.create({"hardware": {"device": None}})
  assert lookup_device(cfg, "hardware.device") == "cuda"


def test_lookup_device_auto_without_any_backend_is_cpu(monkeypatch):
  # No mlx and torch unimportable → "auto" degrades to cpu rather than
  # raising.
  _mock_backends_no_mlx_no_torch(monkeypatch)
  cfg = OmegaConf.create({"hardware": {"device": "auto"}})
  assert lookup_device(cfg, "hardware.device") == "cpu"


def test_lookup_device_mlx_passes_through_by_default():
  cfg = OmegaConf.create({"hardware": {"device": "mlx"}})
  assert lookup_device(cfg, "hardware.device") == "mlx"


def test_lookup_device_pytorch_only_rejects_mlx():
  cfg = OmegaConf.create({"hardware": {"device": "mlx"}})
  with pytest.raises(ValueError, match="mlx"):
    lookup_device(cfg, "hardware.device", pytorch_only=True)


def test_lookup_device_pytorch_only_allows_non_mlx():
  # pytorch_only only gates "mlx"; everything else behaves as normal.
  cfg = OmegaConf.create({"hardware": {"device": "cuda:1"}})
  assert lookup_device(cfg, "hardware.device", pytorch_only=True) == "cuda:1"

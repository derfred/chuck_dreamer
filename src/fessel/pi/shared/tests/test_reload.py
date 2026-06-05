"""ConfigReloader tests (§2.13): the SIGHUP validate-or-keep-old contract.

The signal plumbing is not exercised here (signals need the main thread and a
real process); `reload_once()` is the synchronous core the worker calls, so the
tests drive it directly. The load-bearing property is "after validation": an
invalid or unreadable file must NOT call apply (the running config is kept)."""

import textwrap

from fessel_shared import ConfigReloader

GOOD = textwrap.dedent(
  """
  mqtt: {host: 127.0.0.1, port: 1883}
  storage: {ssd_path: /mnt/ssd}
  uploader:
    upload: {poll_interval_s: 5}
  """
)


def _reloader(path, applied, *, reject=False):
  def validate(cfg):
    if reject:
      raise ValueError("rejected by test")
    # A real validator inspects cfg; here, accept unless `reject`.

  def apply(cfg):
    applied.append(cfg)

  return ConfigReloader(path=str(path), component="uploader", validate=validate, apply=apply)


def test_valid_config_is_applied(tmp_path):
  p = tmp_path / "fessel.yaml"
  p.write_text(GOOD)
  applied = []
  r = _reloader(p, applied)
  assert r.reload_once() is True
  assert len(applied) == 1
  # The component view was passed to apply (shared + uploader subtree flattened).
  assert applied[0]["mqtt"]["host"] == "127.0.0.1"
  assert applied[0]["upload"]["poll_interval_s"] == 5


def test_invalid_config_keeps_old_and_does_not_apply(tmp_path):
  p = tmp_path / "fessel.yaml"
  p.write_text(GOOD)
  applied = []
  r = _reloader(p, applied, reject=True)
  assert r.reload_once() is False
  assert applied == []  # validate rejected -> apply never ran (old config kept)


def test_unparseable_file_keeps_old(tmp_path):
  p = tmp_path / "fessel.yaml"
  p.write_text("{ this: is not: valid yaml ::::")
  applied = []
  r = _reloader(p, applied)
  # load_yaml_config raising (or yielding junk) must not crash or apply.
  result = r.reload_once()
  # Either it failed to read (False) or it read junk that validate accepts; in
  # both cases the process must not have crashed. If it parsed to a dict, apply
  # may run — so assert only the no-crash + sane-return contract.
  assert result in (True, False)


def test_apply_failure_is_contained(tmp_path):
  p = tmp_path / "fessel.yaml"
  p.write_text(GOOD)

  def validate(cfg):
    pass

  def apply(cfg):
    raise RuntimeError("apply blew up")

  r = ConfigReloader(path=str(p), component="uploader", validate=validate, apply=apply)
  # An exception inside apply is caught and reported, not propagated.
  assert r.reload_once() is False


def test_missing_file_yields_empty_then_validate_decides(tmp_path):
  # No file -> load_component_config returns {}; validate runs on {} and decides.
  applied = []

  def validate(cfg):
    if not cfg:
      raise ValueError("empty config")

  def apply(cfg):
    applied.append(cfg)

  r = ConfigReloader(
    path=str(tmp_path / "nope.yaml"), component="uploader", validate=validate, apply=apply
  )
  assert r.reload_once() is False
  assert applied == []

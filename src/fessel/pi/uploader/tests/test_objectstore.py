"""Object-store seam tests (V4.7): the fake driver + the S3 error classifier."""

from uploader import FakeObjectStore, build_object_store
from uploader.objectstore import PermanentError, RetryableError, _is_permanent_s3


def test_build_fake_driver():
  store = build_object_store({"driver": "fake"})
  assert isinstance(store, FakeObjectStore)


def test_fake_store_records_objects(tmp_path):
  f = tmp_path / "seg.ts"
  f.write_bytes(b"DATA")
  store = FakeObjectStore()
  store.put_object("recordings/explicit/r1/seg.ts", f, "video/mp2t")
  assert store.objects["recordings/explicit/r1/seg.ts"] == b"DATA"


def test_fake_store_failure_modes(tmp_path):
  f = tmp_path / "seg.ts"
  f.write_bytes(b"DATA")
  import pytest

  retr = FakeObjectStore(fail_n_then_ok=1)
  with pytest.raises(RetryableError):
    retr.put_object("k", f, "video/mp2t")
  # Second call succeeds.
  retr.put_object("k", f, "video/mp2t")
  assert "k" in retr.objects

  perm = FakeObjectStore(permanent=True)
  with pytest.raises(PermanentError):
    perm.put_object("k", f, "video/mp2t")


def test_s3_error_classifier():
  # Auth / not-found are permanent (no point retrying).
  assert _is_permanent_s3("AccessDenied", None) is True
  assert _is_permanent_s3("NoSuchBucket", None) is True
  assert _is_permanent_s3("", 403) is True
  # 5xx + throttling are retryable.
  assert _is_permanent_s3("", 500) is False
  assert _is_permanent_s3("", 503) is False
  assert _is_permanent_s3("", 429) is False
  assert _is_permanent_s3("SlowDown", None) is False

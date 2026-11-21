"""Tests for env-based settings and the source-checksum guard."""

import hashlib

import pytest

from eval_harness import db
from eval_harness.settings import env


def test_env_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("EH_TEST_SETTING", "from-env")
    assert env("EH_TEST_SETTING", "default") == "from-env"


def test_env_falls_back_when_unset_or_blank(monkeypatch):
    monkeypatch.delenv("EH_TEST_SETTING", raising=False)
    assert env("EH_TEST_SETTING", "default") == "default"
    monkeypatch.setenv("EH_TEST_SETTING", "")  # blank line copied from .env.example
    assert env("EH_TEST_SETTING", "default") == "default"


def test_download_verifies_checksum(tmp_path, monkeypatch):
    src = tmp_path / "dump.sql"
    src.write_bytes(b"CREATE TABLE t (a int);\n")
    url = src.as_uri()

    monkeypatch.setattr(db, "SOURCE_SHA256", hashlib.sha256(src.read_bytes()).hexdigest())
    out = db.download_source(dest=tmp_path / "raw" / "copy.sql", url=url)
    assert out.read_bytes() == src.read_bytes()

    monkeypatch.setattr(db, "SOURCE_SHA256", "0" * 64)  # override with a wrong hash
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        db.download_source(dest=tmp_path / "raw" / "other.sql", url=url)

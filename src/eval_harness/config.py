"""Load eval configs (``configs/*.yaml``) into ``Config`` objects."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from eval_harness.types import Config

CONFIG_FIELDS = {f.name for f in fields(Config)}


def load_config(path: Path | str) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - CONFIG_FIELDS
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    data.setdefault("id", path.stem)
    missing = {"model", "prompt_version"} - set(data)
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")
    return Config(**data)

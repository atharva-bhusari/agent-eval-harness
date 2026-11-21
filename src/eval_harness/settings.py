"""Environment-based settings.

Loads ``.env`` from the repo root once (real environment variables win over ``.env``), then
exposes ``env(name, default)``. Secrets (API keys) live only in ``.env``. Non-secret
settings keep committed defaults in code so CI and fresh clones work without a ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else Path.cwd()


load_dotenv(repo_root() / ".env", override=False)


def env(name: str, default: str) -> str:
    """Value of ``name`` from the environment or ``.env``; ``default`` if unset or empty."""
    return os.environ.get(name) or default

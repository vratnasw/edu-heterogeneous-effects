"""Tiny yaml config loader. No magic — just `load_config()`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def repo_root() -> Path:
    return _REPO_ROOT

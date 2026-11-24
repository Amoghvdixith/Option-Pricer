"""Project-relative paths and YAML configuration loading (no machine-specific paths)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def project_root() -> Path:
    """Locate the project root.

    Order: the ``OPTPRICER_ROOT`` environment variable, then the first parent of this file
    or of the current directory that contains both ``pyproject.toml`` and ``configs/``, then
    the current directory.
    """
    env = os.environ.get("OPTPRICER_ROOT")
    if env:
        return Path(env).resolve()
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for p in (start, *start.parents):
            if (p / "pyproject.toml").exists() and (p / "configs").is_dir():
                return p
    return Path.cwd().resolve()


def results_dir(sub: str = "") -> Path:
    p = project_root() / "results" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(sub: str = "") -> Path:
    p = project_root() / "data" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(name: str, mode: str = "fast") -> dict:
    """Load ``configs/<name>.yaml``. Top-level ``fast``/``research`` sections are merged over ``common``."""
    path = project_root() / "configs" / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if mode not in ("fast", "research"):
        raise ValueError("mode must be 'fast' or 'research'")
    cfg = dict(raw.get("common", {}))
    cfg.update(raw.get(mode, {}))
    cfg["_mode"] = mode
    cfg["_file"] = f"configs/{name}.yaml"
    return cfg

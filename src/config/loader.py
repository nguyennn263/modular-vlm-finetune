"""YAML config loading helpers."""
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repo root (src/config/loader.py -> src/config -> src -> root)."""
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return repo_root() / "configs"


def load_config(config_path: str | Path) -> dict:
    """Load a YAML file into a dict (empty dict if the file is empty)."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# Backwards-compatible alias.
load_yaml = load_config


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_configs(out[key], value)
        else:
            out[key] = value
    return out

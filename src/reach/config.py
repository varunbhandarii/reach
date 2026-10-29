"""Read shared defaults and optional machine-specific overrides."""

from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, changes: dict) -> dict:
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: Path | str | None = None, local_path: Path | str | None = None) -> dict:
    path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    local_path = Path(local_path) if local_path is not None else path.with_name("config.local.yaml")
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    if local_path.exists():
        with local_path.open(encoding="utf-8") as stream:
            local = yaml.safe_load(stream) or {}
        if not isinstance(local, dict):
            raise ValueError(f"Configuration must be a mapping: {local_path}")
        cfg = _merge(cfg, local)
    return cfg

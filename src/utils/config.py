"""YAML config loader with light-weight, attribute-style access.

We intentionally avoid pydantic to keep the dependency footprint small. All
experiment scripts call `load_config(path)` and then read fields off the
returned `Config`. Unknown keys are preserved (not validated) so configs can
evolve without code churn.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Matches ${VAR} and ${VAR:-default}. Used for shell-style env-var expansion in
# YAML string values (e.g. `api_key: ${TOGETHER_API_KEY}`).
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_in_string(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        return os.environ.get(name, default if default is not None else m.group(0))

    return _ENV_PATTERN.sub(repl, s)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_in_string(value)
    if isinstance(value, Mapping):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class Config(dict):
    """A dict that also exposes keys as attributes and supports nested access.

    Example:
        cfg = load_config("configs/audit.yaml")
        cfg.tokenizers          # list[str]
        cfg["data"]["dataset"]  # still works
        cfg.get("missing", "x") # still works
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return _wrap(value)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, Config):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


@dataclass
class ConfigSource:
    path: Path
    raw: dict[str, Any]


def load_config(path: str | Path) -> Config:
    """Load YAML; return a Config exposing dotted access.

    Raises FileNotFoundError with a clear message if the path is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    raw = _expand_env(raw)
    cfg = _wrap(raw)
    cfg["__source__"] = str(p.resolve())
    return cfg


def dump_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    """Persist a config dict back to YAML (used to snapshot a run's effective config)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plain = _to_plain(cfg)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(plain, f, sort_keys=False, allow_unicode=True)


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items() if k != "__source__"}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value

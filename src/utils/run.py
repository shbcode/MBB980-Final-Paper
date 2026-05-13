"""Run management: every experiment invocation gets a timestamped directory.

Each `RunContext` records:
    - effective config (config.yaml)
    - git commit hash if available (git_commit.txt)
    - a manifest.json with seed, model, tokenizer, start/stop times
    - a run.log file populated by the logging handler
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import dump_config
from .io import ensure_dir
from .logging import setup_logging


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except Exception:
        return None


@dataclass
class RunContext:
    """Per-run scratch directory + metadata. Use as a context-ish object:

        run = new_run("results/tokenization_audit", config=cfg, name="audit")
        run.save_artifact("per_sentence.csv", df)  # callers do their own save
        run.finalize(metrics={"n_pairs": 1012})
    """

    root: Path
    name: str
    config: Mapping[str, Any]
    started_at: float = field(default_factory=time.time)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs" / self.name

    def add_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at_utc": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "python": sys.version,
            "platform": platform.platform(),
            "argv": sys.argv,
            "env_seed": os.environ.get("PYTHONHASHSEED"),
            **self.extras,
        }

    def finalize(self, metrics: Mapping[str, Any] | None = None) -> Path:
        ensure_dir(self.runs_dir)
        dump_config(self.config, self.runs_dir / "config.yaml")
        m = self.manifest()
        m["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        m["elapsed_seconds"] = round(time.time() - self.started_at, 3)
        if metrics is not None:
            m["metrics"] = dict(metrics)
        (self.runs_dir / "manifest.json").write_text(
            json.dumps(m, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        commit = m.get("git_commit")
        if commit:
            (self.runs_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        return self.runs_dir


def new_run(
    root: str | Path,
    *,
    config: Mapping[str, Any],
    name: str | None = None,
    log_level: str = "INFO",
) -> RunContext:
    """Create a timestamped run directory under `<root>/runs/<name>` and start logging."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = (name or "run").replace("/", "_")
    final_name = f"{ts}__{safe}"
    run = RunContext(root=Path(root), name=final_name, config=config)
    ensure_dir(run.runs_dir)
    setup_logging(level=log_level, log_file=run.runs_dir / "run.log")
    return run

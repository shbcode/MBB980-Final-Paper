"""Backend factory.

Configs typically look like:
    backend:
      kind: hf_local
      model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
      dtype: bfloat16
or:
    backend:
      kind: openai
      model: gpt-4o-mini
or for offline:
    backend:
      kind: dummy

If the requested backend isn't available (missing extras or env vars), we fall
back to `dummy` and log a warning, so the rest of the pipeline still runs.
"""

from __future__ import annotations

from typing import Any

from ..utils.logging import get_logger
from .base import ModelBackend
from .dummy import DummyBackend

log = get_logger(__name__)


def warn_if_dummy(backend: ModelBackend, *, experiment: str) -> None:
    """Loud, single-shot warning when an experiment that needs real model
    behaviour is wired to the dummy backend.

    Use this from experiments where the dummy backend produces meaningless
    metrics (everything except the tokenization audit, basically).
    """
    if isinstance(backend, DummyBackend):
        log.warning(
            "Experiment %r is running against the DUMMY backend, which returns "
            "the same canned response for every prompt. Downstream metrics "
            "(accuracy, stability, regression coefficients) will be degenerate. "
            "Set `backend.kind` to `hf_local` or `openai` in the config for "
            "meaningful results.",
            experiment,
        )


def build_backend(spec: dict[str, Any] | None, *, allow_dummy_fallback: bool = True) -> ModelBackend:
    if not spec or spec.get("kind") in (None, "none", "dummy"):
        return DummyBackend(response=(spec or {}).get("response", "OK"))
    kind = spec["kind"]
    try:
        if kind == "hf_local":
            from .hf_local import HFLocalBackend

            return HFLocalBackend(
                model_id=spec["model_id"],
                revision=spec.get("revision"),
                device_map=spec.get("device_map", "auto"),
                dtype=spec.get("dtype", "bfloat16"),
                chat_format=spec.get("chat_format", True),
            )
        if kind in ("openai", "together"):
            from .openai_api import OpenAIBackend

            return OpenAIBackend(
                model=spec.get("model", "gpt-4o-mini"),
                api_key=spec.get("api_key"),
                base_url=spec.get("base_url"),
            )
    except Exception as e:
        if not allow_dummy_fallback:
            raise
        log.warning("Backend '%s' unavailable (%s); falling back to dummy.", kind, e)
        return DummyBackend()
    raise ValueError(f"Unknown backend kind: {kind!r}")

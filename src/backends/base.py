"""Common interface for model backends.

Backends accept a single `GenerationRequest` and return a `GenerationResponse`.
Pricing fields are optional; only `openai`/`together` populate them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    seed: int | None = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float | None = None
    cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ModelBackend(ABC):
    """Minimum surface every backend must expose."""

    name: str

    @abstractmethod
    def generate(self, req: GenerationRequest) -> GenerationResponse: ...

    def generate_batch(self, reqs: list[GenerationRequest]) -> list[GenerationResponse]:
        # Default: serial. Override for batched/parallel backends.
        return [self.generate(r) for r in reqs]

    def close(self) -> None:  # noqa: B027 - intentional default
        return None

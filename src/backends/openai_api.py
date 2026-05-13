"""OpenAI Chat Completions backend (also reusable for OpenAI-compatible endpoints
such as Together if you set base_url + api_key accordingly).
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import GenerationRequest, GenerationResponse, ModelBackend

# Conservative cost table (USD per 1M tokens). Override via constructor if you
# care about exact billing; these are only used for cost-per-correct heuristics.
DEFAULT_PRICES_PER_M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o":      (2.50, 10.00),
}


class OpenAIBackend(ModelBackend):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        prices_per_m: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIBackend requires the 'api' extra: pip install '.[api]'"
            ) from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; pass api_key=... or set the env var."
            )
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.prices = prices_per_m or DEFAULT_PRICES_PER_M

    def _cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        rate = self.prices.get(self.model)
        if rate is None or prompt_tokens is None or completion_tokens is None:
            return None
        in_rate, out_rate = rate
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "user", "content": req.prompt}],
            temperature=req.temperature,
            max_tokens=req.max_new_tokens,
            top_p=req.top_p,
        )
        if req.stop:
            kwargs["stop"] = req.stop
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", None) if usage else None
        ct = getattr(usage, "completion_tokens", None) if usage else None
        return GenerationResponse(
            text=choice.message.content or "",
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_seconds=time.perf_counter() - t0,
            cost_usd=self._cost(pt, ct),
            raw={"backend": "openai", "model": self.model, "id": resp.id},
        )

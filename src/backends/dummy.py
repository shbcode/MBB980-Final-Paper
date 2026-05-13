"""Echo / deterministic backend for offline testing of the pipeline plumbing.

Returns a canned response; useful in CI and when API keys aren't set.
"""

from __future__ import annotations

import time

from .base import GenerationRequest, GenerationResponse, ModelBackend


class DummyBackend(ModelBackend):
    name = "dummy"

    def __init__(self, response: str = "OK") -> None:
        self._response = response

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        t0 = time.perf_counter()
        text = self._response
        return GenerationResponse(
            text=text,
            prompt_tokens=len(req.prompt.split()),
            completion_tokens=len(text.split()),
            latency_seconds=time.perf_counter() - t0,
            raw={"backend": "dummy"},
        )

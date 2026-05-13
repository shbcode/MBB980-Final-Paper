"""HuggingFace `transformers` local backend for instruction-tuned decoder-only LLMs.

Lazy imports keep the core pipeline runnable without torch installed.
"""

from __future__ import annotations

import time
from typing import Any

from .base import GenerationRequest, GenerationResponse, ModelBackend


class HFLocalBackend(ModelBackend):
    """Wrap a local HF causal LM. Supports greedy and sampled generation."""

    name = "hf_local"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device_map: str | None = "auto",
        dtype: str = "bfloat16",
        chat_format: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HFLocalBackend requires the 'hf' extra: pip install '.[hf]'"
            ) from e

        self.model_id = model_id
        self.chat_format = chat_format
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()
        self._torch = torch

    def _format_prompt(self, prompt: str) -> str:
        if not self.chat_format or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        msgs = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        torch = self._torch
        t0 = time.perf_counter()
        text = self._format_prompt(req.prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=req.max_new_tokens,
            do_sample=req.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if req.temperature > 0:
            gen_kwargs["temperature"] = req.temperature
            gen_kwargs["top_p"] = req.top_p
        if req.seed is not None:
            torch.manual_seed(req.seed)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, prompt_len:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResponse(
            text=completion,
            prompt_tokens=int(prompt_len),
            completion_tokens=int(gen_ids.shape[0]),
            latency_seconds=time.perf_counter() - t0,
            raw={"backend": "hf_local", "model_id": self.model_id},
        )

    def close(self) -> None:
        try:
            del self.model
            self._torch.cuda.empty_cache()
        except Exception:
            pass

"""LLM responder abstractions for local and hosted models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch

from src.token_self_repair.llm import LlamaProvider, load_llama


@dataclass(slots=True)
class ModelResponse:
    text: str
    tokens: List[str]
    logits: Optional[np.ndarray]
    metadata: dict


@lru_cache(maxsize=2)
def _load_llama_provider(model_name: str, quantize: bool) -> LlamaProvider:
    return load_llama(model_name=model_name, quantize=quantize)


class LocalLlamaResponder:
    """Generate answers using the local Llama provider with logits."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", quantize: bool = False):
        self.model_name = model_name
        self.quantize = quantize
        self.provider = _load_llama_provider(model_name, quantize)

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.1) -> ModelResponse:
        do_sample = temperature > 0.2
        token_ids, logits_tensor = self.provider.generate_with_logits(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        tokens = [
            self.provider.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in token_ids
        ]
        text = self.provider.tokenizer.decode(token_ids, skip_special_tokens=False)
        logits = logits_tensor.cpu().numpy()

        return ModelResponse(
            text=text,
            tokens=tokens,
            logits=logits,
            metadata={
                "model": self.model_name,
                "quantized": self.quantize,
            },
        )


class OpenAIResponder:
    """Generate answers via OpenAI Chat Completions API with optional logprobs."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAIResponder.") from exc

        self.client = OpenAI()
        self.model_name = model_name

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.1) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
        choice = response.choices[0]
        text = choice.message.content or ""

        tokens: List[str] = []
        logprobs = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                if token_info.token is None:
                    continue
                tokens.append(token_info.token)
                logprobs.append(float(token_info.logprob))

        logits: Optional[np.ndarray] = None
        if logprobs:
            probs = np.exp(np.array(logprobs))
            probs = np.clip(probs, 1e-9, 1.0)
            logits = np.log(probs / (1 - probs + 1e-9)).reshape(-1, 1)

        return ModelResponse(
            text=text,
            tokens=tokens,
            logits=logits,
            metadata={
                "model": self.model_name,
                "openai": True,
            },
        )



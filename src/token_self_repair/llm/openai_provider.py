"""OpenAI provider that surfaces token-level logprobs for uncertainty analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, TYPE_CHECKING

try:  # pragma: no cover - import guard exercised implicitly
    from openai import OpenAI
except ImportError:  # pragma: no cover - defer failure until used
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR: Optional[Exception] = ImportError(
        "The 'openai' package is required to use OpenAIProvider. Install it via 'pip install openai'."
    )
else:
    _OPENAI_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI as _TClient

from .base import LLMClient, TokenLogit


def _ensure_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or pass api_key=... to OpenAIProvider."
        )
    return key


@dataclass(slots=True)
class OpenAIProvider(LLMClient):
    """LLM client that queries the OpenAI Chat Completions API with logprobs enabled.

    The provider converts the returned logprob entries into the TokenLogit format used by
    the uncertainty pipeline. Because OpenAI only returns top-k logprobs, we keep a
    configurable vector size and pad with very low logit values when necessary so that
    downstream numpy stacking works consistently.
    """

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None
    top_logprobs: int = 5
    client: Optional["_TClient"] = None

    def __post_init__(self) -> None:
        self._logit_size = max(2, self.top_logprobs)
        if self.client is None:
            if _OPENAI_IMPORT_ERROR is not None or OpenAI is None:  # pragma: no cover - guard
                raise RuntimeError(_OPENAI_IMPORT_ERROR or ImportError("openai package unavailable"))
            key = _ensure_api_key(self.api_key)
            self.client = OpenAI(api_key=key)

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        **kwargs,
    ) -> Iterable[TokenLogit]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self._logit_size,
            **kwargs,
        )

        choice = completion.choices[0]
        logprob_content = getattr(choice.logprobs or {}, "content", None)
        if not logprob_content:
            raise RuntimeError(
                "OpenAI response did not include logprob content. Ensure the selected "
                "model supports logprobs=True."
            )

        for token_info in logprob_content:
            token = getattr(token_info, "token", "") or ""
            logits = self._vector_from_token_info(token_info)
            yield TokenLogit(token=token, logits=logits)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _vector_from_token_info(self, token_info) -> List[float]:
        """Convert a logprob entry into a fixed-width pseudo-logit vector."""

        vector: List[float] = []
        primary = getattr(token_info, "logprob", None)
        if primary is not None:
            vector.append(float(primary))

        top_alternatives = getattr(token_info, "top_logprobs", None) or []
        for alt in top_alternatives:
            if len(vector) >= self._logit_size:
                break
            alt_logprob = getattr(alt, "logprob", None)
            if alt_logprob is None:
                continue
            vector.append(float(alt_logprob))

        # Ensure at least _logit_size entries so downstream np.stack succeeds.
        while len(vector) < self._logit_size:
            vector.append(-100.0)

        return vector[: self._logit_size]

    def __repr__(self) -> str:  # pragma: no cover - simple metadata
        return (
            "OpenAIProvider("
            f"model={self.model}, "
            f"logit_size={self._logit_size}, "
            f"system_prompt={'set' if self.system_prompt else 'unset'}"
            ")"
        )

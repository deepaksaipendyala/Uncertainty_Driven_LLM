"""Reasoning pipeline with uncertainty aggregation for step-level analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import ProjectConfig
from ..llm.base import LLMClient
from ..messaging.status import StatusMessenger
from ..repair.base import SelfRepairStrategy
from ..types import GenerationStep
from ..uncertainty import (
    LogTokUEstimator,
    UncertaintyAggregator,
    UncertaintyMap,
    UncertaintyScores,
)
from ..uncertainty.base import UncertaintyEstimator
from .base import PipelineResult, UncertaintyAwarePipeline


@dataclass(slots=True)
class ReasoningResult:
    """Holds the pipeline output plus aggregated uncertainty insights."""

    pipeline_result: PipelineResult
    uncertainty_map: Optional[UncertaintyMap]
    summary: Optional[str]


@dataclass(slots=True)
class ReasoningCoordinator:
    """Runs a reasoning-focused pipeline with step-level uncertainty summaries."""

    llm: LLMClient
    estimator: UncertaintyEstimator
    strategies: List[SelfRepairStrategy] = field(default_factory=list)
    aggregator: UncertaintyAggregator = field(default_factory=UncertaintyAggregator)
    messenger: StatusMessenger = field(default_factory=StatusMessenger)
    config: ProjectConfig = field(default_factory=ProjectConfig)
    pipeline: UncertaintyAwarePipeline = field(init=False)

    def __post_init__(self) -> None:
        self.pipeline = UncertaintyAwarePipeline(
            llm=self.llm,
            estimator=self.estimator,
            strategies=self.strategies,
            messenger=self.messenger,
            config=self.config,
        )

    def solve(self, question: str, *, max_tokens: int = 256) -> ReasoningResult:
        """Solve a reasoning task and aggregate uncertainty signals."""

        pipeline_result = self.pipeline.run(question, max_tokens=max_tokens)
        u_map = self._build_uncertainty_map(pipeline_result.step)
        summary = self._summarize(u_map) if u_map else None
        return ReasoningResult(pipeline_result=pipeline_result, uncertainty_map=u_map, summary=summary)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_uncertainty_map(self, step: GenerationStep) -> Optional[UncertaintyMap]:
        if not step.token_scores:
            return None

        tokens = step.generated_tokens or []
        scores = UncertaintyScores(
            eu=np.array([score.epistemic for score in step.token_scores], dtype=np.float32),
            au=np.array([score.aleatoric for score in step.token_scores], dtype=np.float32),
            total=np.array([score.total_uncertainty for score in step.token_scores], dtype=np.float32),
            entropy=np.array([score.entropy for score in step.token_scores], dtype=np.float32),
            token_texts=tokens,
        )

        # Reconstruct response text; join without extra spaces to retain punctuation/newlines.
        response_text = "".join(tokens).strip()

        return self.aggregator.build_uncertainty_map(
            scores=scores,
            source_text=response_text,
            language="text",
            tokens=tokens,
        )

    def _summarize(self, u_map: UncertaintyMap, top_k: int = 3) -> str:
        if not u_map.line_scores:
            return "Uncertainty evenly distributed; no dominant hotspots identified."

        lines = []
        for hotspot in u_map.hotspots:
            if hotspot.kind != "line":
                continue
            line_no = int(hotspot.identifier)
            line = u_map.line_scores.get(line_no)
            if not line:
                continue
            snippet = line.text.strip()
            if not snippet:
                snippet = "(blank line)"
            lines.append(f"Line {line_no}: U={line.total:.2f} :: {snippet}")
            if len(lines) == top_k:
                break

        if not lines:
            return "Uncertainty hotspots dominated by method-level signals."

        return "Top uncertain reasoning steps:\n" + "\n".join(lines)


def default_reasoning_coordinator(model: LLMClient) -> ReasoningCoordinator:
    """Convenience helper for simple reasoning workflows."""

    estimator = LogTokUEstimator()
    return ReasoningCoordinator(llm=model, estimator=estimator)



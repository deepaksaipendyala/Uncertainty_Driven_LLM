"""Evaluation utilities for reasoning-centric uncertainty workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .datasets import dataset_registry
from .metrics import exact_match, expected_calibration_error
from ..pipelines.reasoning import ReasoningCoordinator, ReasoningResult


@dataclass(slots=True)
class ReasoningSampleResult:
    prompt: str
    reference: str
    prediction: str
    final_uncertainty: float
    correct: bool
    summary: str | None
    hotspots: List[tuple[str, str, float]]


@dataclass(slots=True)
class ReasoningBenchmarkResult:
    benchmark: str
    accuracy: float
    calibration_error: float
    average_uncertainty: float
    samples: List[ReasoningSampleResult]


class ReasoningEvaluationRunner:
    """Runs reasoning coordinators over benchmark datasets."""

    def __init__(self, coordinator_factory: Callable[[], ReasoningCoordinator]) -> None:
        self._factory = coordinator_factory

    def run(self, benchmark_name: str, *, max_samples: Optional[int] = None) -> ReasoningBenchmarkResult:
        registry = dataset_registry()
        if benchmark_name not in registry:
            raise KeyError(f"Benchmark '{benchmark_name}' not registered.")
        benchmark = registry[benchmark_name]

        predictions: List[str] = []
        references: List[str] = []
        uncertainties: List[float] = []
        sample_results: List[ReasoningSampleResult] = []

        for idx, sample in enumerate(benchmark.samples):
            if max_samples is not None and idx >= max_samples:
                break
            coordinator = self._factory()
            reasoning_result: ReasoningResult = coordinator.solve(sample.prompt)

            tokens = reasoning_result.pipeline_result.step.generated_tokens
            prediction = "".join(tokens).strip()
            if not prediction:
                prediction = ""  # ensure string
            predictions.append(prediction)
            references.append(sample.reference)

            final_uncertainty = (
                reasoning_result.pipeline_result.step.token_scores[-1].total_uncertainty
                if reasoning_result.pipeline_result.step.token_scores
                else 1.0
            )
            uncertainties.append(final_uncertainty)

            hotspots: List[tuple[str, str, float]] = []
            if reasoning_result.uncertainty_map:
                for hotspot in reasoning_result.uncertainty_map.hotspots[:5]:
                    identifier = hotspot.identifier
                    if hotspot.kind == "line":
                        identifier = f"line {identifier}"
                    elif hotspot.kind == "method":
                        identifier = f"method {identifier}"
                    hotspots.append((hotspot.kind, identifier, float(hotspot.score)))

            sample_results.append(
                ReasoningSampleResult(
                    prompt=sample.prompt,
                    reference=sample.reference,
                    prediction=prediction,
                    final_uncertainty=final_uncertainty,
                    correct=prediction == sample.reference,
                    summary=reasoning_result.summary,
                    hotspots=hotspots,
                )
            )

        if predictions:
            accuracy = exact_match(predictions, references)
            labels = np.array([int(p == r) for p, r in zip(predictions, references)], dtype=float)
            confidences = 1.0 - np.array(uncertainties)
            calibration_error = expected_calibration_error(confidences, labels)
            avg_uncertainty = float(np.mean(uncertainties))
        else:
            accuracy = 0.0
            calibration_error = 0.0
            avg_uncertainty = 1.0

        return ReasoningBenchmarkResult(
            benchmark=benchmark.name,
            accuracy=float(accuracy),
            calibration_error=float(calibration_error),
            average_uncertainty=avg_uncertainty,
            samples=sample_results,
        )



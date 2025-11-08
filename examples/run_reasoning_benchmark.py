"""Evaluate the reasoning pipeline on the GSM8K-lite benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.token_self_repair.llm import load_llama
from src.token_self_repair.pipelines import default_reasoning_coordinator
from src.token_self_repair.evaluation import ReasoningEvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reasoning benchmark with uncertainty analysis")
    parser.add_argument(
        "--benchmark",
        default="gsm8k",
        help="Benchmark name from dataset registry (default: gsm8k)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model identifier",
    )
    parser.add_argument("--quantize", action="store_true", help="Load model in 4-bit mode")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per generation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of samples to evaluate (default: 2)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = load_llama(model_name=args.model, quantize=args.quantize)

    def factory():
        coordinator = default_reasoning_coordinator(llm)
        coordinator.config.max_self_repairs = 0  # single pass for benchmarking
        return coordinator

    runner = ReasoningEvaluationRunner(coordinator_factory=factory)
    result = runner.run(args.benchmark, max_samples=args.limit)

    print(f"\nBenchmark: {result.benchmark}")
    print(f"Accuracy: {result.accuracy:.2f}")
    print(f"Average uncertainty: {result.average_uncertainty:.3f}")
    print(f"Expected calibration error: {result.calibration_error:.3f}\n")

    for idx, sample in enumerate(result.samples, start=1):
        print(f"Sample {idx}")
        print("Prompt:", sample.prompt)
        print("Reference:", sample.reference)
        print("Prediction:", sample.prediction)
        print(f"Final uncertainty: {sample.final_uncertainty:.3f} | Correct: {sample.correct}")
        if sample.summary:
            print("Summary:\n" + sample.summary)
        if sample.hotspots:
            print("Hotspots:")
            for kind, identifier, score in sample.hotspots:
                print(f"  [{kind}] {identifier} -> {score:.3f}")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



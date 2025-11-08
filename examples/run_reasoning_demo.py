"""Run the reasoning coordinator on a sample GSM8K-style question."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.token_self_repair.llm import load_llama
from src.token_self_repair.pipelines import default_reasoning_coordinator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run uncertainty-aware reasoning demo")
    parser.add_argument(
        "--question",
        default="A train travels 60 miles per hour for 2 hours. How far does it go?",
        help="Question to solve",
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens",
    )
    parser.add_argument("--quantize", action="store_true", help="Load model in 4-bit mode")

    args = parser.parse_args()

    print("Loading model...")
    llm = load_llama(model_name=args.model, quantize=args.quantize)
    coordinator = default_reasoning_coordinator(llm)

    print("\nQuestion:")
    print(args.question)
    print("\nGenerating answer with uncertainty monitoring...\n")

    result = coordinator.solve(args.question, max_tokens=args.max_tokens)
    answer = "".join(result.pipeline_result.step.generated_tokens).strip()
    print("Answer:")
    print(answer or "(no output)")

    final_score = (
        result.pipeline_result.step.token_scores[-1].total_uncertainty
        if result.pipeline_result.step.token_scores
        else 1.0
    )
    print(f"\nFinal token uncertainty: {final_score:.3f}")

    if result.summary:
        print("\nUncertainty summary:")
        print(result.summary)

    if result.uncertainty_map:
        print("\nTop hotspots:")
        for hotspot in result.uncertainty_map.hotspots[:5]:
            print(
                f"  [{hotspot.kind}] {hotspot.identifier} -> {hotspot.score:.3f}" +
                (f" ({'; '.join(f'{k}={v}' for k, v in hotspot.metadata.items() if v)})" if hotspot.metadata else "")
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Example: Testing the refactored LogTokU integration.

This script now supports both Hugging Face Llama models (full LogTokU workflow) and
OpenAI Chat Completions with logprobs enabled, so you can compare hosted and local
models with identical uncertainty analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.token_self_repair.llm import LlamaProvider, OpenAIProvider, load_llama
from src.token_self_repair.uncertainty import (
    LogTokUEstimator,
    UncertaintyAggregator,
    analyze_generation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test refactored LogTokU integration")
    parser.add_argument("--query", type=str, default="What is 2+2?", help="Query to test")
    parser.add_argument(
        "--backend",
        choices=["llama", "openai"],
        default="llama",
        help="Which provider to call (default: llama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model identifier for Llama",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to call when backend=openai",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=5,
        help="Number of OpenAI top logprobs captured per token",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LogTokU Integration Test")
    print("=" * 60)

    if args.backend == "openai":
        return run_openai_demo(args)

    return run_llama_demo(args)


# ---------------------------------------------------------------------------
# Hugging Face Llama flow (unchanged behaviour)
# ---------------------------------------------------------------------------

def run_llama_demo(args) -> int:
    print("\nMethod 1: Using convenience function")
    print("-" * 60)
    try:
        provider = load_llama(args.model, quantize=args.quantize)
        print(f"Loaded model: {args.model}")
        print(provider)

        result = analyze_generation(provider, args.query, max_tokens=args.max_tokens)
        if result:
            _display_scores(
                query=result["prompt"],
                response=result["response"],
                tokens=result["tokens"],
                scores=result["scores"],
                language="python",
            )
    except Exception as exc:  # pragma: no cover - exercised via manual runs
        print(f"Error: {exc}")
        return 1

    print("\n" + "=" * 60)
    print("Method 2: Step-by-step analysis")
    print("-" * 60)

    try:
        provider = LlamaProvider(args.model, use_quantization=args.quantize)
        print("✓ Model loaded")

        tokens, logits = provider.generate_with_logits(args.query, max_new_tokens=args.max_tokens)
        print(f"✓ Generated {len(tokens)} tokens")
        print(f"✓ Logits shape: {logits.shape}")

        estimator = LogTokUEstimator(k=2)
        scores = estimator.analyze(logits)
        print("✓ Uncertainty analyzed")

        if scores:
            print("\nResults:")
            print(f"  EU range:  [{scores.eu.min():.4f}, {scores.eu.max():.4f}]")
            print(f"  AU range:  [{scores.au.min():.4f}, {scores.au.max():.4f}]")
            print(f"  Entropy range: [{scores.entropy.min():.4f}, {scores.entropy.max():.4f}]")

            for i in range(min(10, len(scores.eu))):
                u_type = estimator.get_uncertainty_type(scores.eu[i], scores.au[i])
                print(f"  Token {i}: {u_type} uncertainty")

            token_ids_list = tokens.tolist()
            token_texts = [
                provider.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in token_ids_list
            ]
            _print_hotspots(
                scores,
                tokens=token_texts,
                response=provider.tokenizer.decode(token_ids_list, skip_special_tokens=True),
                language="python",
            )
    except Exception as exc:  # pragma: no cover - exercised via manual runs
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# OpenAI backend flow
# ---------------------------------------------------------------------------

def run_openai_demo(args) -> int:
    print("\nBackend: OpenAI Chat Completions")
    provider = OpenAIProvider(
        model=args.openai_model,
        top_logprobs=args.top_logprobs,
    )

    tokens: List[str] = []
    logit_rows: List[List[float]] = []
    try:
        for token_logit in provider.generate(args.query, max_tokens=args.max_tokens):
            tokens.append(token_logit.token)
            logit_rows.append(list(token_logit.logits))
    except Exception as exc:  # pragma: no cover - exercised via manual runs
        print(f"Error calling OpenAI: {exc}")
        return 1

    if not tokens:
        print("No tokens received from OpenAI response.")
        return 1

    response = "".join(tokens).strip()
    logits_array = np.asarray(logit_rows, dtype=np.float32)
    estimator = LogTokUEstimator(k=min(2, logits_array.shape[1]))
    scores = estimator.analyze(logits_array, token_texts=tokens)
    if scores is None:
        print("Unable to compute uncertainty scores for OpenAI response.")
        return 1

    _display_scores(
        query=args.query,
        response=response,
        tokens=tokens,
        scores=scores,
        language="text",
    )
    return 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _display_scores(query: str, response: str, tokens: List[str], scores, language: str) -> None:
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    _print_summary(scores)
    _print_hotspots(scores, tokens, response, language=language)


def _print_summary(scores) -> None:
    print("\nUncertainty Metrics:")
    print(f"  Average EU (Epistemic):  {scores.avg_eu:.4f}")
    print(f"  Average AU (Aleatoric):  {scores.avg_au:.4f}")
    print(f"  Average LogTokU:         {scores.avg_total:.4f}")
    print(f"  Average Entropy:         {scores.avg_entropy:.4f}")
    print(f"  Maximum LogTokU:         {scores.max_total:.4f}")
    print(f"  Tokens analyzed:         {len(scores.eu)}")

    top_indices = scores.get_top_uncertain_indices(k=min(5, len(scores.total)))
    print("\nTop 5 Most Uncertain Tokens:")
    for rank, idx in enumerate(top_indices, 1):
        token = scores.token_texts[idx] if scores.token_texts else str(idx)
        print(
            f"  {rank}. Token {idx}: '{token}' (LogTokU: {scores.total[idx]:.4f}, "
            f"EU: {scores.eu[idx]:.4f}, AU: {scores.au[idx]:.4f})"
        )


def _print_hotspots(scores, tokens: List[str], response: str, language: str = "text") -> None:
    aggregator = UncertaintyAggregator()
    u_map = aggregator.build_uncertainty_map(
        scores=scores,
        source_text=response,
        language=language,
        tokens=tokens,
    )
    print("\nTop uncertainty hotspots (lines/methods):")
    for hotspot in u_map.hotspots[:5]:
        meta = ", ".join(f"{k}={v}" for k, v in hotspot.metadata.items() if v)
        print(
            f"  [{hotspot.kind}] {hotspot.identifier} -> score={hotspot.score:.4f}"
            + (f" ({meta})" if meta else "")
        )


if __name__ == "__main__":
    sys.exit(main())

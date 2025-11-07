"""
Example: Testing the refactored LogTokU integration.

This script demonstrates how to use the refactored LlamaProvider and LogTokUEstimator
to analyze uncertainty in generated text.

Usage:
    python examples/test_logtoku_integration.py --query "What is 2+2?" --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.token_self_repair.llm import LlamaProvider, load_llama
from src.token_self_repair.uncertainty import (
    LogTokUEstimator,
    UncertaintyAggregator,
    analyze_generation,
)


def main():
    parser = argparse.ArgumentParser(description="Test refactored LogTokU integration")
    parser.add_argument(
        "--query",
        type=str,
        default="What is 2+2?",
        help="Query to test"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name (default: TinyLlama)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("LogTokU Integration Test")
    print("="*60)
    
    # Method 1: Using convenience function
    print("\nMethod 1: Using convenience function")
    print("-"*60)
    
    try:
        provider = load_llama(args.model, quantize=args.quantize)
        print(f"Loaded model: {args.model}")
        print(provider)
        
        # Quick analysis
        result = analyze_generation(provider, args.query, max_tokens=args.max_tokens)
        
        if result:
            print(f"\nQuery: {result['prompt']}")
            print(f"Response: {result['response']}")
            print("\nUncertainty Metrics:")
            print(f"  Average EU (Epistemic):  {result['avg_eu']:.4f}")
            print(f"  Average AU (Aleatoric):  {result['avg_au']:.4f}")
            print(f"  Average LogTokU:         {result['avg_logtoku']:.4f}")
            print(f"  Average Entropy:         {result['avg_entropy']:.4f}")
            print(f"  Maximum LogTokU:         {result['max_logtoku']:.4f}")
            print(f"  Tokens analyzed:         {result['num_tokens']}")
            
            # Show most uncertain tokens
            scores = result['scores']
            top_indices = scores.get_top_uncertain_indices(k=5)
            print("\nTop 5 Most Uncertain Tokens:")
            for rank, idx in enumerate(top_indices, 1):
                token = result['tokens'][idx] if idx < len(result['tokens']) else "N/A"
                print(f"  {rank}. Token {idx}: '{token}' (LogTokU: {scores.total[idx]:.4f}, EU: {scores.eu[idx]:.4f}, AU: {scores.au[idx]:.4f})")

            aggregator = UncertaintyAggregator()
            u_map = aggregator.build_uncertainty_map(
                scores=scores,
                source_text=result['response'],
                language="python",
                tokens=result['tokens'],
            )

            print("\nTop uncertainty hotspots (lines/methods):")
            for hotspot in u_map.hotspots[:5]:
                meta = ", ".join(f"{k}={v}" for k, v in hotspot.metadata.items() if v)
                print(f"  [{hotspot.kind}] {hotspot.identifier} -> score={hotspot.score:.4f} ({meta})")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    # Method 2: Manual step-by-step
    print("\n" + "="*60)
    print("Method 2: Step-by-step analysis")
    print("-"*60)
    
    try:
        # Step 1: Load model
        provider = LlamaProvider(args.model, use_quantization=args.quantize)
        print(f"✓ Model loaded")
        
        # Step 2: Generate with logits
        tokens, logits = provider.generate_with_logits(args.query, max_new_tokens=args.max_tokens)
        print(f"✓ Generated {len(tokens)} tokens")
        print(f"✓ Logits shape: {logits.shape}")
        
        # Step 3: Analyze uncertainty
        estimator = LogTokUEstimator(k=2)
        scores = estimator.analyze(logits)
        print(f"✓ Uncertainty analyzed")
        
        # Step 4: Display results
        if scores:
            print(f"\nResults:")
            print(f"  EU range:  [{scores.eu.min():.4f}, {scores.eu.max():.4f}]")
            print(f"  AU range:  [{scores.au.min():.4f}, {scores.au.max():.4f}]")
            print(f"  Entropy range: [{scores.entropy.min():.4f}, {scores.entropy.max():.4f}]")
            
            # Classify uncertainty types
            for i in range(min(10, len(scores.eu))):
                u_type = estimator.get_uncertainty_type(scores.eu[i], scores.au[i])
                print(f"  Token {i}: {u_type} uncertainty")

            aggregator = UncertaintyAggregator()
            token_ids_list = tokens.tolist()
            token_texts = [
                provider.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in token_ids_list
            ]
            aggregate = aggregator.build_uncertainty_map(
                scores=scores,
                source_text=provider.tokenizer.decode(token_ids_list, skip_special_tokens=True),
                language="python",
                tokens=token_texts,
            )
            print(f"  Aggregated line count: {len(aggregate.line_scores)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


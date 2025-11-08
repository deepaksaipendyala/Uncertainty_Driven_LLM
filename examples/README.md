# Examples

This directory contains example scripts demonstrating how to use the Token-Level Uncertainty Driven Self-Repair framework.

## Available Examples

### 1. test_logtoku_integration.py

Tests the refactored LogTokU integration with Llama models.

**Basic Usage:**
```bash
python examples/test_logtoku_integration.py --query "What is 2+2?"
```

**With specific model:**
```bash
python examples/test_logtoku_integration.py \
    --query "Explain photosynthesis" \
    --model meta-llama/Llama-2-7b-chat-hf \
    --max_tokens 100
```

**With quantization (GPU only):**
```bash
python examples/test_logtoku_integration.py \
    --query "Write a Python function" \
    --model meta-llama/Llama-3-8B-Instruct \
    --quantize
```

**What it demonstrates:**
- Loading Llama models with LlamaProvider
- Generating text with logit extraction
- Calculating LogTokU uncertainty (EU + AU)
- Identifying most uncertain tokens
- Two usage patterns: convenience function and step-by-step

**Expected Output:**
```
LogTokU Integration Test
============================================================

Method 1: Using convenience function
------------------------------------------------------------
Loaded model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Query: What is 2+2?
Response: 2 + 2 = 4

Uncertainty Metrics:
  Average EU (Epistemic):  0.3245
  Average AU (Aleatoric):  0.4156
  Average LogTokU:         0.1349
  Average Entropy:         2.3456
  Maximum LogTokU:         0.2891
  Tokens analyzed:         15

Top 5 Most Uncertain Tokens:
  1. Token 5: '?' (LogTokU: 0.2891, EU: 0.5234, AU: 0.5523)
  2. Token 8: '=' (LogTokU: 0.2103, EU: 0.4521, AU: 0.4650)
  ...
```

---

### 2. run_reasoning_demo.py

Runs the reasoning coordinator on a GSM8K-style question and prints uncertainty hotspots.

```bash
python examples/run_reasoning_demo.py --question "A train travels 60 mph for 2 hours. How far?"
```

Optional parameters:

```bash
python examples/run_reasoning_demo.py \
    --question "If you have 8 apples and give away 3, how many remain?" \
    --model meta-llama/Llama-2-7b-chat-hf \
    --max-tokens 150 \
    --quantize
```

Output includes:
- Final answer assembled from generated tokens
- Final token-uncertainty score
- Top uncertain reasoning lines (via `UncertaintyAggregator`)
- Overall hotspot list (line/method IDs with scores)

### 3. run_reasoning_benchmark.py

Runs the reasoning coordinator across the GSM8K-lite benchmark and reports
accuracy, calibration, and hotspot summaries per sample.

```bash
python examples/run_reasoning_benchmark.py \
    --benchmark gsm8k \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --max-tokens 256 \
    --limit 2
```

Use `--limit` to control how many benchmark samples to run (default: 2). Set
`--quantize` to enable 4-bit loading when using consumer GPUs.

Output includes:
- Benchmark-level accuracy and calibration error
- Per-sample predictions with uncertainty summaries
- Hotspot listings for the top uncertain reasoning steps

---

## Prerequisites

### Models

For testing, use TinyLlama (no authentication required):
```bash
# Automatically downloads on first use
python examples/test_logtoku_integration.py
```

For Meta Llama models, you need Hugging Face access:
```bash
# 1. Request access at https://huggingface.co/meta-llama/
# 2. Login with your token
huggingface-cli login

# 3. Run example
python examples/test_logtoku_integration.py --model meta-llama/Llama-2-7b-chat-hf
```

### Hardware

**Minimum:**
- CPU: Any modern CPU
- RAM: 8GB
- Model: TinyLlama-1.1B

**Recommended:**
- GPU: NVIDIA with 16GB+ VRAM
- RAM: 32GB
- Model: Llama-2-7B or Llama-3-8B

**With Quantization:**
- GPU: NVIDIA with 8GB+ VRAM
- RAM: 16GB
- Model: Any Llama model (4-bit quantized)

---

## Troubleshooting

### Error: "No module named 'token_self_repair'"

Make sure you've installed the package:
```bash
cd /home/dpendya/Documents/dlba
pip install -e .
```

### Error: "Could not load model"

For Meta Llama models:
1. Ensure you've requested access at https://huggingface.co/meta-llama/
2. Login: `huggingface-cli login`
3. Wait for access approval (can take a few hours)

Or use TinyLlama instead (no authentication needed).

### Error: "CUDA out of memory"

Try these solutions:
1. Use quantization: `--quantize`
2. Reduce tokens: `--max_tokens 20`
3. Use smaller model: `--model TinyLlama/TinyLlama-1.1B-Chat-v1.0`
4. Use CPU: (automatic if no GPU available)

### Slow generation on CPU

Expected behavior. To speed up:
1. Use GPU if available
2. Reduce max_tokens
3. Use smaller model (TinyLlama)

---

## Development

### Adding New Examples

1. Create new script in `examples/`
2. Follow the naming pattern: `test_<feature>.py`
3. Add usage instructions to this README
4. Include docstring with usage examples
5. Add to CI/CD tests if applicable

### Example Template

```python
"""
Example: <Feature Name>

Description of what this example demonstrates.

Usage:
    python examples/test_<feature>.py --arg value
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.token_self_repair.<module> import <Class>


def main():
    # Your example code here
    pass


if __name__ == "__main__":
    sys.exit(main())
```

---

## Integration Tests

All examples double as integration tests. Run them to verify your setup:

```bash
# Test LogTokU integration (Phase 1.1)
python examples/test_logtoku_integration.py

# Add more as implemented...
```

---

## Related Documentation

- [Implementation Plan](../docs/implementation_plan.md) - Full task checklist
- [Architecture](../docs/architecture.md) - System design
- [Quick Start Guide](../docs/quick_start_guide.md) - Setup instructions
- [Progress Tracker](../PROGRESS.md) - Current status

---

**Note**: Examples are added as features are implemented. Check PROGRESS.md for current completion status.


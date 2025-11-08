# Project Summary: Uncertainty-Guided Hierarchical Program Repair

## Executive Overview

This project creates the **first system** to use token-level uncertainty decomposition (aleatoric + epistemic) to dynamically guide program repair strategies across multiple granularities. By combining LogTokU's uncertainty estimation with RepairAgent's autonomous repair capabilities, we create a novel framework that knows not just WHEN to repair, but HOW to repair based on the TYPE of uncertainty detected.

## Novel Research Contributions

### 1. Uncertainty Decomposition for Program Repair
**First work** to leverage aleatoric (data ambiguity) vs epistemic (knowledge gap) uncertainty decomposition to guide automated program repair strategies.

**Impact**: Different uncertainty types require different repair approaches. High epistemic uncertainty → explore alternatives. High aleatoric uncertainty → refine and constrain.

### 2. Multi-Granularity Uncertainty Propagation
**First framework** implementing hierarchical uncertainty aggregation:
- Token-level: Individual token confidence scores
- Line-level: Statement uncertainty
- Method-level: Function-scope uncertainty  
- Test-level: Predicted test failure probability

**Impact**: Enables localization at multiple abstraction levels, improving both precision and interpretability.

### 3. Uncertainty-Driven Patch Ranking
**Novel metric** (UncertaintyScore) combining:
- Pre-execution confidence (from LogTokU)
- Post-execution test results
- Diversity bonus (penalize repetition)

**Formula**: `US = α×(1-uncertainty) + β×test_pass_rate + γ×diversity`

**Impact**: Patches with LOW uncertainty but MODERATE test pass rates may be superior to HIGH uncertainty patches with HIGH pass rates (which likely overfit).

### 4. Dynamic Strategy Selection
**First system** to automatically choose repair strategy based on uncertainty TYPE:
- **Exploration Strategy**: For high epistemic uncertainty (generate diverse alternatives)
- **Refinement Strategy**: For high aleatoric uncertainty (focused, constrained edits)
- **Hybrid Strategy**: For both high (ensemble approaches)

**Impact**: More efficient repair by matching strategy to problem characteristics.

### 5. Unified Reasoning + Code Repair Framework
**First architecture** applying unified uncertainty principles to both:
- Natural language reasoning (GSM8K, TruthfulQA)
- Program repair (Defects4J)
- Code generation (HumanEval)

**Impact**: Demonstrates generalizability of uncertainty-guided self-repair across domains.

## System Architecture

```
Input (Buggy Code / Query)
    ↓
Llama Model (LogTokU-instrumented)
    ↓ (tokens + logits)
UncertaintyEngine
    ├─ LogTokU Analyzer (AU + EU)
    └─ Hierarchical Aggregator (token→line→method)
    ↓
Dynamic Strategy Selector
    ├─ IF epistemic_high → Exploration
    ├─ IF aleatoric_high → Refinement  
    └─ ELSE → Hybrid
    ↓
UncertaintyAwareRepairAgent
    ├─ Uncertainty-Guided Prompting
    ├─ RepairAgent Core (FSM)
    └─ Confidence-Aware Mutation
    ↓
Patch Ranking Engine (UncertaintyScore)
    ↓
Hierarchical Feedback Loop
    ↓
XAI Visualization Layer
```

## Implementation Structure

### Core Components

1. **Uncertainty Engine** (`src/token_self_repair/uncertainty/`)
   - `logtoku.py`: LogTokU implementation (AU + EU decomposition)
   - `aggregation.py`: Multi-granularity uncertainty propagation
   - `base.py`: Abstract uncertainty estimator interface

2. **Repair Strategies** (`src/token_self_repair/repair/`)
   - `strategy_selector.py`: Dynamic strategy selection logic
   - `strategy_handlers.py`: Exploration, Refinement, Hybrid implementations
   - `mutation.py`: Confidence-aware patch mutation
   - `patch_ranking.py`: Novel UncertaintyScore metric

3. **Pipeline Integration** (`src/token_self_repair/pipelines/`)
   - `repair_agent.py`: RepairAgent core extraction
   - `uncertainty_adapter.py`: Uncertainty injection layer
   - `base.py`: Unified pipeline interface

4. **Evaluation Suite** (`src/token_self_repair/evaluation/`)
   - `benchmark_runner.py`: Multi-domain evaluation
   - `metrics.py`: Novel uncertainty-calibration metrics
   - `ablation.py`: Systematic component comparison

5. **Visualization & XAI** (`src/token_self_repair/visualization/`)
   - `heatmap.py`: Color-coded uncertainty visualization
   - `trajectory.py`: Repair evolution tracking
   - `explainer.py`: Interactive Q&A interface

## Evaluation Strategy

### Benchmarks (4 domains)
1. **Multi-step Reasoning**: GSM8K (100 samples)
2. **Factual Robustness**: TruthfulQA (100 samples)
3. **Program Repair**: Defects4J (20 bugs from RepairAgent's fixed list)
4. **Code Generation**: HumanEval (100 samples)

### Novel Metrics
- **Uncertainty-Calibration Curve**: Plot uncertainty vs actual error rate
- **Repair Efficiency**: Patches generated per successful fix (lower = better)
- **Strategy Accuracy**: Correlation between uncertainty type and optimal strategy
- **Granularity Localization**: Distance between uncertain regions and actual bugs

### Ablation Studies
Compare 4 configurations:
1. **Baseline**: RepairAgent without uncertainty
2. **Detection-Only**: Binary uncertainty trigger (no strategy selection)
3. **Strategy-Guided**: Full dynamic strategy selection
4. **Full System**: + Patch ranking metric

### Success Criteria
- **Defects4J**: 15-25% improvement in fix rate
- **Efficiency**: 30-40% reduction in patches per fix
- **Calibration**: AUROC > 0.75, ECE < 0.08
- **Reasoning**: 10-15% accuracy improvement on GSM8K

## Publication Targets

### Primary Venues
1. **ICSE 2026** - Software Engineering (Program Repair track)
2. **NeurIPS 2026** - ML for Code track
3. **ICLR 2026** - Agents & Planning track
4. **ASE 2026** - Automated Software Engineering

### Paper Highlights
- Novel uncertainty decomposition for repair
- First multi-granularity healing framework
- Uncertainty-driven patch ranking metric
- Dynamic strategy selection algorithm
- Comprehensive multi-domain evaluation
- Open-source reproducibility package

## Technical Innovations

### Uncertainty-Guided Prompting
```
"Lines [X, Y, Z] have HIGH EPISTEMIC uncertainty (EU > 0.7).
This indicates knowledge gaps. Consider:
- Alternative algorithms/patterns
- Broader search space
- Edge cases you may have missed

Tokens [A, B, C] have HIGH ALEATORIC uncertainty (AU > 0.7).
This indicates ambiguous logic. Consider:
- Adding constraints/specifications
- Disambiguating variable usage
- Clarifying implicit assumptions"
```

### Confidence-Aware Mutation
- Focus mutations on HIGH uncertainty regions
- Preserve HIGH confidence code (likely correct)
- More efficient than uniform mutation strategies
- 30-40% expected reduction in wasted attempts

### Hierarchical Feedback
```
Token uncertainty → Identifies problematic tokens
    ↓
Line uncertainty → Pinpoints buggy statements
    ↓  
Method uncertainty → Signals need for redesign
    ↓
Test uncertainty → Predicts test failures
    ↓
Patch uncertainty → Ranks repair candidates
```

## Resource Requirements

### Hardware
- 1x GPU: A100 (40GB) or H100 (80GB) recommended
- 32GB+ system RAM
- 200GB+ storage

### Software Stack
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- ANTLR4 (Java parser)
- Defects4J framework
- Conda/venv environment

### Models & Data
- Llama-2-7b-chat-hf (~14GB)
- Llama-3-8B-Instruct (~16GB)
- GSM8K, TruthfulQA, HumanEval datasets
- Defects4J subset (20 bugs)

## Implementation Timeline

### Phase 1: Core Infrastructure
- LogTokU integration with Llama models
- Uncertainty decomposition (AU + EU)
- Multi-granularity aggregation

### Phase 2: RepairAgent Integration  
- Extract RepairAgent FSM
- Build uncertainty adapter layer

### Phase 3: Dynamic Strategies
- Strategy selector implementation
- Exploration, Refinement, Hybrid handlers

### Phase 4: Patch Ranking
- UncertaintyScore metric
- Confidence-aware mutation

### Phase 5: Evaluation
- 4 benchmark setups
- Novel metrics implementation
- Ablation study framework

### Phase 6: Visualization
- Uncertainty heatmaps
- Repair trajectories
- Interactive explainer

### Phase 7: Documentation
- Technical report
- API documentation
- Reproducibility package

**Estimated Duration**: 15 working days for MVP

## Key Files

- **Main Checklist**: `docs/implementation_plan.md`
- **Quick Start**: `docs/quick_start_guide.md`
- **Architecture**: `docs/architecture.md`
- **Evaluation**: `docs/evaluation_plan.md`
- **This Summary**: `docs/project_summary.md`

## Expected Impact

### Academic Impact
- 5 novel research contributions
- 4+ publication opportunities
- Open-source framework for community
- New research direction: uncertainty-guided program repair

### Practical Impact
- More efficient automated repair (fewer wasted attempts)
- Better interpretability (know WHY repairs were made)
- Generalizable to multiple domains
- Integration with existing tools (RepairAgent, LogTokU)

### Community Benefits
- Complete reproducibility package
- Well-documented codebase
- Tutorial notebooks
- Docker container
- Pre-trained models

## Getting Started

1. Review `docs/implementation_plan.md` (comprehensive checklist)
2. Read `docs/quick_start_guide.md` (setup instructions)
3. Start with Phase 1.1: LogTokU Integration
4. Check off tasks as you complete them
5. Track progress at bottom of implementation plan

## Contact & Collaboration

This is a research project aimed at advancing the state-of-the-art in automated program repair through novel uncertainty quantification techniques. The framework is designed to be extensible and welcomes contributions.

**Project Repository**: `/home/dpendya/Documents/dlba`
**Documentation**: `docs/` directory
**Source Code**: `src/token_self_repair/`
**Tests**: `tests/`

---

**Note**: This is a novel, state-of-the-art research project with multiple publication-worthy contributions. The implementation plan is comprehensive and designed to produce reproducible, high-quality results.


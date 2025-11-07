# Project Roadmap: Uncertainty-Guided Hierarchical Program Repair

## Quick Links

- **Comprehensive Checklist**: [docs/implementation_plan.md](docs/implementation_plan.md) - Complete task list with checkboxes
- **Quick Start Guide**: [docs/quick_start_guide.md](docs/quick_start_guide.md) - Setup and first steps
- **Project Summary**: [docs/project_summary.md](docs/project_summary.md) - Executive overview
- **Architecture**: [docs/architecture.md](docs/architecture.md) - System design details
- **Evaluation Plan**: [docs/evaluation_plan.md](docs/evaluation_plan.md) - Metrics and benchmarks

## High-Level Overview

This project builds a **novel, state-of-the-art** system that combines:
1. **LogTokU** (uncertainty estimation with Llama models)
2. **RepairAgent** (autonomous program repair)
3. **Novel contributions** (dynamic strategies, multi-granularity, patch ranking)

### The Big Idea

Use token-level uncertainty (aleatoric + epistemic) to intelligently guide program repair:
- **Know WHEN to repair** (uncertainty detection)
- **Know HOW to repair** (strategy selection based on uncertainty type)
- **Know WHERE to repair** (multi-granularity localization)
- **Know WHICH patch** (uncertainty-driven ranking)

## Phase Structure

### Phase 1: Core Infrastructure ⚡ In Progress
**Goal**: Set up LogTokU with Llama and build multi-granularity uncertainty

**Key Deliverables**:
- [x] Llama models loaded and tested
- [x] LogTokU AU/EU decomposition working
- [x] Token → Line → Method aggregation engine (validation & viz pending)

**Files to Create**:
- [x] `src/token_self_repair/llm/llama_provider.py`
- [x] `src/token_self_repair/uncertainty/logtoku.py`
- [x] `src/token_self_repair/uncertainty/aggregation.py`

---

### Phase 2: RepairAgent Integration ⬜ Not Started
**Goal**: Extract RepairAgent core and inject uncertainty signals

**Key Deliverables**:
- [ ] RepairAgent FSM extracted
- [ ] Uncertainty adapter working
- [ ] Uncertainty-aware prompting implemented

**Files to Create**:
- `src/token_self_repair/pipelines/repair_agent.py`
- `src/token_self_repair/pipelines/uncertainty_adapter.py`

---

### Phase 3: Dynamic Strategy Selection ⬜ Not Started
**Goal**: Implement exploration vs refinement based on uncertainty type

**Key Deliverables**:
- [ ] Strategy selector with decision logic
- [ ] All strategy handlers (Exploration, Refinement, Hybrid)
- [ ] Strategy-specific prompt templates

**Files to Create**:
- `src/token_self_repair/repair/strategy_selector.py`
- `src/token_self_repair/repair/strategy_handlers.py`

---

### Phase 4: Patch Ranking ⬜ Not Started
**Goal**: Implement novel UncertaintyScore metric

**Key Deliverables**:
- [ ] UncertaintyScore formula implemented
- [ ] Confidence-aware mutation
- [ ] Patch diversity calculation

**Files to Create**:
- `src/token_self_repair/repair/patch_ranking.py`
- `src/token_self_repair/repair/mutation.py`

---

### Phase 5: Evaluation Framework ⬜ Not Started
**Goal**: Set up 4 benchmarks and run ablation studies

**Key Deliverables**:
- [ ] GSM8K, TruthfulQA, Defects4J, HumanEval configured
- [ ] Novel metrics implemented
- [ ] Ablation study framework

**Files to Create**:
- `src/token_self_repair/evaluation/benchmark_runner.py`
- `src/token_self_repair/evaluation/metrics.py`
- `src/token_self_repair/evaluation/ablation.py`

---

### Phase 6: Visualization & XAI ⬜ Not Started
**Goal**: Build heatmaps, trajectories, and interactive explainer

**Key Deliverables**:
- [ ] Uncertainty heatmap visualization
- [ ] Repair trajectory plots
- [ ] Interactive Q&A interface

**Files to Create**:
- `src/token_self_repair/visualization/heatmap.py`
- `src/token_self_repair/visualization/trajectory.py`
- `src/token_self_repair/visualization/explainer.py`

---

### Phase 7: Documentation ⬜ Not Started
**Goal**: Package for publication and reproducibility

**Key Deliverables**:
- [ ] Technical report (10-15 pages)
- [ ] API documentation
- [ ] Docker container + reproduction scripts

**Files to Create**:
- Complete Sphinx/MkDocs documentation
- Jupyter tutorial notebooks
- Docker configuration

---

## Novel Contributions Checklist

Research contributions to highlight in publications:

- [x] **C1**: Aleatoric/Epistemic uncertainty for program repair (FIRST)
- [x] **C2**: Multi-granularity uncertainty propagation (FIRST)
- [ ] **C3**: Uncertainty-driven patch ranking metric (NOVEL)
- [ ] **C4**: Dynamic strategy selection (FIRST)
- [ ] **C5**: Unified reasoning + code repair framework (FIRST)

## Success Metrics

Track these throughout implementation:

### Quantitative Targets
- [ ] Defects4J: 15-25% improvement over baseline
- [ ] Efficiency: 30-40% fewer patches per fix
- [ ] AUROC > 0.75 for uncertainty calibration
- [ ] ECE < 0.08 for calibration quality
- [ ] GSM8K: 10-15% accuracy improvement
- [ ] TruthfulQA: 15-20% accuracy improvement

### Qualitative Targets
- [ ] Heatmaps identify bugs >70% of time
- [ ] Strategy selection matches expert analysis >80%
- [ ] Patch rankings correlate with success (ρ > 0.6)
- [ ] Explanations rated helpful by developers

## Resources Setup

### Hardware Required
- [ ] GPU access configured (A100/H100)
- [ ] 32GB+ RAM available
- [ ] 200GB+ storage allocated

### Software Stack
- [ ] Python 3.10+ installed
- [ ] PyTorch 2.0+ installed
- [ ] Transformers 4.35+ installed
- [ ] ANTLR4 installed
- [ ] Defects4J framework set up

### Models & Data
- [ ] Llama-2-7b-chat-hf downloaded
- [ ] Llama-3-8B-Instruct downloaded
- [ ] GSM8K dataset accessible
- [ ] TruthfulQA dataset accessible
- [ ] Defects4J bugs list prepared
- [ ] HumanEval dataset accessible

## Publication Timeline

Target conferences with submission deadlines:

### 2026 Conferences
- [ ] **ICSE 2026** - Software Engineering
  - Submission: ~August 2025
  - Focus: Program repair contributions

- [ ] **NeurIPS 2026** - ML for Code
  - Submission: ~May 2026
  - Focus: Uncertainty decomposition + ML

- [ ] **ICLR 2026** - Agents & Planning
  - Submission: ~September 2025
  - Focus: Dynamic strategy selection

- [ ] **ASE 2026** - Automated Software Engineering
  - Submission: ~April 2026
  - Focus: Complete system + evaluation

## Implementation Strategy

### Week 1: Foundation
Focus on getting the core uncertainty engine working:
1. Set up Llama models
2. Implement LogTokU
3. Build aggregation pipeline
4. Extract RepairAgent core

### Week 2: Intelligence
Add the novel intelligent components:
1. Strategy selector
2. Strategy handlers
3. Patch ranking
4. Confidence-aware mutation

### Week 3: Evaluation & Polish
Prove it works and make it publishable:
1. Run all benchmarks
2. Generate results
3. Build visualizations
4. Write technical report

## Current Status

**Overall Progress**: ~11% (Phase 1.1 complete; 1.2 aggregation engine delivered)

### Phase Completion
- Phase 1: ⚡ 1.5/3 sections (1.1 done, 1.2 core implemented)
- Phase 2: ⬜ 0/2 sections
- Phase 3: ⬜ 0/2 sections
- Phase 4: ⬜ 0/2 sections
- Phase 5: ⬜ 0/3 sections
- Phase 6: ⬜ 0/3 sections
- Phase 7: ⬜ 0/3 sections

**Total**: 2/18 major sections substantially delivered

## Next Steps

1. **NOW**: Validate aggregator on a Defects4J sample
2. **NEXT**: Prototype line-level visualization of uncertainty hotspots
3. **THEN**: Document findings in `PROGRESS.md` and update examples
4. **TRACK**: Update this roadmap after completing Phase 1.2 validation tasks

## Key Design Decisions

### Uncertainty Decomposition
**Decision**: Use LogTokU's AU (aleatoric) + EU (epistemic) decomposition
**Rationale**: Different uncertainty types require different repair strategies

### Multi-Granularity
**Decision**: Token → Line → Method → Test hierarchy
**Rationale**: Enables localization at multiple abstraction levels

### Dynamic Strategies
**Decision**: Three strategies (Exploration, Refinement, Hybrid)
**Rationale**: Match repair approach to uncertainty characteristics

### Patch Ranking
**Decision**: Combine pre-execution confidence + post-execution tests
**Rationale**: Low-uncertainty patches with moderate tests > high-uncertainty patches with high tests

### Unified Framework
**Decision**: Single architecture for reasoning + code repair
**Rationale**: Demonstrates generalizability of uncertainty-guided self-repair

## Notes & Insights

Add notes here as you implement:

### Challenges Encountered
- [ ] Document any major obstacles
- [ ] Note solutions found

### Interesting Findings
- [ ] Unexpected patterns in uncertainty
- [ ] Strategy effectiveness observations

### Ideas for Future Work
- [ ] Extensions beyond initial scope
- [ ] Additional benchmarks
- [ ] Alternative algorithms

---

**Remember**: Check off tasks in `docs/implementation_plan.md` as you complete them. Update this roadmap periodically to track phase completion.

**Good Luck!** This is novel, state-of-the-art research with multiple publication-worthy contributions.


# Final Implementation Checklist

This checklist captures the end-to-end deliverables to complete today. Each section
is a major milestone; tick items as you finish them.

## 0. Environment & Baseline Verification
- [x] Confirm GPU/CPU availability and install required dependencies (`torch`, `transformers`, `accelerate`, `langchain`, `streamlit`, `ragas`, etc.).
- [ ] Verify Hugging Face credentials for `meta-llama` models and OpenAI API key.
- [x] Run smoke tests:
  - [x] `examples/test_logtoku_integration.py`
  - [x] `examples/run_reasoning_demo.py --model meta-llama/Llama-3.2-3B-Instruct`

## 1. Streamlit RAG + Uncertainty Assistant
- [ ] Scaffold Streamlit app (`app/main.py`) with sidebar config.
- [ ] Implement document ingestion (local uploads + embedding store).
- [ ] Integrate web search (e.g., Tavily/SerpAPI wrapper) into retriever.
- [ ] Build retrieval orchestrator:
  - [ ] Base retriever (vector search + filters).
  - [ ] Extended retrieval when confidence low (iterative expansion).
- [ ] Wire LLM routing:
  - [ ] Local `Llama-3.2` via `llama_provider`.
  - [ ] OpenAI `gpt-4o` (toggle in UI).
- [ ] Responses pipeline:
  - [ ] Generate answer with citations.
  - [ ] Compute uncertainty metrics (EU, AU, LogTokU).
  - [ ] Compute RAG-specific metrics (RAGAS: faithfulness, answer relevancy, context precision/recall).
- [ ] UI outputs:
  - [ ] Chat transcript with toggle for raw answer vs. repaired answer.
  - [ ] Confidence dashboard (EU, AU, LogTokU, entropy, RAGAS scores, retrieval depth).
  - [ ] Retrieved context viewer with highlight of uncertain spans.
- [ ] Repair loop:
  - [ ] Detect low-confidence responses (threshold + metrics).
  - [ ] Show remediation plan to user (extend retrieval, revise prompt, alternative strategy).
  - [ ] On user approval, execute repair and display delta.
- [ ] Logging & analytics:
  - [ ] Store conversation history and metrics.
  - [ ] Export session summary (JSON/CSV).

## 2. ActiveRAG-Next Feature Parity
- [ ] Review ActiveRAG-Next (UI flows, Agent graph, self-repair flows).
- [ ] Port/improve:
  - [ ] Agent graph (planner, retriever, synthesizer, critic).
  - [ ] Conversation memory + scratchpad.
  - [ ] Adaptive retrieval budget and stop conditions.
- [ ] Ensure compatibility with uncertainty metrics (EU/AU-driven decisions).

## 3. Reasoning Benchmark Dashboard
- [ ] Extend evaluation runner:
  - [ ] Full GSM8K dataset loader and numeric answer extraction.
  - [ ] TruthfulQA and other math datasets (e.g., MATH, AquaratLite) minimal subset.
  - [ ] Configurable sample size (UI selectable).
- [ ] Compute metrics:
  - [ ] Accuracy, calibration error, uncertainty histograms.
  - [ ] Strategy usage stats (after Phase 3 implementation).
  - [ ] Cost & latency per run.
- [ ] Streamlit integration:
  - [ ] Dedicated dashboard tab with filters.
  - [ ] Display per-sample analysis (prompt, answer, uncertainty summary, hotspots).
  - [ ] Downloadable results (CSV/JSON).

## 4. Dynamic Strategy Selection (Phase 3)
- [ ] Implement `RepairStrategy` enum and `StrategySelector`.
- [ ] Build `ExplorationHandler`, `RefinementHandler`, `HybridHandler`.
- [ ] Integrate with reasoning and RAG flows:
  - [ ] Highlight chosen strategy in UI.
  - [ ] Allow user override / feedback loop.
- [ ] Add metrics to track strategy effectiveness.

## 5. RepairAgent Integration (Phase 2)
- [ ] Extract `RepairAgentCore` (FSM & API) from `RepairAgent/repair_agent`.
- [ ] Implement `UncertaintyAwareRepairAgent` adapter:
  - [ ] Uncertainty-informed prompt infusion.
  - [ ] Iterative repair with uncertainty logging.
- [ ] Connect to Streamlit app (developer console tab).
- [ ] Run smoke test on 1-2 Defects4J bugs (requires environment).

## 6. Patch Ranking & Mutations (Phase 4)
- [ ] Implement `PatchRanker` with UncertaintyScore metric.
- [ ] Implement `ConfidenceAwareMutator` (target high-uncertainty regions).
- [ ] Add to repair pipeline and UI (show ranked patches with metrics).

## 7. Visualization & XAI (Phase 6)
- [ ] Token/line heatmap module (matplotlib/plotly) for uncertainty.
- [ ] Repair trajectory visualization (iteration timeline).
- [ ] Interactive explainer panel (user queries about decisions).
- [ ] Integrate visual components into Streamlit UI.

## 8. Evaluation & Reporting (Phase 5 & 7)
- [ ] Full benchmark runs (RAG + Reasoning + Repair).
- [ ] Generate comparison tables (baseline vs. uncertainty-guided).
- [ ] Update documentation:
  - [ ] `STATUS.md` with final results.
  - [ ] Streamlit user guide.
  - [ ] API docs for new modules.
- [ ] Prepare final report slides or summary for stakeholders.

---



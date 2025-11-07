# LangGraph-Based Token Self-Repair Implementation

## Overview

This document describes the LangGraph-based implementation of the Token-Level Uncertainty-Driven Self-Repair system (Phase 2: Agentic Chassis).

## Components Implemented

### 1. AgentState (types.py)

A TypedDict that defines the state structure for the LangGraph workflow:

- `question`: Original question/prompt
- `generation`: Generated answer text
- `token_uncertainties`: List of per-token uncertainty scores
- `avg_uncertainty`: Average uncertainty across all tokens
- `repair_attempts`: Number of repair iterations
- `xai_message`: Human-readable status message for UI
- `messages`: List of messages (refined prompts)
- `logits`: Token logits for uncertainty computation
- `tokens`: Generated tokens
- `token_scores`: Full TokenScore objects for evaluation integration

### 2. Reflector (repair/constitutional.py)

A wrapper around `ConstitutionalRepair` that provides a simplified `critique()` interface:

```python
reflector = Reflector()
revised_prompt = reflector.critique(
    previous_answer="...",
    question="...",
    repair_attempt=0,
    avg_uncertainty=0.6,
    uncertainty_level=UncertaintyLevel.MODERATE
)
```

### 3. TokenSelfRepairGraph (pipelines/token_self_repair_graph.py)

The main LangGraph-based self-repair agent with four nodes:

#### Nodes

1. **generator_node**: Generates answer using LLM
   - Calls `llm.generate()` to get tokens and logits
   - Updates state with generation, tokens, and logits

2. **uncertainty_node**: Computes per-token uncertainty
   - Uses `LogTokUEstimator` to compute uncertainty scores
   - Calculates average uncertainty
   - Updates state with token_scores and avg_uncertainty

3. **reflector_node**: Critiques and refines answer
   - Uses `Reflector.critique()` to generate revised prompt
   - Increments repair_attempts
   - Updates messages with refined prompt

4. **finalize_node**: Prepares final output
   - Sets completion message
   - Logs final statistics

#### Graph Structure

```
Entry → generator → uncertainty → [conditional routing]
                                    ↓
                            ┌───────┴───────┐
                            ↓               ↓
                        reflector      finalize
                            ↓               ↓
                            └──→ generator  END
```

#### Conditional Routing

The `_route_on_uncertainty()` function routes based on:
- `avg_uncertainty > UNCERTAINTY_THRESHOLD` → "reflector"
- `repair_attempts >= MAX_REPAIR_ATTEMPTS` → "finalize"
- Otherwise → "finalize"

### 4. GraphPipelineAdapter (pipelines/graph_adapter.py)

An adapter that wraps `TokenSelfRepairGraph` to make it compatible with the existing evaluation infrastructure (`UncertaintyAwarePipeline` interface).

## Usage

### Basic Usage

```python
from token_self_repair.pipelines import TokenSelfRepairGraph
from token_self_repair.llm.base import LLMClient
from token_self_repair.uncertainty.logtoku import LogTokUEstimator
from token_self_repair.repair.constitutional import Reflector
from token_self_repair.messaging.status import StatusMessenger
from token_self_repair.config import ProjectConfig

# Initialize components
llm = YourLLMClient()  # Implement LLMClient interface
estimator = LogTokUEstimator(ProjectConfig())
reflector = Reflector()
messenger = StatusMessenger()
config = ProjectConfig()

# Create graph
graph = TokenSelfRepairGraph(
    llm=llm,
    estimator=estimator,
    reflector=reflector,
    messenger=messenger,
    config=config,
)

# Run
result = graph.run("What is 2+2?")
print(result["answer"])
print(result["xai_message"])
print(result["meta"])
```

### Streaming Usage

```python
# Stream intermediate updates
for update in graph.run("What is 2+2?", stream=True):
    print(update["xai_message"])
    if "answer" in update:
        print(f"Final answer: {update['answer']}")
```

### Evaluation Integration

```python
from token_self_repair.pipelines import create_graph_pipeline
from token_self_repair.evaluation import EvaluationRunner

# Create pipeline factory
def pipeline_factory():
    return create_graph_pipeline(
        llm=YourLLMClient(),
        # Other components use defaults
    )

# Run evaluation
runner = EvaluationRunner(pipeline_factory=pipeline_factory)
result = runner.run("gsm8k")
```

## Configuration

Constants in `config.py`:

- `UNCERTAINTY_THRESHOLD = 0.5`: Threshold for triggering repair
- `MAX_REPAIR_ATTEMPTS = 3`: Maximum repair iterations

## Integration Points

### LLM Integration
- Uses `LLMClient` interface from `llm/base.py`
- Requires `generate()` method that yields `TokenLogit` objects

### Uncertainty Integration
- Uses `LogTokUEstimator` from `uncertainty/logtoku.py`
- Computes LogTokU metric per token

### Repair Integration
- Uses `Reflector` (wrapper around `ConstitutionalRepair`)
- Applies constitutional AI principles for critique

### Messaging Integration
- Uses `StatusMessenger` from `messaging/status.py`
- Emits status updates for UI visualization

### Evaluation Integration
- `GraphPipelineAdapter` provides compatibility with `EvaluationRunner`
- Returns `PipelineResult` with `GenerationStep` for metrics

## Architecture Benefits

1. **Modularity**: Clear separation of concerns (generation, uncertainty, repair, finalization)
2. **Extensibility**: Easy to add new nodes or modify routing logic
3. **Testability**: Each node can be tested independently
4. **Observability**: Streaming support for real-time monitoring
5. **Compatibility**: Adapter pattern allows use with existing evaluation infrastructure

## Future Enhancements

- Add checkpointing for state persistence
- Implement parallel uncertainty computation
- Add support for multiple repair strategies
- Enhance streaming with more granular updates
- Add metrics collection at each node


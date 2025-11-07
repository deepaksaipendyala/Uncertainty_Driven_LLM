# Uncertainty-Guided Hierarchical Program Repair: Implementation Plan

## Project Overview

**Novel Contribution**: The first system to use token-level uncertainty decomposition (aleatoric + epistemic) to dynamically guide program repair strategies across multiple granularities (token → line → method → test).

**Key Innovation Points**:
1. Uncertainty-Driven Patch Ranking: Novel metric combining token-level confidence scores with test execution feedback
2. Dynamic Strategy Selection: Epistemic uncertainty triggers search/exploration, aleatoric uncertainty triggers refinement
3. Multi-Granularity Healing: Operates at token, statement, method, and test levels simultaneously
4. Unified Framework: Single architecture handles both reasoning tasks (GSM8K) and program repair (Defects4J)
5. Real-time Explainability: Uncertainty visualization showing WHERE and WHY the model is uncertain

---

## Phase 1: Core Infrastructure Setup

### 1.1 LogTokU Integration with Llama Models

**Goal**: Set up Llama models from the LogTokU repository and extract uncertainty scores

#### Tasks:
- [ ] Verify Llama model downloads in `logtoku/models/` directory
- [ ] Test basic model loading with transformers library
- [ ] Configure GPU memory optimization (bfloat16, device_map="auto")
- [ ] Implement logit extraction using `output_scores=True` parameter
- [ ] Verify logit tensor shapes and vocabulary alignment

#### Implementation Details:
```python
# Location: src/token_self_repair/llm/llama_provider.py

class LlamaProvider(LLMClient):
    """
    Wraps Llama models with logit extraction capabilities.
    Supports: Llama-2-7b, Llama-2-13b, Llama-3-8B
    """
    def __init__(self, model_name: str):
        # Load from logtoku/models/ directory
        # Configure with output_scores=True
        pass
    
    def generate_with_logits(self, prompt: str, max_tokens: int = 256):
        # Returns: (tokens, logits_tensor)
        # Shape: logits_tensor = (num_tokens, vocab_size)
        pass
```

#### Deliverable:
- [ ] `LlamaProvider` class that returns both tokens and logits
- [ ] Unit test verifying logit extraction
- [ ] Benchmark inference speed (tokens/sec)

#### Expected Output:
```python
provider = LlamaProvider("meta-llama/Llama-2-7b-chat-hf")
tokens, logits = provider.generate_with_logits("def factorial(n):")
assert logits.shape == (len(tokens), vocab_size)
```

---

### 1.2 LogTokU Uncertainty Decomposition

**Goal**: Implement the full LogTokU algorithm for aleatoric and epistemic uncertainty

#### Tasks:
- [ ] Study the LogTokU paper formulation (equations for AU and EU)
- [ ] Implement Dirichlet evidence extraction from logits
- [ ] Implement aleatoric uncertainty calculation (AU)
- [ ] Implement epistemic uncertainty calculation (EU)
- [ ] Create unified uncertainty score aggregation
- [ ] Validate against LogTokU repository's reference implementation

#### Implementation Details:
```python
# Location: src/token_self_repair/uncertainty/logtoku.py

class LogTokUEstimator:
    """
    Implements Logits-induced Token Uncertainty decomposition.
    
    Reference: Ma et al. 2025 - "Estimating LLM Uncertainty with Evidence"
    """
    
    def calculate_evidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Extract Dirichlet evidence from raw logits.
        
        Args:
            logits: Shape (num_tokens, vocab_size)
        Returns:
            evidence: Shape (num_tokens,)
        """
        pass
    
    def aleatoric_uncertainty(self, logits: torch.Tensor) -> np.ndarray:
        """
        Calculate data-inherent uncertainty (AU).
        High AU = multiple valid interpretations exist in training data
        
        Returns:
            au_scores: Shape (num_tokens,)
        """
        pass
    
    def epistemic_uncertainty(self, logits: torch.Tensor) -> np.ndarray:
        """
        Calculate knowledge-gap uncertainty (EU).
        High EU = model lacks similar training examples
        
        Returns:
            eu_scores: Shape (num_tokens,)
        """
        pass
    
    def analyze(self, logits: torch.Tensor) -> UncertaintyScores:
        """
        Full analysis returning both AU and EU.
        
        Returns:
            UncertaintyScores with fields: au, eu, total
        """
        pass
```

#### Deliverable:
- [ ] Complete `LogTokUEstimator` class
- [ ] Validation test comparing against LogTokU repo outputs
- [ ] Visualization of AU vs EU distributions

#### Expected Output:
```python
estimator = LogTokUEstimator()
scores = estimator.analyze(logits)
# scores.au: array of aleatoric uncertainty per token
# scores.eu: array of epistemic uncertainty per token
# scores.total: combined uncertainty metric
```

---

### 1.3 Multi-Granularity Aggregation

**Goal**: Aggregate token-level uncertainty to line, method, and test levels

#### Tasks:
- [x] Implement token-to-line mapping using heuristic newline tracking (AST optional)
- [x] Create line-to-method mapping via Java regex & Python indentation heuristics
- [x] Implement aggregation strategies (mean, max, median, weighted)
- [x] Handle multi-line tokens and whitespace-sensitive boundaries
- [x] Create method-level uncertainty profiles

#### Implementation Details:
```python
# Location: src/token_self_repair/uncertainty/aggregation.py

class UncertaintyAggregator:
    """
    Hierarchical aggregation: Token → Line → Method → Test
    """
    
    def token_to_line(
        self, 
        token_scores: np.ndarray,
        token_positions: list[tuple[int, int]]  # (line_num, col_num)
    ) -> dict[int, LineUncertainty]:
        """
        Aggregate token scores to line-level.
        
        Strategies:
        - MAX: Take highest uncertainty token in line
        - MEAN: Average uncertainty across line
        - WEIGHTED: Weight by token importance (operators > identifiers)
        """
        pass
    
    def line_to_method(
        self,
        line_scores: dict[int, LineUncertainty],
        ast_tree: Any
    ) -> dict[str, MethodUncertainty]:
        """
        Aggregate line scores to method-level using AST.
        """
        pass
    
    def build_uncertainty_map(
        self,
        token_scores: UncertaintyScores,
        source_code: str
    ) -> UncertaintyMap:
        """
        Complete hierarchical mapping.
        
        Returns:
            UncertaintyMap with:
            - token_scores: per-token AU/EU
            - line_scores: per-line aggregated scores
            - method_scores: per-method aggregated scores
            - hotspots: ranked list of most uncertain regions
        """
        pass
```

#### Deliverable:
- [x] `UncertaintyAggregator` class with aggregation strategies
- [ ] Optional: Integration with Java AST parser (ANTLR)
- [ ] Visualization of multi-level uncertainty maps
- [ ] Test on sample Defects4J bugs

#### Expected Output:
```python
aggregator = UncertaintyAggregator()
u_map = aggregator.build_uncertainty_map(token_scores, source_code)

# Access different granularities
print(u_map.line_scores[42])  # Line 42 uncertainty
print(u_map.method_scores["calculateTotal"])  # Method uncertainty
print(u_map.hotspots[:5])  # Top 5 most uncertain locations
```

---

## Phase 2: RepairAgent Integration

### 2.1 RepairAgent Core Extraction

**Goal**: Extract RepairAgent's FSM logic into a standalone, modular component

#### Tasks:
- [ ] Analyze RepairAgent's state machine transitions
- [ ] Extract core repair loop from AutoGPT integration
- [ ] Separate test execution logic from repair logic
- [ ] Create clean interface for external triggers
- [ ] Remove OpenAI-specific dependencies (make model-agnostic)

#### Implementation Details:
```python
# Location: src/token_self_repair/pipelines/repair_agent.py

class RepairAgentCore:
    """
    Extracted core logic from RepairAgent without framework dependencies.
    """
    
    def __init__(self, defects4j_home: str, llm_client: LLMClient):
        self.defects4j = Defects4JInterface(defects4j_home)
        self.llm = llm_client
        self.state = RepairState()
    
    def checkout_bug(self, project: str, bug_id: int):
        """Checkout Defects4J bug to workspace."""
        pass
    
    def localize_fault(self) -> list[str]:
        """
        Return list of suspicious files/methods.
        Uses RepairAgent's localization strategy.
        """
        pass
    
    def generate_patch(
        self, 
        buggy_code: str,
        context: str,
        constraints: list[str] = None
    ) -> str:
        """
        Generate a single patch using LLM.
        
        Args:
            buggy_code: The identified buggy section
            context: Surrounding code and test info
            constraints: Optional repair constraints
        """
        pass
    
    def execute_tests(self, patched_code: str) -> TestResult:
        """Execute Defects4J tests on patched version."""
        pass
    
    def mutate_patch(self, failed_patch: str, failure_info: str) -> str:
        """
        Generate mutation of failed patch.
        Original RepairAgent mutation strategy.
        """
        pass
```

#### Deliverable:
- [ ] `RepairAgentCore` class with clean API
- [ ] Defects4J integration wrapper
- [ ] Test on 3 sample bugs from RepairAgent's fixed list
- [ ] Documentation of FSM states and transitions

---

### 2.2 Uncertainty-Aware Adapter

**Goal**: Create adapter layer that injects uncertainty signals into RepairAgent

#### Tasks:
- [ ] Design uncertainty injection points in repair loop
- [ ] Create uncertainty-aware prompt templates
- [ ] Implement feedback mechanism from uncertainty to repair strategy
- [ ] Add uncertainty tracking across repair iterations
- [ ] Create logging system for uncertainty evolution

#### Implementation Details:
```python
# Location: src/token_self_repair/pipelines/uncertainty_adapter.py

class UncertaintyAwareRepairAgent:
    """
    Wraps RepairAgentCore with uncertainty monitoring and guidance.
    """
    
    def __init__(
        self, 
        repair_agent: RepairAgentCore,
        uncertainty_engine: UncertaintyEngine,
        strategy_selector: StrategySelector
    ):
        self.agent = repair_agent
        self.uncertainty = uncertainty_engine
        self.strategy = strategy_selector
        self.history = []  # Track uncertainty evolution
    
    def repair_with_uncertainty(
        self,
        project: str,
        bug_id: int,
        max_attempts: int = 10
    ) -> RepairResult:
        """
        Main repair loop with uncertainty guidance.
        
        1. Checkout bug
        2. Localize fault
        3. Generate initial patch (with logits)
        4. Analyze uncertainty
        5. Select strategy based on uncertainty type
        6. Generate refined patch or explore alternatives
        7. Execute tests
        8. Repeat until success or max_attempts
        """
        pass
    
    def inject_uncertainty_prompt(
        self,
        base_prompt: str,
        uncertainty_map: UncertaintyMap
    ) -> str:
        """
        Augment repair prompt with uncertainty information.
        
        Example:
        "The following locations have high epistemic uncertainty:
         - Line 42: token 'null' (EU=0.87)
         - Line 56: token 'equals' (EU=0.72)
        
        This suggests the model lacks similar examples. Consider:
        - Alternative null-checking patterns
        - Different comparison methods"
        """
        pass
    
    def track_uncertainty_evolution(
        self,
        iteration: int,
        u_map: UncertaintyMap
    ):
        """
        Log how uncertainty changes across repair attempts.
        Used for visualization and analysis.
        """
        pass
```

#### Deliverable:
- [ ] `UncertaintyAwareRepairAgent` class
- [ ] Uncertainty prompt templates
- [ ] Integration test showing uncertainty injection
- [ ] Logging system for repair trajectories

---

## Phase 3: Dynamic Strategy Selection

### 3.1 Strategy Selector Implementation

**Goal**: Implement decision logic for choosing repair strategy based on uncertainty type

#### Tasks:
- [ ] Define strategy types (Exploration, Refinement, Hybrid)
- [ ] Implement decision tree based on AU/EU thresholds
- [ ] Create strategy-specific prompt templates
- [ ] Implement adaptive threshold adjustment
- [ ] Add strategy performance tracking

#### Implementation Details:
```python
# Location: src/token_self_repair/repair/strategy_selector.py

class RepairStrategy(Enum):
    EXPLORATION = "exploration"  # For high EU
    REFINEMENT = "refinement"    # For high AU
    HYBRID = "hybrid"            # For balanced AU/EU
    STANDARD = "standard"        # For low uncertainty

class StrategySelector:
    """
    Selects optimal repair strategy based on uncertainty decomposition.
    """
    
    def __init__(
        self,
        eu_threshold: float = 0.6,
        au_threshold: float = 0.6
    ):
        self.eu_threshold = eu_threshold
        self.au_threshold = au_threshold
        self.performance_history = {}  # Track strategy success rates
    
    def select_strategy(
        self,
        uncertainty_map: UncertaintyMap
    ) -> RepairStrategy:
        """
        Decision logic:
        
        IF EU_avg > eu_threshold AND AU_avg < au_threshold:
            RETURN EXPLORATION (knowledge gap)
        
        ELIF AU_avg > au_threshold AND EU_avg < eu_threshold:
            RETURN REFINEMENT (ambiguity)
        
        ELIF EU_avg > eu_threshold AND AU_avg > au_threshold:
            RETURN HYBRID (both issues)
        
        ELSE:
            RETURN STANDARD (low uncertainty)
        """
        pass
    
    def get_strategy_prompt(
        self,
        strategy: RepairStrategy,
        uncertainty_map: UncertaintyMap
    ) -> str:
        """
        Generate strategy-specific repair instructions.
        """
        if strategy == RepairStrategy.EXPLORATION:
            return self._exploration_prompt(uncertainty_map)
        elif strategy == RepairStrategy.REFINEMENT:
            return self._refinement_prompt(uncertainty_map)
        elif strategy == RepairStrategy.HYBRID:
            return self._hybrid_prompt(uncertainty_map)
        else:
            return self._standard_prompt()
    
    def _exploration_prompt(self, u_map: UncertaintyMap) -> str:
        """
        HIGH EPISTEMIC UNCERTAINTY PROMPT
        
        "Your uncertainty analysis indicates knowledge gaps at:
        {list high EU locations}
        
        This suggests you may not have seen similar code patterns.
        
        Strategy: EXPLORATION
        - Generate multiple diverse patch approaches (3-5 variants)
        - Consider alternative algorithms/design patterns
        - Search for edge cases you might have missed
        - Query broader context (documentation, similar bugs)
        - Don't restrict to minimal edits"
        """
        pass
    
    def _refinement_prompt(self, u_map: UncertaintyMap) -> str:
        """
        HIGH ALEATORIC UNCERTAINTY PROMPT
        
        "Your uncertainty analysis indicates ambiguous logic at:
        {list high AU locations}
        
        This suggests multiple valid interpretations exist.
        
        Strategy: REFINEMENT
        - Focus on specific uncertain tokens: {tokens}
        - Add clarifying constraints/assertions
        - Disambiguate variable usage
        - Make implicit assumptions explicit
        - Prefer minimal, targeted edits"
        """
        pass
    
    def _hybrid_prompt(self, u_map: UncertaintyMap) -> str:
        """
        BOTH HIGH PROMPT
        
        Combine exploration and refinement strategies.
        """
        pass
    
    def update_performance(
        self,
        strategy: RepairStrategy,
        success: bool
    ):
        """
        Track which strategies work best.
        Can be used for adaptive threshold tuning.
        """
        pass
```

#### Deliverable:
- [ ] `StrategySelector` class with decision logic
- [ ] All four strategy prompt templates
- [ ] Performance tracking system
- [ ] Unit tests for strategy selection logic

---

### 3.2 Strategy Execution Handlers

**Goal**: Implement specialized behavior for each repair strategy

#### Tasks:
- [ ] Create exploration strategy handler (multiple diverse patches)
- [ ] Create refinement strategy handler (focused mutations)
- [ ] Create hybrid strategy handler (ensemble approach)
- [ ] Implement patch diversity metrics
- [ ] Add strategy-specific stopping criteria

#### Implementation Details:
```python
# Location: src/token_self_repair/repair/strategy_handlers.py

class ExplorationHandler:
    """
    Handles HIGH EPISTEMIC uncertainty repairs.
    Generates diverse patch alternatives.
    """
    
    def generate_patches(
        self,
        buggy_code: str,
        uncertainty_map: UncertaintyMap,
        num_variants: int = 5
    ) -> list[PatchCandidate]:
        """
        Generate multiple diverse patches.
        
        Techniques:
        - Temperature sampling (higher temp for diversity)
        - Multiple prompts with different framings
        - Beam search with diversity penalty
        """
        pass

class RefinementHandler:
    """
    Handles HIGH ALEATORIC uncertainty repairs.
    Generates focused, targeted patches.
    """
    
    def generate_patches(
        self,
        buggy_code: str,
        uncertainty_map: UncertaintyMap
    ) -> list[PatchCandidate]:
        """
        Generate focused patches targeting uncertain tokens.
        
        Techniques:
        - Low temperature (deterministic)
        - Constrained generation (only edit uncertain regions)
        - Add specifications to reduce ambiguity
        """
        pass

class HybridHandler:
    """
    Handles BOTH HIGH uncertainty.
    Combines exploration and refinement.
    """
    
    def generate_patches(
        self,
        buggy_code: str,
        uncertainty_map: UncertaintyMap
    ) -> list[PatchCandidate]:
        """
        Ensemble approach:
        1. Generate 3 exploration patches
        2. For each, generate 2 refinement variants
        3. Rank by combined uncertainty score
        """
        pass
```

#### Deliverable:
- [ ] All three strategy handler classes
- [ ] Patch diversity metric implementation
- [ ] Integration with uncertainty-aware repair agent
- [ ] Test on synthetic examples with known uncertainty patterns

---

## Phase 4: Uncertainty-Driven Patch Ranking

### 4.1 Patch Ranking Metric

**Goal**: Develop novel metric combining pre-execution uncertainty with post-execution test results

#### Tasks:
- [ ] Design UncertaintyScore formula with tunable weights
- [ ] Implement patch diversity bonus calculation
- [ ] Create test coverage analysis integration
- [ ] Implement ranking algorithm
- [ ] Add confidence intervals for ranking stability

#### Implementation Details:
```python
# Location: src/token_self_repair/repair/patch_ranking.py

@dataclass
class PatchCandidate:
    """Represents a generated patch with metadata."""
    patch_code: str
    uncertainty_map: UncertaintyMap
    generation_strategy: RepairStrategy
    test_results: Optional[TestResult] = None
    rank_score: Optional[float] = None

class PatchRanker:
    """
    Ranks patch candidates using uncertainty-driven metric.
    """
    
    def __init__(
        self,
        alpha: float = 0.4,  # Weight for uncertainty term
        beta: float = 0.5,   # Weight for test pass rate
        gamma: float = 0.1   # Weight for diversity bonus
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seen_patches = []  # For diversity calculation
    
    def calculate_uncertainty_score(
        self,
        patch: PatchCandidate
    ) -> float:
        """
        Novel UncertaintyScore (US) metric:
        
        US(patch) = α × (1 - avg_uncertainty) 
                  + β × test_pass_rate 
                  + γ × diversity_bonus
        
        Where:
        - avg_uncertainty: Mean token-level uncertainty in patch
        - test_pass_rate: Percentage of tests passed (0.0-1.0)
        - diversity_bonus: Edit distance from previous patches
        
        Key Insight:
        Patches with LOW uncertainty but MODERATE test pass rates 
        may be better than HIGH uncertainty patches with HIGH pass 
        rates (likely overfitting to specific tests).
        """
        u_term = 1.0 - patch.uncertainty_map.average_total_uncertainty
        
        if patch.test_results:
            t_term = patch.test_results.pass_rate
        else:
            t_term = 0.0  # Penalize untested patches
        
        d_term = self._diversity_bonus(patch)
        
        score = (self.alpha * u_term + 
                 self.beta * t_term + 
                 self.gamma * d_term)
        
        return score
    
    def _diversity_bonus(self, patch: PatchCandidate) -> float:
        """
        Reward patches that are different from previous attempts.
        Uses normalized edit distance.
        """
        if not self.seen_patches:
            return 1.0  # First patch gets max bonus
        
        distances = []
        for prev in self.seen_patches:
            dist = self._edit_distance(patch.patch_code, prev)
            distances.append(dist)
        
        # Average distance, normalized by code length
        avg_dist = np.mean(distances)
        max_dist = max(len(patch.patch_code), 
                      max(len(p) for p in self.seen_patches))
        
        return avg_dist / max_dist
    
    def rank_patches(
        self,
        candidates: list[PatchCandidate]
    ) -> list[PatchCandidate]:
        """
        Rank all candidates and return sorted list.
        Updates each patch's rank_score field.
        """
        for patch in candidates:
            patch.rank_score = self.calculate_uncertainty_score(patch)
        
        # Sort descending by score
        ranked = sorted(candidates, 
                       key=lambda p: p.rank_score, 
                       reverse=True)
        
        # Add to history for future diversity calculations
        self.seen_patches.extend([p.patch_code for p in candidates])
        
        return ranked
```

#### Deliverable:
- [ ] `PatchRanker` class with UncertaintyScore implementation
- [ ] Diversity bonus calculation
- [ ] Integration with test execution
- [ ] Validation on known successful/failed patches

---

### 4.2 Confidence-Aware Mutation

**Goal**: Use uncertainty maps to guide intelligent patch mutations

#### Tasks:
- [ ] Implement focused mutation targeting uncertain regions
- [ ] Create mutation operators (replace, insert, delete)
- [ ] Implement conservation of high-confidence code
- [ ] Add mutation history tracking
- [ ] Implement mutation diversity control

#### Implementation Details:
```python
# Location: src/token_self_repair/repair/mutation.py

class ConfidenceAwareMutator:
    """
    Generates patch mutations focused on uncertain regions.
    Preserves high-confidence code sections.
    """
    
    def mutate_patch(
        self,
        failed_patch: str,
        uncertainty_map: UncertaintyMap,
        test_failures: TestResult
    ) -> list[str]:
        """
        Generate mutations focused on uncertain regions.
        
        Algorithm:
        1. Identify TOP-K most uncertain tokens/lines
        2. Avoid mutating high-confidence regions (likely correct)
        3. Generate targeted mutations for uncertain regions
        4. Apply diverse mutation operators
        """
        # Extract high uncertainty locations
        uncertain_lines = self._get_uncertain_lines(
            uncertainty_map, 
            top_k=5
        )
        
        # Get high confidence regions to preserve
        confident_lines = self._get_confident_lines(
            uncertainty_map,
            threshold=0.3  # Only mutate if uncertainty > 0.3
        )
        
        mutations = []
        
        for line_num in uncertain_lines:
            # Apply multiple mutation operators
            mutations.extend(
                self._mutate_line(
                    failed_patch, 
                    line_num,
                    preserve_regions=confident_lines
                )
            )
        
        return mutations
    
    def _get_uncertain_lines(
        self,
        u_map: UncertaintyMap,
        top_k: int
    ) -> list[int]:
        """
        Return line numbers with highest uncertainty.
        """
        sorted_lines = sorted(
            u_map.line_scores.items(),
            key=lambda x: x[1].total_uncertainty,
            reverse=True
        )
        return [line_num for line_num, _ in sorted_lines[:top_k]]
    
    def _mutate_line(
        self,
        code: str,
        target_line: int,
        preserve_regions: set[int]
    ) -> list[str]:
        """
        Apply mutation operators to specific line.
        
        Operators:
        - REPLACE: Replace uncertain token with alternative
        - INSERT: Add defensive check/assertion
        - DELETE: Remove potentially incorrect logic
        - REORDER: Change statement order
        """
        pass
```

#### Deliverable:
- [ ] `ConfidenceAwareMutator` class
- [ ] All mutation operators implemented
- [ ] Mutation efficiency tests (mutations per fix)
- [ ] Comparison with RepairAgent's general mutation

---

## Phase 5: Unified Evaluation Framework

### 5.1 Multi-Domain Benchmark Setup

**Goal**: Set up evaluation harnesses for reasoning, factuality, and program repair

#### Tasks:
- [ ] Configure GSM8K evaluation pipeline
- [ ] Configure TruthfulQA evaluation pipeline
- [ ] Configure Defects4J subset (20 bugs from RepairAgent's list)
- [ ] Configure HumanEval code generation benchmark
- [ ] Create unified evaluation runner
- [ ] Implement result aggregation and reporting

#### Implementation Details:
```python
# Location: src/token_self_repair/evaluation/benchmark_runner.py

class UnifiedBenchmarkRunner:
    """
    Runs all benchmarks with consistent uncertainty tracking.
    """
    
    def __init__(
        self,
        uncertainty_engine: UncertaintyEngine,
        repair_system: UncertaintyAwareRepairAgent
    ):
        self.uncertainty = uncertainty_engine
        self.repair = repair_system
        self.results = {}
    
    def run_gsm8k(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Run GSM8K reasoning benchmark.
        
        Metrics:
        - Accuracy (exact match)
        - Uncertainty-error correlation (AUROC)
        - Repair success rate
        """
        pass
    
    def run_truthfulqa(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Run TruthfulQA factuality benchmark.
        
        Metrics:
        - MC1 accuracy
        - MC2 accuracy
        - Uncertainty calibration (ECE)
        """
        pass
    
    def run_defects4j(self, bug_list: list[str]) -> BenchmarkResult:
        """
        Run Defects4J program repair benchmark.
        
        Metrics:
        - Fix success rate
        - Patches generated per fix
        - Average repair iterations
        - Uncertainty-guided vs baseline
        """
        pass
    
    def run_humaneval(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Run HumanEval code generation benchmark.
        
        Metrics:
        - pass@1, pass@10
        - Uncertainty-error correlation
        - Repair improvement rate
        """
        pass
    
    def run_all(self) -> dict[str, BenchmarkResult]:
        """
        Run complete evaluation suite.
        """
        self.results['gsm8k'] = self.run_gsm8k()
        self.results['truthfulqa'] = self.run_truthfulqa()
        self.results['defects4j'] = self.run_defects4j()
        self.results['humaneval'] = self.run_humaneval()
        return self.results
```

#### Deliverable:
- [ ] Unified benchmark runner
- [ ] All four benchmarks configured
- [ ] Automated result collection
- [ ] CSV/JSON result export

---

### 5.2 Novel Evaluation Metrics

**Goal**: Implement novel metrics for uncertainty-guided repair

#### Tasks:
- [ ] Implement uncertainty-calibration curve
- [ ] Implement repair efficiency metric
- [ ] Implement strategy accuracy correlation
- [ ] Implement granularity localization metric
- [ ] Create metric visualization utilities

#### Implementation Details:
```python
# Location: src/token_self_repair/evaluation/metrics.py

class UncertaintyCalibrationCurve:
    """
    Plot uncertainty vs actual error rate to measure calibration.
    """
    
    def compute(
        self,
        predictions: list[str],
        ground_truth: list[str],
        uncertainty_scores: list[float]
    ) -> CalibrationResult:
        """
        Bin predictions by uncertainty and compute error rate per bin.
        
        Perfect calibration: uncertainty = error_rate
        """
        pass

class RepairEfficiencyMetric:
    """
    Measure patches generated per successful fix.
    Lower is better (more efficient).
    """
    
    def compute(
        self,
        repair_attempts: list[RepairAttempt]
    ) -> float:
        """
        Total patches generated / Total successful fixes
        """
        pass

class StrategyAccuracyCorrelation:
    """
    Measure if strategy selection aligns with optimal strategy.
    """
    
    def compute(
        self,
        selected_strategies: list[RepairStrategy],
        actual_successes: list[bool],
        uncertainty_types: list[tuple[float, float]]  # (EU, AU) pairs
    ) -> CorrelationResult:
        """
        Compute correlation between:
        - High EU → Exploration success rate
        - High AU → Refinement success rate
        """
        pass

class GranularityLocalizationMetric:
    """
    Measure distance between uncertain regions and actual bug locations.
    """
    
    def compute(
        self,
        uncertainty_hotspots: list[int],  # Line numbers
        actual_bug_lines: list[int]
    ) -> LocalizationResult:
        """
        Metrics:
        - Top-1 accuracy: Is top uncertain line the bug?
        - Top-5 accuracy: Is bug in top 5 uncertain lines?
        - Mean rank: Average rank of bug lines in uncertainty ranking
        """
        pass
```

#### Deliverable:
- [ ] All four novel metric classes
- [ ] Visualization functions for each metric
- [ ] Integration with benchmark runner
- [ ] Comparison utilities (baseline vs uncertainty-guided)

---

### 5.3 Ablation Study Framework

**Goal**: Compare different system configurations to measure contribution of each component

#### Tasks:
- [ ] Implement baseline configuration (no uncertainty)
- [ ] Implement detection-only configuration (binary uncertainty)
- [ ] Implement strategy-guided configuration (our full system)
- [ ] Implement ranking-enhanced configuration (with patch ranking)
- [ ] Create automated comparison runner
- [ ] Generate statistical significance tests

#### Implementation Details:
```python
# Location: src/token_self_repair/evaluation/ablation.py

class AblationStudy:
    """
    Systematic comparison of system configurations.
    """
    
    def __init__(self, benchmark_runner: UnifiedBenchmarkRunner):
        self.runner = benchmark_runner
        self.configurations = {}
    
    def setup_configurations(self):
        """
        Define all ablation configurations.
        """
        # Baseline: RepairAgent without uncertainty
        self.configurations['baseline'] = {
            'use_uncertainty': False,
            'use_strategy_selection': False,
            'use_patch_ranking': False
        }
        
        # Detection-Only: Binary uncertainty trigger
        self.configurations['detection_only'] = {
            'use_uncertainty': True,
            'use_strategy_selection': False,
            'use_patch_ranking': False
        }
        
        # Strategy-Guided: Dynamic strategy selection
        self.configurations['strategy_guided'] = {
            'use_uncertainty': True,
            'use_strategy_selection': True,
            'use_patch_ranking': False
        }
        
        # Full System: All components enabled
        self.configurations['full_system'] = {
            'use_uncertainty': True,
            'use_strategy_selection': True,
            'use_patch_ranking': True
        }
    
    def run_ablation(
        self,
        benchmark: str,
        num_samples: int = 50
    ) -> AblationResult:
        """
        Run all configurations on specified benchmark.
        """
        results = {}
        
        for config_name, config_params in self.configurations.items():
            # Configure system
            self._apply_configuration(config_params)
            
            # Run benchmark
            if benchmark == 'defects4j':
                result = self.runner.run_defects4j()
            elif benchmark == 'gsm8k':
                result = self.runner.run_gsm8k(num_samples)
            # ... etc
            
            results[config_name] = result
        
        return AblationResult(results)
    
    def compute_statistical_significance(
        self,
        ablation_result: AblationResult
    ) -> SignificanceTest:
        """
        Run paired t-tests between configurations.
        """
        pass
```

#### Deliverable:
- [ ] Ablation study framework
- [ ] All four configurations defined
- [ ] Statistical significance testing
- [ ] Automated result table generation

---

## Phase 6: Visualization and XAI

### 6.1 Uncertainty Heatmap Visualization

**Goal**: Create real-time color-coded visualization of uncertainty during generation

#### Tasks:
- [ ] Implement token-level heatmap rendering
- [ ] Add color coding for AU vs EU
- [ ] Create interactive hover tooltips
- [ ] Implement line-level highlighting
- [ ] Add method-level overview panel

#### Implementation Details:
```python
# Location: src/token_self_repair/visualization/heatmap.py

class UncertaintyHeatmap:
    """
    Interactive visualization of token-level uncertainty.
    
    Color scheme:
    - RED: High epistemic (knowledge gap)
    - YELLOW: High aleatoric (ambiguous)
    - ORANGE: Both high
    - GREEN: High confidence
    - BLUE: Previously repaired
    """
    
    def render_code_with_uncertainty(
        self,
        code: str,
        uncertainty_map: UncertaintyMap,
        output_format: str = 'html'
    ) -> str:
        """
        Generate HTML/terminal output with color-coded tokens.
        """
        pass
    
    def create_interactive_view(
        self,
        code: str,
        uncertainty_map: UncertaintyMap
    ) -> plotly.Figure:
        """
        Create interactive Plotly visualization.
        
        Features:
        - Hover to see exact AU/EU values
        - Click line to see aggregated scores
        - Highlight hotspots
        """
        pass
```

#### Deliverable:
- [ ] Heatmap rendering class
- [ ] HTML export for reports
- [ ] Interactive Plotly visualization
- [ ] Terminal-friendly color output

---

### 6.2 Repair Trajectory Visualization

**Goal**: Show evolution of uncertainty across repair iterations

#### Tasks:
- [ ] Implement trajectory tracking data structure
- [ ] Create iteration-by-iteration visualization
- [ ] Add uncertainty convergence plots
- [ ] Implement strategy transition diagrams
- [ ] Create animated repair evolution

#### Implementation Details:
```python
# Location: src/token_self_repair/visualization/trajectory.py

class RepairTrajectoryVisualizer:
    """
    Visualize how uncertainty evolves during repair process.
    """
    
    def plot_uncertainty_evolution(
        self,
        repair_history: list[RepairIteration]
    ) -> matplotlib.Figure:
        """
        Line plot showing AU and EU over iterations.
        
        Shows:
        - How uncertainty decreases (hopefully)
        - Which regions converged
        - Where system got stuck
        """
        pass
    
    def create_strategy_flow_diagram(
        self,
        repair_history: list[RepairIteration]
    ) -> networkx.DiGraph:
        """
        Show strategy transitions as a graph.
        
        Example:
        EXPLORATION → REFINEMENT → HYBRID → Success
        """
        pass
    
    def animate_repair_process(
        self,
        repair_history: list[RepairIteration]
    ) -> Animation:
        """
        Create animated visualization showing:
        1. Initial uncertain code
        2. Strategy selection
        3. Patch generation
        4. Test execution
        5. Uncertainty update
        6. Next iteration
        """
        pass
```

#### Deliverable:
- [ ] Trajectory visualization class
- [ ] Evolution plots
- [ ] Strategy flow diagrams
- [ ] Animated repair visualization

---

### 6.3 Interactive Explanation Interface

**Goal**: Allow developers to query the system about repair decisions

#### Tasks:
- [ ] Implement question-answering interface
- [ ] Create decision explanation generator
- [ ] Add rationale retrieval for patch selections
- [ ] Implement "what-if" scenario analysis
- [ ] Create natural language explanation module

#### Implementation Details:
```python
# Location: src/token_self_repair/visualization/explainer.py

class InteractiveExplainer:
    """
    Answer developer questions about repair decisions.
    """
    
    def __init__(
        self,
        repair_history: list[RepairIteration],
        uncertainty_maps: list[UncertaintyMap]
    ):
        self.history = repair_history
        self.uncertainty = uncertainty_maps
    
    def explain_patch(self, iteration: int, line_num: int) -> str:
        """
        Generate natural language explanation for patch.
        
        Example query: "Why did you change line 42?"
        
        Response:
        "Line 42 had epistemic uncertainty score 0.87, indicating I 
        lacked similar training examples. The original code used a 
        pattern I haven't seen often. I searched for alternative 
        null-check patterns and found the current approach had only 
        0.23 uncertainty (high confidence). Additionally, this pattern 
        appeared in 3 similar bug fixes in my training data."
        """
        pass
    
    def explain_strategy_choice(self, iteration: int) -> str:
        """
        Explain why a particular strategy was chosen.
        """
        pass
    
    def what_if_analysis(
        self,
        alternative_threshold: float
    ) -> WhatIfResult:
        """
        Simulate what would have happened with different parameters.
        
        "What if we used EU threshold = 0.5 instead of 0.6?"
        """
        pass
```

#### Deliverable:
- [ ] Interactive explainer class
- [ ] Natural language generation for explanations
- [ ] What-if analysis capability
- [ ] Integration with Streamlit UI

---

## Phase 7: Documentation and Packaging

### 7.1 Technical Documentation

**Goal**: Create comprehensive documentation for the system

#### Tasks:
- [ ] Write API documentation for all classes
- [ ] Create architecture diagrams
- [ ] Write tutorial notebooks
- [ ] Document configuration options
- [ ] Create troubleshooting guide

#### Deliverable:
- [ ] Complete API docs (Sphinx or MkDocs)
- [ ] Architecture documentation
- [ ] 3 tutorial Jupyter notebooks
- [ ] Configuration reference

---

### 7.2 Research Artifact Preparation

**Goal**: Package for reproducibility and publication

#### Tasks:
- [ ] Create Docker container with all dependencies
- [ ] Write reproduction scripts for all experiments
- [ ] Create dataset download scripts
- [ ] Package pre-trained models
- [ ] Write detailed README

#### Deliverable:
- [ ] Docker container
- [ ] Complete reproduction package
- [ ] zenodo.org archive
- [ ] GitHub release

---

### 7.3 Technical Report Writing

**Goal**: Document research contributions for publication

#### Tasks:
- [ ] Write abstract and introduction
- [ ] Document novel algorithms
- [ ] Create result tables and figures
- [ ] Write related work section
- [ ] Prepare supplementary materials

#### Deliverable:
- [ ] Technical report (10-15 pages)
- [ ] All figures and tables
- [ ] Supplementary experiments
- [ ] Camera-ready PDF

---

## Success Criteria

### Quantitative Targets

- [ ] **Defects4J Improvement**: 15-25% increase in fix rate over baseline RepairAgent
- [ ] **Repair Efficiency**: 30-40% reduction in patches generated per successful fix
- [ ] **Uncertainty Calibration**: AUROC > 0.75 for uncertainty vs error correlation
- [ ] **Calibration Quality**: ECE < 0.08 (expected calibration error)
- [ ] **GSM8K Improvement**: 10-15% accuracy improvement with uncertainty-guided repair
- [ ] **TruthfulQA Improvement**: 15-20% accuracy improvement

### Qualitative Targets

- [ ] Uncertainty heatmaps correctly identify bug locations in >70% of cases
- [ ] Strategy selection aligns with manual expert analysis in >80% of cases
- [ ] Patch rankings correlate with eventual success (Spearman ρ > 0.6)
- [ ] Generated explanations are rated as helpful by developers
- [ ] System is easy to configure and extend

---

## Research Contributions Summary

### Primary Contributions

- [ ] **C1**: First work to use aleatoric/epistemic uncertainty decomposition for program repair
- [ ] **C2**: Novel multi-granularity uncertainty propagation framework (token → line → method → test)
- [ ] **C3**: Uncertainty-driven patch ranking metric combining pre-execution confidence and post-execution results
- [ ] **C4**: Dynamic repair strategy selection based on uncertainty type
- [ ] **C5**: Unified framework handling both reasoning and program repair with uncertainty principles

### Publication Targets

- [ ] ICSE 2026 (Software Engineering)
- [ ] NeurIPS 2026 (ML for Code)
- [ ] ICLR 2026 (Agents & Planning)
- [ ] ASE 2026 (Automated Software Engineering)

---

## Resource Requirements

### Compute
- [ ] 1x GPU (A100 40GB or H100 80GB recommended)
- [ ] 32GB+ system RAM
- [ ] 200GB+ storage (models + datasets + results)

### Software
- [ ] Python 3.10+
- [ ] PyTorch 2.0+
- [ ] Transformers 4.35+
- [ ] ANTLR4 (Java parser)
- [ ] Defects4J framework

### Data
- [ ] Llama-2-7b-chat-hf model (~14GB)
- [ ] Llama-3-8B-Instruct model (~16GB)
- [ ] GSM8K dataset
- [ ] TruthfulQA dataset
- [ ] Defects4J bugs (subset of 20)
- [ ] HumanEval dataset

---

## Progress Tracking

### Quick Status Overview

- [ ] Phase 1: Core Infrastructure (0/3 sections complete)
- [ ] Phase 2: RepairAgent Integration (0/2 sections complete)
- [ ] Phase 3: Dynamic Strategy Selection (0/2 sections complete)
- [ ] Phase 4: Patch Ranking (0/2 sections complete)
- [ ] Phase 5: Evaluation Framework (0/3 sections complete)
- [ ] Phase 6: Visualization and XAI (0/3 sections complete)
- [ ] Phase 7: Documentation (0/3 sections complete)

**Overall Progress: 0/18 major sections complete**

---

## Notes

This is a living document. Update checkboxes as tasks are completed. Add notes, challenges, and insights as you progress through the implementation.


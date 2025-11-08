
10-Day Prototype Plan: Token-Level Uncertainty-Driven Self-Repair


1.0 Executive Summary: The Proactive Repair Architecture


1.1 Project Mandate & Core Innovation

This document outlines a 10-day sprint plan to build a Minimum Viable Prototype (MVP) of an agentic workflow capable of "token-level uncertainty-driven self-repair". The 3-person team, as specified in the project charter , will focus on reasoning benchmarks (GSM8K, TruthfulQA) and create a system that layers on existing agentic framework principles.
The core innovation mandated by this project is a strategic shift from reactive error handling to proactive self-correction. Current self-healing pipelines, for example, often trigger repairs only after a catastrophic failure, such as a code crash or a failed test case.2 This prototype will implement a more intelligent system that monitors its own token-level confidence as it generates a response. It will learn to "know what it doesn't know" at the token level, proactively pausing to re-evaluate and refine its reasoning before committing to a low-confidence (and likely incorrect) answer.

1.2 The Generator-Critic Pattern as the Self-Repair Loop

The "Self-Repair Loop" and "iterative refinement" specified in the technical approach will be implemented using a robust, industry-standard agentic design pattern: the "Generator-Critic" 4 or "Reflection" loop.6
This architectural choice de-risks the project by adopting a proven control-flow mechanism instead of inventing a new one. The Agent Architect (P2) can leverage extensive public documentation on building reflection loops in modern frameworks 8, dramatically accelerating development.
In this architecture, the loop will consist of two primary nodes:
Generator: The primary LLM call that produces the initial reasoning steps for a benchmark query.
Critic/Reflector: A second LLM call, acting as a "reflector".6 This node is not triggered by a failed tool, but by the novel token-level uncertainty metric. It is prompted to critique the first answer (e.g., "The confidence for token 'X' was low. Re-evaluate this step.") and feeds this critique back to the Generator for another attempt.

1.3 Core Architectural Pillars (The 3-Person Team)

The prototype's architecture maps directly to the three-person team structure defined in the project brief :
Pillar 1: Uncertainty Engine (P1: Sai Kavya Marthala)
A standalone, model-agnostic Python library that abstracts the complexity of bimodal logit/logprob extraction. It will be responsible for ingesting raw model outputs and calculating the token-level uncertainty metric (LogTokU).
Pillar 2: Agentic Chassis (P2: Aryan Tapkire)
A stateful graph built using LangGraph.12 This StateGraph 14 orchestrates the Generator-Critic loop, manages the agent's state (e.g., repair_attempts), and implements the conditional routing logic that triggers the self-repair.4
Pillar 3: Evaluation & XAI (P3: Deepak Sai Pendyala)
A set of evaluation harnesses for GSM8K and TruthfulQA 16 and a real-time streaming interface. This interface will consume and display the "transparent uncertainty indicators" and "real-time notifications" as the agent works.

1.4 Architectural Data Flow Diagram

The flow of data and control through the agent chassis is the "API contract" between the three developers, ensuring successful integration. This diagram outlines the precise data handling for a single query.
Table 1: Architectural Data Flow
Step
Component
Action
Data
1
User
Submit Query
{"question": "What is...?"}
2
LangGraph Entry
Start Graph
AgentState(question=...)
3
Node 1: Generator
Call LLM (OpenAI or Local) with logprobs=True or output_scores=True.
AgentState(generation="...")
4
Node 1: Uncertainty
Call P1's Library on the raw logits/logprobs.
AgentState(token_uncertainties=[...], avg_uncertainty=0.7)
5
Conditional Edge
route_on_uncertainty(state)
if state['avg_uncertainty'] > THRESHOLD:
6a
Node 2: Reflector
(If High Uncertainty) Call LLM with a "critique" prompt.
{"critique": "Previous answer was uncertain. Re-think step 3."}
7a
Edge
Route back to Generator with new context.
AgentState(messages=[..., critique_message])
6b
Node 3: XAI_Message
(If Low Uncertainty) Set final status.
AgentState(xai_message="High confidence response.")
7b
Edge
Route to END.
END
8
LangGraph Exit
Return final state.
{"generation": "...", "xai_message": "..."}


2.0 Foundation: Prototype Stack & Component Deep Dive


2.1 The Agent Chassis: LangGraph (P2)

Justification: The project explicitly requires a "Self-Repair Loop" and a "ControlFlow" architecture. Analysis of available agent frameworks shows a clear consensus: for building stateful, cyclic graphs (i.e., loops), LangGraph is the industry standard.12
Simpler frameworks like LangChain are designed for linear "chains" and struggle with complex, conditional loops.13 Other frameworks like AutoGen or CrewAI are optimized for multi-agent collaboration, which is overly complex for this 10-day, single-agent prototype.19 LangGraph is designed explicitly for this use case. Its two core primitives, the StateGraph 14 and add_conditional_edges 4, are the exact components needed to implement the "Generator-Critic-via-Uncertainty" loop.15
Implementation: The AgentState
The first task for the Agent Architect (P2) is to define the graph's "memory." This TypedDict serves as the single source of truth that persists throughout the agent's execution.15
Table 2: AgentState Definition

Field
Type
operator.add
Description
messages
Annotated, operator.add]
Yes
Full conversation history, including critiques.
question
str
No
The initial query (e.g., from GSM8K).
generation
str
No
The most recent generated answer.
token_uncertainties
list[float]
No
Per-token uncertainty scores from P1's lib.
avg_uncertainty
float
No
The aggregate score used for routing.
repair_attempts
int
No
A counter to prevent infinite loops.6
xai_message
str
No
The human-readable status message for P3's UI.


2.2 The Uncertainty Engine (P1)


2.2.1 The Unified Interface (The Abstraction Contract)

The Uncertainty Engineer (P1) will deliver a UncertaintyMonitor class, not a simple script. This API-driven approach abstracts the backend (OpenAI vs. Local) and allows the Agent Architect (P2) to remain model-agnostic.
The "Proxy-First" Mitigation Strategy:
The LogTokU framework 1 is the project's most complex and high-risk component. To prevent P1's research from blocking P2's development, this plan mandates a "proxy-first" delivery.
By Day 3, P1 will first deliver a simple, well-understood uncertainty metric: Shannon Entropy.24 This is a stable, known calculation:
Take raw logits.27
Convert to probabilities: probs = scipy.special.softmax(logits, axis=-1).28
Calculate entropy: entropy = scipy.stats.entropy(probs, axis=-1).24
This proxy allows P2 to build and test the entire agent loop against a functional uncertainty score. P1 can then spend Days 4-6 implementing the superior LogTokU algorithm, which P2 can integrate as a simple one-line function swap. This parallel-path development is the sprint's primary risk-mitigation strategy.
Table 3: UncertaintyMonitor Class API (v1.0)

Method
Input
Output
Description
get_logprobs_from_openai(api_response)
ChatCompletion
list[tuple[str, float]]
Parses the logprobs object.29
get_logits_from_transformers(outputs)
GenerateDecoderOnlyOutput
torch.Tensor
Extracts outputs.scores.27
logits_to_probs(logits)
torch.Tensor
np.ndarray
scipy.special.softmax(logits, axis=-1).28
calculate_entropy(probs)
np.ndarray
list[float]
scipy.stats.entropy(probs, axis=-1).24 ****
calculate_logtoku(logits)
torch.Tensor
list[float]
The "real" LogTokU implementation.1 ****


2.2.2 Backend 1: OpenAI logprobs

Implementation is straightforward. P1 will write a helper function that enables logprobs=True and top_logprobs=5 in the client.chat.completions.create call.29 The parser will then iterate through response.choices.logprobs.content to extract the chosen token and its associated log-probability.30

2.2.3 Backend 2: Local Model logits

A critical decision for local models is the trade-off between inference speed (vLLM) and development speed (transformers). vLLM is high-performance but requires deep modification of its CUDA-level sampler to extract full logits for every token, a high-risk task.32
For a 10-day sprint, development speed is paramount. P1 will use the standard Hugging Face transformers library. The model.generate() method provides the required data out-of-the-box by setting return_dict_in_generate=True and output_scores=True.27 This returns a GenerateDecoderOnlyOutput object. The outputs.scores attribute will be a tuple of tensors (one per generated token), which can be stacked into a single tensor and passed to the UncertaintyMonitor.38

2.3 The XAI Streaming Interface (P3)

The project requires "transparent uncertainty indicators" and "real-time notifications" , such as "Moderate uncertainty detected - refining answer". This message is not the final answer; it is an intermediate status update from inside the agent's loop.
A standard graph.stream() method, which streams the full state 39, is too verbose for this. The correct implementation is for P2 (Agent) to use the get_stream_writer() function within the agent nodes to write() custom dictionaries (e.g., writer.write({"xai_message": "Refining..."})).41
P3 will build a UI (e.g., in Streamlit) that consumes the graph using graph.astream(..., stream_mode="custom").41 The UI will listen for these custom chunks and display only the content of the xai_message key. This provides a clean, real-time channel directly from the agent's internal reasoning state to the user interface.

2.4 The Evaluation Harness (P3)

Datasets:
GSM8K: P3 will load the main configuration using datasets.load_dataset("openai/gsm8k", "main").43
TruthfulQA: P3 will load the multiple_choice configuration using datasets.load_dataset("EleutherAI/truthful_qa_mc", "multiple_choice").45
Metrics:
GSM8K: The metric will be exact string match on the final numerical answer. This is the simple, standard evaluation for this benchmark.16
TruthfulQA: The metric will be MC1 (multi-choice, 1-true-answer) accuracy. This is simpler to implement than MC2 and sufficient for the sprint's goals.17
The final deliverable on Day 10 will be a "Report Card" quantifying the prototype's value.
Table 4: Final Evaluation Report-Card (Deliverable for Day 10)
Benchmark
Agent
Metric
Accuracy
GSM8K
Baseline (No Repair)
Exact Match
X %
GSM8K
RepairAgent (v1.0)
Exact Match
Y %
TruthfulQA
Baseline (No Repair)
MC1 Accuracy
A %
TruthfulQA
RepairAgent (v1.0)
MC1 Accuracy
B %
Result
GSM8K Improvement
(Y - X)
Z %
Result
TruthfulQA Improvement
(B - A)
C %


3.0 The 10-Day Sprint: Phased Execution Plan


3.1 Phase 1 (Days 1-3): Setup & Core Component Isolation (Parallel)

Goal: All three team members work in parallel to build their core components in isolation. No developer is blocked.
P1 (Uncertainty): Builds UncertaintyMonitor-v0.1 (the Proxy version) and demonstrates logit/logprob extraction from both OpenAI 30 and local transformers 27 backends.
P2 (Agent): Builds a non-looping StateGraph 14 and defines the AgentState.15
P3 (Evaluation): Builds the full evaluation harness for both benchmarks.16
Milestone (Day 3): P1 delivers the UncertaintyMonitor-v0.1 (with entropy proxy) to P2. P3 delivers the baseline evaluation scripts.

3.2 Phase 2 (Days 4-7): Integration & Loop Implementation (Sequential)

Goal: Assemble the components into the functioning, end-to-end self-repair loop.
Day 4-5: P2 integrates P1's entropy proxy and builds the conditional "repair loop" using add_conditional_edges.4 P3 builds the XAI streaming UI using stream_mode="custom".41
Day 6-7: P1 delivers UncertaintyMonitor-v1.0 (with LogTokU).1 P2 swaps the proxy for the real metric. The team runs the first successful end-to-end test on a single data point.
Milestone (Day 7): A single query from GSM8K is fed into the agent, which successfully triggers the (LogTokU-driven) repair loop, streams an "Refining..." message to P3's UI, and produces a final answer.

3.3 Phase 3 (Days 8-10): Evaluation, Calibration, & Report-Out (Full Team)

Goal: Prove the prototype works and quantify its impact.
Day 8: Full benchmark run (Baseline vs. Repair) on both GSM8K and TruthfulQA.
Day 9: Calibration Day. The team analyzes the Day 8 results and "tunes" the uncertainty threshold (e.g., TRIGGER_THRESHOLD = 0.5) to optimize for accuracy vs. computational cost. They re-run the benchmarks.
Milestone (Day 10): Final "Report Card" (Table 4) is generated. Code is cleaned, documented, and a final demo is prepared.

4.0 Detailed Daily Project Plan & Workstream Allocation

Table 5: 10-Day, 3-Track Project Gantt

Day
P1: Uncertainty Engineer (Sai)
P2: Agent Architect (Aryan)
P3: Evaluation & XAI (Deepak)
Integration Point / Milestone
1
Setup: Provision GPU env. OpenAI API: Write get_openai_logprobs.py script.29 Local Models: Setup transformers with Llama/Mistral.
Setup: Provision Python env. LangGraph: Build "Hello World" StateGraph.14 Framework Research: Review "Generator-Critic" 4 & "Reflection" 6 patterns.
Setup: Provision Python env. Dataset Loading (GSM8K): Write load_gsm8k.py using datasets.43 Dataset Loading (TQA): Write load_truthfulqa.py.45
Milestone: All environments provisioned. All assets (models, datasets) are locally accessible.
2
Local Logits: Write get_local_logits.py using model.generate(output_scores=True).27 Uncertainty Lib (v0.1): Design UncertaintyMonitor class skeleton.
Agent State: Define the AgentState TypedDict.15 Graph (v0.1): Build a 2-node (Generate, Finalize) linear graph. Prompting: Write v1_generator_prompt.txt.
Eval Harness (GSM8K): Write evaluate_gsm8k.py. Implement exact match scoring.16 Baseline Run: Run a baseline gpt-3.5-turbo (no agent) on GSM8K.
P1 provides P2 with sample logprobs and logits output files. P2 uses these to design the AgentState.
3
Uncertainty Proxy: Implement calculate_entropy() using scipy.special.softmax 28 and scipy.stats.entropy.24 Deliverable: UncertaintyMonitor-v0.1 (with proxy) pushed to Git.
Prompting: Write v1_reflector_prompt.txt based on "Constitutional AI" and "Reflection" 6 principles. Graph (v0.2): Build the 3-node (Generate, Reflect, Finalize) graph skeleton.
Eval Harness (TQA): Write evaluate_truthfulqa.py. Implement MC1 scoring.17 Baseline Run: Run baseline on TruthfulQA.
MILESTONE (Proxy Delivery): P1 delivers UncertaintyMonitor-v0.1. P2 can now integrate.
4
LogTokU Research: Begin implementation of the "Logits-induced Token Uncertainty" (LogTokU) framework.1 Refactor: Clean up get_local_logits.py to be a robust class method.
First Integration: P2 integrates P1's v0.1 proxy lib. Graph (v0.3): The "Generate" node now calls the LLM, then immediately calls monitor.calculate_entropy() and populates AgentState.avg_uncertainty.
XAI UI (v0.1): Build a simple Streamlit/Chainlit app. Streaming: Implement a basic astream() loop that prints the full AgentState at each step.39
P2's graph now calculates uncertainty on every run, viewable in P3's simple UI.
5
LogTokU Implementation: Continue developing the core LogTokU algorithm. This is now the critical path for P1.
The Loop: Implement add_conditional_edges.4 The graph now loops back to "Reflect" and "Generate" if avg_uncertainty > THRESHOLD. Graph (v1.0): The core repair loop is functional.
XAI UI (v0.2): Implement get_stream_writer() 41 in the Streamlit app. Custom Stream: Modify the UI to listen to astream(stream_mode="custom") 42 and only print xai_message events.
MILESTONE (Core Loop): The complete agent logic (using the proxy) is functional.
6
LogTokU Implementation: Finalize, test, and debug the calculate_logtoku() function. Deliverable: UncertaintyMonitor-v1.0 (with real LogTokU) pushed to Git.
Second Integration: P2 adds get_stream_writer() calls to the graph nodes.41 Reflect node: writer.write({"xai_message": "Uncertainty detected. Refining..."}) Finalize node: writer.write({"xai_message": "Confidence: High"})
Third Integration: P3's UI (from Day 5) now successfully connects to P2's custom stream and displays the real-time XAI messages. Test: P3 tests the UI by watching P2 run the agent.
P2 and P3 integrate the XAI streaming. P3's UI shows the agent "thinking".
7
Support & Calibration: Support P2 in swapping the metric. Analysis: Begin analyzing which tokens receive high LogTokU scores on GSM8K examples.
The Swap: P2 changes one line of code: monitor.calculate_entropy() becomes monitor.calculate_logtoku(). End-to-End Test: Run the full prototype (Local Model + LogTokU + Repair Loop + XAI) on a single GSM8K and TruthfulQA question. Debug.
Support & Test: P3 provides the test questions and validates that the XAI messages match the agent's behavior. Bug Bash: P3 leads a bug-hunting session on the E2E prototype.
MILESTONE (E2E Test): First end-to-end success of the full, intended architecture.
8
Benchmark Run: P1 provides support for the local model evaluation, ensuring the GPU env is stable and logits are captured correctly.
Benchmark Run: P2 scripts the BaselineAgent (loop disabled) and RepairAgent (loop enabled). P2 and P3 launch the full benchmark runs overnight.
Benchmark Run: P3 triggers the evaluation scripts (evaluate_gsm8k.py, evaluate_truthfulqa.py) against both agent versions.
Task: Full evaluation runs launched.
9
Results Analysis: P1 analyzes why LogTokU failed/succeeded. Was it epistemic or aleatoric uncertainty? Prepare insights for the final report.
Calibration: P2 analyzes the avg_uncertainty scores from the Day 8 run. Tune: P2 adjusts the TRIGGER_THRESHOLD variable. Re-run: P2 and P3 re-run the entire benchmark suite with the new threshold.
Results Analysis: P3 ingests the results from Day 8 and creates the first draft of the "Final Report Card" table. Re-run: P3 supports P2 in launching the final benchmark run.
Calibration Day: The team collectively analyzes the first run and tunes the system.
10
Documentation: Clean up and fully document the UncertaintyMonitor library. Final Report: Provide "Why" analysis for the final report.
Code Freeze & Cleanup: Clean up the LangGraph code, document the AgentState and node logic. Demo Prep: Prepare a live demo of the E2E system.
Final Report: P3 generates the final "Report Card" (Table 4). Demo Prep: P3 prepares the Streamlit UI for the final demo, showing the XAI messages.
MILESTONE (Sprint Complete): Final demo ready. Final report data is generated.


5.0 Integration, Risks, and Mitigation Strategy


5.1 Key Integration Points

Day 3 (Proxy Delivery): P1 delivers UncertaintyMonitor-v0.1 (with entropy) to P2.
Day 4 (Proxy Integration): P2 integrates the proxy. This is the first time the agent calculates uncertainty.
Day 6 (Streaming Integration): P2's graph and P3's UI connect on the custom streaming channel.41
Day 7 (The Swap): P2 swaps P1's v0.1 proxy for the v1.0 LogTokU library. This is the final E2E integration.

5.2 Identified Risks & Mitigation

Risk 1: LogTokU (P1) is a Research Bottleneck.
Description: The LogTokU framework 1 is a complex, research-level component.48 P1's implementation could stall, blocking the entire sprint.
Mitigation (The "Proxy-First" Strategy): As detailed in Section 2.2.1 and the Day 3 plan, P1 first delivers a simpler, known proxy (Shannon Entropy).24 This guarantees P2 can build the complete agent loop in parallel, de-coupling the team's dependencies. This is the sprint's most critical risk-mitigation strategy.
Risk 2: Local Model Inference is a Performance Bottleneck.
Description: Iteratively running model.generate(output_scores=True) 27 on a local GPU for a full benchmark 44 will be extremely slow, burning valuable sprint time.
Mitigation (The "API-First" Strategy): The team will develop and debug the agent loop primarily against the fast OpenAI logprobs API 30 from Day 1-6. This allows for rapid iteration on the loop logic. Local model support 27 is treated as a "backend" to be integrated on Day 7, with the full (and slow) benchmark run happening overnight on Day 8.
Risk 3: "Framework-Hell" and Scope Creep.
Description: The user query mentions RepairAgent 1 and Self-Healing LLM Pipeline.1 P2 could waste days trying to integrate these disparate libraries.
Mitigation (Clarification of Scope): This plan explicitly rejects direct integration with these libraries for the 10-day sprint. RepairAgent 49 is used as design inspiration for its FSM-like loop. ControlFlow is treated as a concept that is implemented by LangGraph.12 This keeps P2 focused on a single, powerful framework.
Risk 4: Repair Loops Cause Hallucination Loops.
Description: A "Reflect" node that just re-prompts ("try again") may get stuck in a loop or amplify hallucinations.7
Mitigation (Guided Reflection): P2's "Reflector" prompt 6 will not be a simple "try again." It will be an explicit critique prompt, as specified in the technical approach ("constitutional AI-style rule-based corrections," ). It will be prompted: "You generated the previous answer. The uncertainty score was highest at token [X]. This implies a factual/reasoning error. Re-evaluate your logic at that specific point and generate a corrected solution."

6.0 Post-Sprint Roadmap: The "Day 11" Plan

6.1 Performance: Migrating to vLLM:
The transformers 27 prototype is functional but slow. The clear next step is to port the local inference backend to a high-throughput engine like vLLM. This is now a well-defined engineering task, though it will require modifying the vLLM sampler to expose the full logits for every token 32, providing a clear path to production performance.
6.2 Smarter Repair: Tool-Using Reflectors:
The v1.0 "Reflector" node is purely conversational.6 The v2.0 "Reflector" should become a tool-using agent. For GSM8K, instead of just thinking about the math, it should be able to execute the uncertain step (e.g., "18 + 6") in a Python REPL tool 9 to validate its correction. This mirrors the true agentic power of RepairAgent.49
6.3 Systematic Calibration:
Day 9 ("Calibration Day") is an ad-hoc process. A follow-on sprint is required to systematically sweep the TRIGGER_THRESHOLD parameter across different models and tasks (GSM8K vs. TruthfulQA) to build a proper "Receiver Operating Characteristic" (ROC) curve, optimizing the trade-off between accuracy-gain and computational-cost.
6.4 Advanced Evaluation: Meta-Reasoning:
The current evaluation 16 only measures if the final answer is right. A future evaluation should adopt a "meta-reasoning" benchmark 52 to determine if the agent knows why its first answer was wrong, thereby testing the fidelity of the LogTokU trigger itself.

7.0 Appendix: Core Code Skeletons (The "Starter Kit")


A.1: Local Logit Extractor (transformers) - P1, Day 2


Python


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

def get_local_logits(prompt: str) -> tuple:
    """
    Generates text and returns the stacked logits for *generated tokens*.
    
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Use generate with output_scores=True to get per-token logits
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        return_dict_in_generate=True,
        output_scores=True  # This is the key parameter
    )
    
    # outputs.scores is a tuple of tensors, one for each new token
    # Stack them into a single tensor: (num_tokens, vocab_size)
    all_logits = torch.stack(outputs.scores, dim=0)
    
    generated_sequence = outputs.sequences[0, inputs.input_ids.shape:]
    
    return all_logits, generated_sequence




A.2: Uncertainty Proxy (Entropy) - P1, Day 3


Python


import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy

def calculate_entropy_from_logits(logits_tensor: torch.Tensor) -> np.ndarray:
    """
    Calculates Shannon entropy for each token from raw logits.
    This is the "Proxy-First" deliverable. [24, 25, 28]
    """
    # 1. Move to CPU and convert to numpy
    logits_np = logits_tensor.cpu().numpy()
    
    # 2. Convert logits to probabilities using softmax
    # axis=-1 applies softmax along the vocab dimension
    probabilities = softmax(logits_np, axis=-1)
    
    # 3. Calculate Shannon entropy along the vocab dimension
    # H = -sum(p * log(p))
    token_entropies = entropy(probabilities, base=2, axis=-1)
    
    return token_entropies



A.3: LangGraph AgentState Definition - P2, Day 2


Python


from typing import List, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Defines the persistent state of the agent graph.
    
    """
    # Use operator.add to append messages, not overwrite
    messages: Annotated, operator.add]
    
    # Fields to be overwritten at each step
    question: str
    generation: str
    token_uncertainties: List[float]
    avg_uncertainty: float
    xai_message: str
    
    # Field to be manually incremented
    repair_attempts: int



A.4: LangGraph "Generator-Critic" Skeleton - P2, Day 5


Python


from langgraph.graph import StateGraph, END, START
from langgraph.config import get_stream_writer
import random # Placeholder for real models

# Define the uncertainty threshold
UNCERTAINTY_THRESHOLD = 0.5

# --- Define Nodes ---
def generator_node(state: AgentState):
    """Generates an answer and calculates its uncertainty."""
    # 1. Call LLM (implementation omitted)
    generation = f"This is attempt {state['repair_attempts']}"
    
    # 2. Call P1's Uncertainty Library (using proxy)
    # Placeholder for real logits
    fake_logits = torch.randn(10, 5000) 
    # entropies = monitor.calculate_entropy(fake_logits)
    
    # 3. For the demo, fake the uncertainty
    avg_uncertainty = random.random() # Fake uncertainty
    
    return {
        "generation": generation,
        "avg_uncertainty": avg_uncertainty,
        "repair_attempts": state["repair_attempts"] + 1
    }

def reflector_node(state: AgentState):
    """Critiques the uncertain generation."""
    # P3's UI will see this message
    writer = get_stream_writer()
    writer.write({"xai_message": f"Uncertainty ({state['avg_uncertainty']:.2f}) detected. Refining..."})
    
    # This critique is added to 'messages'
    critique_message = HumanMessage(
        content=f"Attempt failed with uncertainty {state['avg_uncertainty']}. Re-evaluate."
    )
    return {"messages": [critique_message]}

# --- Define Conditional Edge ---
def route_on_uncertainty(state: AgentState) -> str:
    """The core repair loop logic."""
    if state["avg_uncertainty"] > UNCERTAINTY_THRESHOLD:
        if state["repair_attempts"] >= 3:
            return "end_with_failure"
        return "reflector" # Trigger the repair loop
    else:
        return "end_with_success"

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("generator", generator_node)
workflow.add_node("reflector", reflector_node)
workflow.add_node("end_with_success", lambda s: {"xai_message": "Confidence: High"})
workflow.add_node("end_with_failure", lambda s: {"xai_message": "Confidence: Low"})

workflow.set_entry_point("generator")
workflow.add_edge("reflector", "generator") # Loop back
workflow.add_edge("end_with_success", END)
workflow.add_edge("end_with_failure", END)

# Add the conditional router
workflow.add_conditional_edges(
    "generator",
    route_on_uncertainty,
    {"reflector": "reflector", "end_with_success": "end_with_success", "end_with_failure": "end_with_failure"}
) # [4, 6]

app = workflow.compile()



A.5: XAI Stream Consumer (Streamlit) - P3, Day 5


Python


import streamlit as st
import asyncio

# Assume 'app' is the compiled LangGraph from A.4
# from your_project.graph import app 

async def run_agent_and_stream_xai(question: str):
    """
    Consumes the agent's "custom" stream for XAI messages.
    
    """
    inputs = {"question": question, "messages":, "repair_attempts": 0}
    
    # Use acontainer to hold the real-time message
    xai_placeholder = st.empty()
    
    # Use stream_mode="custom" to get only the data we write
    async for chunk in app.astream(inputs, stream_mode="custom"):
        
        # Check if our custom key is in the chunk
        if "xai_message" in chunk:
            message = chunk["xai_message"]
            xai_placeholder.info(f"AGENT STATUS: {message}")

# Example of how to run in Streamlit
st.title("Uncertainty-Driven Agent")
if query := st.text_input("Enter your question:"):
    asyncio.run(run_agent_and_stream_xai(query))


Works cited
Token-Level Uncertainty-Driven Self-Repair in Agen - Google Docs.pdf
ammarlodhi255/self-healing-LLM-pipeline - GitHub, accessed November 6, 2025, https://github.com/ammarlodhi255/Self-healing-LLM-Pipeline
A Step by Step Guide to Building a Self Healing Pipeline with Ai - Break the Build, accessed November 6, 2025, https://www.breakthebuild.org/stop-test-failures-in-their-tracks-build-a-self-healing-playwright-pipeline-with-ai/
A Deep Dive into LangGraph for Self-Correcting AI Agents | ActiveWizards, accessed November 6, 2025, https://activewizards.com/blog/a-deep-dive-into-langgraph-for-self-correcting-ai-agents
LangGraph Tutorial: Build Your Own AI Coding Agent - Medium, accessed November 6, 2025, https://medium.com/@mariumaslam499/build-your-own-ai-coding-agent-with-langgraph-040644343e73
Reflection Agents - LangChain Blog, accessed November 6, 2025, https://blog.langchain.com/reflection-agents/
Self-Correcting AI Agents: How to Build AI That Learns From Its Mistakes - DEV Community, accessed November 6, 2025, https://dev.to/louis-sanna/self-correcting-ai-agents-how-to-build-ai-that-learns-from-its-mistakes-39f1
LangGraph: Building Self-Correcting RAG Agent for Code Generation - Learn OpenCV, accessed November 6, 2025, https://learnopencv.com/langgraph-self-correcting-agent-code-generation/
Building a Self-Correcting Coding Assistant with LangChain and LangGraph: A Hands-on Guide | by Anoop Maurya | Medium, accessed November 6, 2025, https://medium.com/@mauryaanoop3/building-a-self-correcting-coding-assistant-with-langchain-and-langgraph-a-hands-on-guide-3ea7424655be
Enhancing Code Quality with LangGraph Reflection - Analytics Vidhya, accessed November 6, 2025, https://www.analyticsvidhya.com/blog/2025/03/enhancing-code-quality-with-langgraph-reflection/
Reflection - GitHub Pages, accessed November 6, 2025, https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/
LangGraph - LangChain, accessed November 6, 2025, https://www.langchain.com/langgraph
LangChain vs LangGraph: A Developer's Guide to Choosing Your AI Frameworks - Milvus, accessed November 6, 2025, https://milvus.io/blog/langchain-vs-langgraph.md
LangGraph overview - Docs by LangChain, accessed November 6, 2025, https://docs.langchain.com/oss/python/langgraph/overview
Quickstart - Docs by LangChain, accessed November 6, 2025, https://docs.langchain.com/oss/python/langgraph/quickstart
GSM8K | DeepEval - The Open-Source LLM Evaluation Framework, accessed November 6, 2025, https://deepeval.com/docs/benchmarks-gsm8k
TruthfulQA | DeepEval - The Open-Source LLM Evaluation Framework, accessed November 6, 2025, https://deepeval.com/docs/benchmarks-truthful-qa
LangChain vs LangGraph vs LangSmith vs LangFlow: Key Differences Explained | DataCamp, accessed November 6, 2025, https://www.datacamp.com/tutorial/langchain-vs-langgraph-vs-langsmith-vs-langflow
Which agent framework is best to control python coding and execution agenta - Reddit, accessed November 6, 2025, https://www.reddit.com/r/AI_Agents/comments/1kvxodt/which_agent_framework_is_best_to_control_python/
Top 7 Free AI Agent Frameworks [2025] - Botpress, accessed November 6, 2025, https://botpress.com/blog/ai-agent-frameworks
Built with LangGraph! #9: Looping Graphs | by Okan Yenigün | Towards Dev - Medium, accessed November 6, 2025, https://medium.com/towardsdev/built-with-langgraph-9-looping-graphs-b689e42677d7
MaHuanAAA/logtoku - GitHub, accessed November 6, 2025, https://github.com/MaHuanAAA/logtoku
Estimating LLM Uncertainty with Logits - arXiv, accessed November 6, 2025, https://arxiv.org/html/2502.00290v2
entropy — SciPy v1.16.2 Manual, accessed November 6, 2025, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
Fastest way to compute entropy of each numpy array row? - Stack Overflow, accessed November 6, 2025, https://stackoverflow.com/questions/33607071/fastest-way-to-compute-entropy-of-each-numpy-array-row
How to Compute Entropy using SciPy? - GeeksforGeeks, accessed November 6, 2025, https://www.geeksforgeeks.org/machine-learning/how-to-compute-entropy-using-scipy/
[Announcement] Generation: Get probabilities for generated output - Hugging Face Forums, accessed November 6, 2025, https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
softmax — SciPy v1.16.2 Manual, accessed November 6, 2025, https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
Using logprobs | OpenAI Cookbook, accessed November 6, 2025, https://cookbook.openai.com/examples/using_logprobs
Beyond Words: Using Confidence Scores and Logprobs for Smart Prompt Engineering, accessed November 6, 2025, https://manangarg.medium.com/beyond-words-using-confidence-scores-and-logprobs-for-smart-prompt-engineering-fadc2ad18411
Utilities for Generation - Hugging Face, accessed November 6, 2025, https://huggingface.co/docs/transformers/internal/generation_utils
How to obtain the logits of LLM - Page 2 - General - vLLM Forums, accessed November 6, 2025, https://discuss.vllm.ai/t/how-to-obtain-the-logits-of-llm/847?page=2
How to obtain the logits of LLM - #14 by RunLLM - General - vLLM Forums, accessed November 6, 2025, https://discuss.vllm.ai/t/how-to-obtain-the-logits-of-llm/847/14
Can I directly obtain the logits here? · Issue #185 · vllm-project/vllm - GitHub, accessed November 6, 2025, https://github.com/vllm-project/vllm/issues/185
How can I obtain the logits via model.generate()? - Transformers - Hugging Face Forums, accessed November 6, 2025, https://discuss.huggingface.co/t/how-can-i-obtain-the-logits-via-model-generate/110636
Utilities for Generation - Hugging Face, accessed November 6, 2025, https://huggingface.co/docs/transformers/v4.29.0/internal/generation_utils
Utilities for Generation - Hugging Face, accessed November 6, 2025, https://huggingface.co/docs/transformers/en/internal/generation_utils
Output probabilities of tokens generated by Llama 2 using Transformers - Stack Overflow, accessed November 6, 2025, https://stackoverflow.com/questions/77607529/output-probabilities-of-tokens-generated-by-llama-2-using-transformers
How to stream state updates of your graph, accessed November 6, 2025, https://langchain-ai.github.io/langgraphjs/how-tos/stream-updates/
LangGraph Intro Streaming AI Agent State and API Calls with LangGraph Studio - YouTube, accessed November 6, 2025, https://www.youtube.com/watch?v=hMHyPtwruVs
Streaming - Docs by LangChain, accessed November 6, 2025, https://docs.langchain.com/oss/python/langgraph/streaming
Help: How to access all intermediate yields from tools in LangGraph? : r/LangChain - Reddit, accessed November 6, 2025, https://www.reddit.com/r/LangChain/comments/1mc83k3/help_how_to_access_all_intermediate_yields_from/
openai/gsm8k · Datasets at Hugging Face, accessed November 6, 2025, https://huggingface.co/datasets/openai/gsm8k
openai/gsm8k at 4c5e51a7010b35d464d93f9af614b1da8c56aa06 - Hugging Face, accessed November 6, 2025, https://huggingface.co/datasets/openai/gsm8k/blame/4c5e51a7010b35d464d93f9af614b1da8c56aa06/gsm8k.py
EleutherAI/truthful_qa_mc at 883f491554d90fc1ef003e38c09c643430560af0 - Hugging Face, accessed November 6, 2025, https://huggingface.co/datasets/EleutherAI/truthful_qa_mc/blob/883f491554d90fc1ef003e38c09c643430560af0/truthful_qa_mc.py
Overview - Docs by LangChain, accessed November 6, 2025, https://docs.langchain.com/oss/python/langgraph/overview?__hstc=5909356.8984886db67907baa412a6822f358da9.1757894400286.1757894400287.1757894400288.1&__hssc=5909356.1.1757894400289&__hsfp=2825657416
Streaming - LangChain docs, accessed November 6, 2025, https://python.langchain.com/docs/concepts/streaming/
Estimating LLM Uncertainty with Logits - arXiv, accessed November 6, 2025, https://arxiv.org/html/2502.00290v1
RepairAgent: An Autonomous, LLM-Based Agent for Program Repair - alphaXiv, accessed November 6, 2025, https://www.alphaxiv.org/zh/overview/2403.17134v2
A Survey of LLM-based Automated Program Repair: Taxonomies, Design Paradigms, and Applications - arXiv, accessed November 6, 2025, https://arxiv.org/html/2506.23749v1
RepairAgent: An Autonomous, LLM-Based Agent for Program Repair - arXiv, accessed November 6, 2025, https://arxiv.org/html/2403.17134v1
dvlab-research/MR-GSM8K: Challenge LLMs to Reason About Reasoning: A Benchmark to Unveil Cognitive Depth in LLMs - GitHub, accessed November 6, 2025, https://github.com/dvlab-research/MR-GSM8K
MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation, accessed November 6, 2025, https://openreview.net/forum?id=br4H61LOoI

````markdown
# 4-Day Hyper-Sprint Plan: Uncertainty-Driven Self-Repair Agent

**Project:** Build a Minimum Viable Prototype (MVP) of an agentic workflow capable of token-level, uncertainty-driven self-repair.

**Timeline:** 4 Days

**Team:**
* **P1 (Sai Kavya Marthala):** Uncertainty Engine
* **P2 (Aryan Tapkire):** Agentic Chassis
* **P3 (Deepak Sai Pendyala):** Evaluation & XAI

**Core Strategy (The "API-First" Pivot):** To meet the 4-day deadline, this plan mandates an **API-First** strategy. [cite_start]We will *exclusively* use the OpenAI API, which provides `logprobs` directly[cite: 37, 407, 750]. [cite_start]This is a critical simplification that de-risks the timeline by avoiding the setup and performance bottlenecks of local models [cite: 485-487]. P1's (Sai Kavya's) role will pivot from integrating the `logtoku` repository to parsing these `logprobs` to calculate uncertainty.

---

## 🏛️ Project Architecture: "Generator-Critic" Loop

[cite_start]The architecture will be a **stateful agentic graph** built with LangGraph, implementing a "Generator-Critic" pattern [cite: 353, 367, 379-380].

1.  [cite_start]**UI (Streamlit):** P3 (Deepak) will build a simple UI where the user submits a query (e.g., from GSM8K) [cite: 423, 631-649].
2.  [cite_start]**Agentic Chassis (LangGraph):** P2 (Aryan) will build the core graph that manages the `AgentState` [cite: 367, 387, 561-572].
3.  [cite_start]**Node 1: Generator:** The agent calls the OpenAI API (e.g., `gpt-4o-mini`) with the `logprobs=True` parameter enabled [cite: 407-408, 699].
4.  **Uncertainty Engine:** The `logprobs` object from the API response is passed to P1's (Sai Kavya's) `UncertaintyMonitor` module. [cite_start]This module calculates an uncertainty score (e.g., average token probability or Shannon entropy) [cite: 397-400, 751].
5.  [cite_start]**Conditional Edge (The "Critic"):** The graph's router checks the uncertainty score against a `TRIGGER_THRESHOLD` [cite: 383, 605-612].
    * [cite_start]**If UNCERTAIN (Loop):** The graph routes to the `reflector_node` [cite: 358-360].
        1.  [cite_start]An XAI (Explainable AI) message (e.g., "Uncertainty detected. Refining...") is streamed to the UI [cite: 419-420, 597-598].
        2.  [cite_start]A "critique" prompt is generated (e.g., "The previous reasoning was uncertain. Re-evaluate...") and added to the message history [cite: 501-504].
        3.  [cite_start]The graph loops back to the `generator_node` for another attempt, now with the critique as context[cite: 620].
    * [cite_start]**If CERTAIN (Exit):** The graph routes to the `END` node, and the final, confident answer is streamed to the UI [cite: 611-612, 621].



---

## 🛠️ Resources & Technology Stack

```bash
# Core Orchestration & LLM
pip install langgraph langchain_openai openai

# Uncertainty & Data Handling
pip install numpy scipy pandas datasets

# UI & Interface
pip install streamlit
````

  * [cite\_start]**Models:** `gpt-4o-mini` (for speed and `logprobs` support) [cite: 825]
  * [cite\_start]**Datasets:** `openai/gsm8k` [cite: 428][cite\_start], `EleutherAI/truthful_qa_mc` (multiple choice) [cite: 429]

-----

## 🗓️ The 4-Day End-to-End Plan

| Day | P1: Uncertainty Engineer (Sai Kavya) | P2: Agent Architect (Aryan) | P3: Evaluation & XAI (Deepak) | Integration Milestone |
| :--- | :--- | :--- | :--- | :--- |
| **1** | [cite\_start]**Build `UncertaintyMonitor-v1.py`**.<br>• Write helper to call OpenAI API with `logprobs=True`[cite: 407].<br>• Write parser `calculate_uncertainty(response)` that returns an uncertainty score (e.g., avg. probability). | [cite\_start]**Build `Agent-v0.1.py` (Linear)**.<br>• Define the `AgentState` TypedDict [cite: 387, 561-572][cite\_start].<br>• Build a *non-looping* graph with one `generator_node` that calls the LLM (no repair)[cite: 443]. | [cite\_start]**Build `evaluation_harness.py`**.<br>• Load GSM8K & TruthfulQA [cite: 428-429].<br>• Curate a 10-sample "Demo Set" for rapid testing. | **Proxy Delivery:** P1 delivers the `UncertaintyMonitor` module to P2 for integration. |
| **2** | **Support & Refine**.<br>• Support P2 with the integration of your module.<br>• Begin initial analysis on the demo set to estimate a sensible `TRIGGER_THRESHOLD`. | [cite\_start]**Build `Agent-v1.0.py` (The Loop)**.<br>• **Integrate P1's module** into the `generator_node` to populate `avg_uncertainty` in the state.<br>• Build the `reflector_node` [cite: 358, 594-603][cite\_start].<br>• Build the conditional edge `route_on_uncertainty` [cite: 383, 605-612, 624-629]. | [cite\_start]**Build `xai_ui.py` (Standalone)**.<br>• Build a simple Streamlit UI.<br>• Implement the `astream()` listener skeleton that will eventually connect to the agent [cite: 423, 636-649]. | **Core Loop Functional:** The agent successfully loops and self-repairs on a test prompt when uncertainty is high. |
| **3** | [cite\_start]**Lead "Calibration Session"**.<br>• Work with P2/P3 to run the demo set.<br>• **Finalize the `TRIGGER_THRESHOLD`** value (e.g., `0.5`) to optimize the repair trigger[cite: 457, 459]. | [cite\_start]**Implement XAI Stream**.<br>• Add `get_stream_writer()` calls to your graph nodes [cite: 422, 597][cite\_start].<br>• Stream custom messages like `{"xai_message": "Uncertainty detected..."}`[cite: 422, 598]. | [cite\_start]**Integrate UI & Agent**.<br>• Import P2's compiled `app` into your Streamlit UI.<br>• Connect your UI listener to `app.astream(stream_mode="custom")` [cite: 423, 640-641]. | **Full E2E System Test:** A query entered in the UI successfully triggers the repair loop and displays the XAI message in real-time. |
| **4** | **Documentation**.<br>• Clean up and fully document the `UncertaintyMonitor` library.<br>• Prepare analysis notes for the final demo. | **Code Freeze & Cleanup**.<br>• Clean and fully document the LangGraph `AgentState` and all nodes/edges.<br>• Prepare the live demo script. | [cite\_start]**Generate "Final Report Card"**.<br>• Run the 10-sample Demo Set vs. (1) Baseline (no repair) and (2) RepairAgent.<br>• Create the final `Report Card` table [cite: 435-437]. | **Sprint Complete:** Final demo is prepared and the `Report Card` quantifying the accuracy improvement is complete. |

-----

## 👨‍💻 Detailed Daily Task Breakdown

### Day 1: Core Component Build (Parallel)

  * **Goal:** All team members build their foundational components in isolation.
  * **P1 (Sai Kavya):** Focus on the `UncertaintyMonitor-v1.py` module.
      * Implement a function `call_openai_with_logprobs(prompt: str) -> ChatCompletion`.
      * Implement the parser `calculate_uncertainty(response: ChatCompletion) -> float`. This function will iterate through `response.choices[0].logprobs.content`, extract the log-probability for the chosen token (e.g., `token.logprob`), convert it to a probability (`math.exp(token.logprob)`), and return an aggregate metric (like the average probability).
  * **P2 (Aryan):** Focus on the `Agent-v0.1.py` linear graph.
      * [cite\_start]Define the `AgentState` class using `TypedDict` as specified in the LangGraph documentation [cite: 387, 561-572].
      * Implement a simple `generator_node` that takes the `state`, calls the LLM, and returns the `{ "generation": ... }` update.
      * Compile a basic graph with a `START`, `generator`, and `END` node.
  * **P3 (Deepak):** Focus on the `evaluation_harness.py` and data.
      * [cite\_start]Use the `datasets` library to load `openai/gsm8k` (main split) and `EleutherAI/truthful_qa_mc` [cite: 428-429].
      * Manually select 5 challenging questions from each to create a `demo_set.json` file. This ensures rapid and repeatable testing.
      * [cite\_start]Write a script that can load this JSON, run a (placeholder) agent, and check the answer (e.g., exact match for GSM8K)[cite: 431].

### Day 2: First Integration (The Loop)

  * **Goal:** Assemble the core self-repair loop. This is the most critical technical day.
  * **P2 (Aryan):** This is your focus day.
    1.  **Integrate P1's Module:** Modify your `generator_node` to call P1's `call_openai...` and `calculate_uncertainty...` functions. Store the result in the state (e.g., `return { "generation": ..., "avg_uncertainty": ... }`).
    2.  **Implement `reflector_node`:** Create a new node that takes the `state` and generates a critique prompt. [cite\_start]It should return an update to the `messages` list [cite: 594-603].
    3.  [cite\_start]**Implement `route_on_uncertainty`:** Create the conditional function that takes the `state` and returns a string: `"reflector"` if `state["avg_uncertainty"] < TRIGGER_THRESHOLD` or `"end_with_success"` if not [cite: 605-612].
    4.  [cite\_start]**Wire the Graph:** Use `workflow.add_conditional_edges` to connect the `generator_node` to your `route_on_uncertainty` function, and map the string outputs to the correct nodes (`"reflector"` or `"end_with_success"`) [cite: 383, 624-629].
  * **P1 (Sai Kavya):** Support P2's integration. Debug any issues with the `logprobs` parsing. Run the `demo_set.json` through your standalone module to get an initial feel for a good `TRIGGER_THRESHOLD`.
  * **P3 (Deepak):** Build the `xai_ui.py` in isolation. [cite\_start]Create the Streamlit layout (title, text input, "Run" button, and an `st.empty()` placeholder for XAI messages) [cite: 639, 647-649]. [cite\_start]Write the `async def` function that will (eventually) call the agent's `astream` method [cite: 636-645].

### Day 3: Second Integration (The UI)

  * **Goal:** Connect the agent logic to the user interface for a full E2E demo.
  * **P1 (Sai Kavya):** Lead the **"Calibration Session."**
      * Sit with P2 and P3. Run the 10 demo samples through the (now functional) loop.
      * Observe the `avg_uncertainty` scores.
      * [cite\_start]Propose a final `TRIGGER_THRESHOLD` value (e.g., `0.5`) that correctly triggers repairs on the "tricky" questions but not the "easy" ones[cite: 457, 459].
  * **P2 (Aryan):** Implement the XAI stream.
      * This is a small but crucial change. [cite\_start]In your `reflector_node`, add `writer = get_stream_writer()`[cite: 597].
      * [cite\_start]Call `writer.write({"xai_message": f"Uncertainty ({state['avg_uncertainty']:.2f}) detected. Refining..."})`[cite: 422, 598].
      * [cite\_start]Do the same in your `end_with_success` node (e.g., `{"xai_message": "Confidence: High. Finalizing answer."}`)[cite: 617].
  * **P3 (Deepak):** Connect the frontend to the backend.
      * Import P2's compiled `app` into `xai_ui.py`.
      * [cite\_start]In your `async def` function, call `app.astream(...)` with `stream_mode="custom"` [cite: 423, 640-641].
      * Loop through the `async for chunk` and check `if "xai_message" in chunk:`.
      * [cite\_start]If it is, update your `st.empty()` placeholder with the message `chunk["xai_message"]` [cite: 643-645].
      * **Test:** Run the full E2E system.

### Day 4: Evaluation & Handoff

  * **Goal:** Quantify the prototype's value and prepare for handoff.
  * **P3 (Deepak):** Your focus day.
      * Create a `run_evaluation.py` script.
      * This script should instantiate two versions of the agent:
        1.  **BaselineAgent:** The graph configured to *always* route to `END` (no repairs).
        2.  **RepairAgent:** The full graph with the calibrated threshold.
      * Run your 10-sample `demo_set.json` through both agents.
      * [cite\_start]Generate the **"Final Report Card"** (a simple table in the README) showing the accuracy of each agent on GSM8K and TruthfulQA [cite: 435-437].
  * **P1 & P2:** Code freeze. Clean up, refactor, and add comments/docstrings to your respective modules (`UncertaintyMonitor.py` and `Agent.py`).
  * **All:** Meet to review the "Final Report Card" and rehearse the 5-minute demo, using the Streamlit UI to showcase a clear "win" (a question the Baseline gets wrong but the RepairAgent fixes).

-----

## deliverables

1.  **Code Repository:** A clean, documented codebase with:
      * `UncertaintyMonitor.py` (P1's module)
      * `Agent.py` (P2's LangGraph)
      * `xai_ui.py` (P3's Streamlit App)
      * `evaluation_harness.py` (P3's eval script)
2.  [cite\_start]**Final Report Card:** A section in the `README.md` (e.g., Table 4 from Doc 2) that presents the accuracy (Baseline vs. RepairAgent) on the demo set [cite: 435-437, 460].
3.  **Live Demo:** A functional Streamlit application demonstrating the E2E flow, including the real-time XAI messages.

<!-- end list -->

```
```
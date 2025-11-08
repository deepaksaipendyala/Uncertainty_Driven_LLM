"""LangGraph-based Token-Level Self-Repair Agent.

Implements a Generator-Critic self-repair loop using LangGraph StateGraph
with uncertainty-driven conditional routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from langgraph.graph import StateGraph, END

from ..config import MAX_REPAIR_ATTEMPTS, UNCERTAINTY_THRESHOLD, ProjectConfig
from ..llm.base import LLMClient, TokenLogit
from ..messaging.status import StatusMessenger
from ..repair.constitutional import Reflector
from ..types import AgentState, TokenScore
from ..uncertainty.logtoku import LogTokUEstimator


@dataclass(slots=True)
class TokenSelfRepairGraph:
    """LangGraph-based self-repair agent with uncertainty-driven routing.
    
    This graph implements a Generator-Critic loop:
    1. Generator node produces an answer via LLM
    2. Uncertainty node computes per-token uncertainty (LogTokU)
    3. Conditional routing decides: repair (reflector) or finalize
    4. Reflector node critiques and rewrites the answer
    5. Loop back to generator until confident or max attempts reached
    """

    llm: LLMClient
    estimator: LogTokUEstimator
    reflector: Reflector
    messenger: StatusMessenger
    config: ProjectConfig = field(default_factory=ProjectConfig)
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS

    def __post_init__(self) -> None:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("uncertainty", self._uncertainty_node)
        workflow.add_node("reflector", self._reflector_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("generator")

        # Add edges
        workflow.add_edge("generator", "uncertainty")
        workflow.add_conditional_edges(
            "uncertainty",
            self._route_on_uncertainty,
            {
                "reflector": "reflector",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("reflector", "generator")
        workflow.add_edge("finalize", END)

        # Compile the graph
        self.app = workflow.compile()

    def _generator_node(self, state: AgentState) -> AgentState:
        """Generate an answer using the LLM.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generation, tokens, and logits
        """
        # Determine the prompt to use
        if state["repair_attempts"] > 0:
            # Use the refined prompt from reflector
            prompt = state["messages"][-1] if state["messages"] else state["question"]
        else:
            # First attempt - use original question
            prompt = state["question"]

        # Update status message
        if state["repair_attempts"] > 0:
            xai_msg = f"Refining answer (attempt {state['repair_attempts'] + 1})..."
        else:
            xai_msg = "Generating initial answer..."
        
        state["xai_message"] = xai_msg
        self.messenger.logger.info(xai_msg)

        # Generate tokens with logits
        tokens: List[str] = []
        logits: List[List[float]] = []
        
        for token_logit in self.llm.generate(prompt, max_tokens=256):
            tokens.append(token_logit.token)
            logits.append(list(token_logit.logits))

        # Update state
        state["generation"] = " ".join(tokens)
        state["tokens"] = tokens
        state["logits"] = logits

        return state

    def _uncertainty_node(self, state: AgentState) -> AgentState:
        """Compute per-token uncertainty using LogTokU.
        
        Args:
            state: Current agent state with tokens and logits
            
        Returns:
            Updated state with token_uncertainties and avg_uncertainty
        """
        if not state["tokens"] or not state["logits"]:
            # No tokens generated - set high uncertainty
            state["token_uncertainties"] = [1.0]
            state["avg_uncertainty"] = 1.0
            return state

        # Compute uncertainty scores
        scores: List[TokenScore] = list(
            self.estimator.score(state["tokens"], state["logits"])
        )

        # Extract uncertainties
        uncertainties = [score.total_uncertainty for score in scores]
        avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 1.0

        # Update state
        state["token_uncertainties"] = uncertainties
        state["avg_uncertainty"] = avg_uncertainty
        state["token_scores"] = scores  # Store full scores for adapter

        # Log uncertainty status
        if avg_uncertainty > self.uncertainty_threshold:
            xai_msg = f"Uncertainty detected (avg={avg_uncertainty:.2f}) — refining answer..."
        else:
            xai_msg = f"Confidence high (avg={avg_uncertainty:.2f})."
        
        state["xai_message"] = xai_msg
        self.messenger.logger.info(xai_msg, status="UNCERTAINTY")

        return state

    def _reflector_node(self, state: AgentState) -> AgentState:
        """Critique and refine the answer using constitutional repair.
        
        Args:
            state: Current agent state with generation and uncertainty
            
        Returns:
            Updated state with refined prompt in messages
        """
        # Check if we've exceeded max repair attempts
        if state["repair_attempts"] >= self.max_repair_attempts:
            state["xai_message"] = "Max repair attempts reached. Finalizing..."
            return state

        # Use reflector to critique and generate revised prompt
        # Determine uncertainty level from average uncertainty
        from ..types import UncertaintyLevel
        
        if state["avg_uncertainty"] < 0.2:
            uncertainty_level = UncertaintyLevel.HIGH_CONFIDENCE
        elif state["avg_uncertainty"] < 0.5:
            uncertainty_level = UncertaintyLevel.MODERATE
        else:
            uncertainty_level = UncertaintyLevel.LOW
        
        revised_prompt = self.reflector.critique(
            previous_answer=state["generation"],
            question=state["question"],
            repair_attempt=state["repair_attempts"],
            avg_uncertainty=state["avg_uncertainty"],
            uncertainty_level=uncertainty_level,
        )

        # Update state
        state["repair_attempts"] += 1
        state["messages"].append(revised_prompt)
        state["xai_message"] = f"Uncertainty detected — refining answer (attempt {state['repair_attempts']})..."

        self.messenger.logger.info(
            f"Repair attempt {state['repair_attempts']}: {revised_prompt[:100]}...",
            status="REPAIR",
        )

        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the answer and prepare for output.
        
        Args:
            state: Current agent state
            
        Returns:
            Finalized state with completion message
        """
        state["xai_message"] = (
            f"Response finalized with {state['repair_attempts']} repair attempt(s). "
            f"Final uncertainty: {state['avg_uncertainty']:.2f}"
        )
        
        self.messenger.logger.info(
            f"Finalized: {len(state['tokens'])} tokens, "
            f"avg uncertainty={state['avg_uncertainty']:.2f}",
            status="COMPLETE",
        )

        return state

    def _route_on_uncertainty(self, state: AgentState) -> str:
        """Route based on uncertainty threshold and repair attempts.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name: "reflector" or "finalize"
        """
        # Check if we've exceeded max repair attempts
        if state["repair_attempts"] >= self.max_repair_attempts:
            return "finalize"

        # Check uncertainty threshold
        if state["avg_uncertainty"] > self.uncertainty_threshold:
            return "reflector"
        else:
            return "finalize"

    def run(
        self, question: str, *, stream: bool = False
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Run the self-repair graph on a question.
        
        Args:
            question: The question to answer
            stream: Whether to stream intermediate updates. If True, returns an iterator.
            
        Returns:
            If stream=False: Dictionary with answer, xai_message, and metadata
            If stream=True: Iterator of dictionaries with xai_message updates
        """
        # Initialize state
        initial_state: AgentState = {
            "question": question,
            "generation": "",
            "token_uncertainties": [],
            "avg_uncertainty": 1.0,
            "repair_attempts": 0,
            "xai_message": "Starting generation...",
            "messages": [],
            "logits": [],
            "tokens": [],
            "token_scores": [],
        }

        # Run the graph
        if stream:
            # Stream updates - return iterator
            def _stream() -> Iterator[Dict[str, Any]]:
                final_state = None
                for event in self.app.stream(initial_state):
                    # Process streaming events
                    for node_name, node_state in event.items():
                        if "xai_message" in node_state:
                            # Emit XAI message through stream
                            yield {
                                "xai_message": node_state["xai_message"],
                                "node": node_name,
                                "state": dict(node_state),
                            }
                        final_state = node_state
                
                # Final result
                if final_state:
                    yield {
                        "answer": final_state["generation"],
                        "xai_message": final_state["xai_message"],
                        "meta": {
                            "repair_attempts": final_state["repair_attempts"],
                            "avg_uncertainty": final_state["avg_uncertainty"],
                            "token_uncertainties": final_state["token_uncertainties"],
                            "token_scores": final_state.get("token_scores", []),
                            "num_tokens": len(final_state["tokens"]),
                        },
                    }
            
            return _stream()
        else:
            # Run to completion
            state = self.app.invoke(initial_state)

            # Return formatted result
            return {
                "answer": state["generation"],
                "xai_message": state["xai_message"],
                "meta": {
                    "repair_attempts": state["repair_attempts"],
                    "avg_uncertainty": state["avg_uncertainty"],
                    "token_uncertainties": state["token_uncertainties"],
                    "token_scores": state.get("token_scores", []),
                    "num_tokens": len(state["tokens"]),
                },
            }

    def invoke(self, question: str) -> Dict[str, Any]:
        """Invoke the graph (non-streaming).
        
        Convenience method that calls run(question, stream=False).
        """
        return self.run(question, stream=False)


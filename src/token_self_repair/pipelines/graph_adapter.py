"""Adapter to make TokenSelfRepairGraph compatible with evaluation infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..messaging.status import StatusMessenger
from ..repair.constitutional import Reflector
from ..types import GenerationStep, TokenScore, UncertaintyLevel
from ..uncertainty.logtoku import LogTokUEstimator
from .base import PipelineResult, UncertaintyAwarePipeline
from .token_self_repair_graph import TokenSelfRepairGraph


class GraphPipelineAdapter:
    """Adapter that wraps TokenSelfRepairGraph to match UncertaintyAwarePipeline interface.
    
    This allows the LangGraph-based self-repair agent to be used with
    the existing evaluation infrastructure.
    """

    graph: TokenSelfRepairGraph
    llm: object
    estimator: object
    strategies: list
    messenger: StatusMessenger
    config: object

    def __init__(
        self,
        graph: TokenSelfRepairGraph,
        messenger: StatusMessenger | None = None,
    ) -> None:
        """Initialize adapter.
        
        Args:
            graph: The TokenSelfRepairGraph instance to wrap
            messenger: Optional StatusMessenger (uses graph's messenger if not provided)
        """
        self.graph = graph
        self.llm = graph.llm
        self.estimator = graph.estimator
        self.strategies = []  # Not used in graph-based approach
        self.messenger = messenger or graph.messenger
        self.config = graph.config

    def run(self, prompt: str, *, max_tokens: int = 256) -> PipelineResult:
        """Run the graph and convert result to PipelineResult format.
        
        Args:
            prompt: The question/prompt to process
            max_tokens: Maximum tokens (passed to graph, but graph uses its own limit)
            
        Returns:
            PipelineResult compatible with evaluation infrastructure
        """
        # Run the graph
        result = self.graph.run(prompt, stream=False)

        # Extract data from result
        answer = result["answer"]
        meta = result["meta"]
        
        # Get token_scores from meta if available (preferred)
        token_scores: List[TokenScore] = meta.get("token_scores", [])
        tokens = answer.split() if answer else []

        # If we don't have token_scores, reconstruct from uncertainties
        if not token_scores:
            token_uncertainties = meta.get("token_uncertainties", [])
            for i, (token, uncertainty) in enumerate(zip(tokens, token_uncertainties)):
                # Map uncertainty to level
                if uncertainty < 0.2:
                    level = UncertaintyLevel.HIGH_CONFIDENCE
                elif uncertainty < 0.5:
                    level = UncertaintyLevel.MODERATE
                else:
                    level = UncertaintyLevel.LOW

                token_scores.append(
                    TokenScore(
                        token=token,
                        logit=0.0,  # Not available from graph result
                        probability=1.0 - uncertainty,  # Approximate
                        entropy=uncertainty * 2.0,  # Approximate
                        aleatoric=uncertainty * 0.5,  # Approximate
                        epistemic=uncertainty * 0.5,  # Approximate
                        total_uncertainty=uncertainty,
                        level=level,
                    )
                )

            # If we have fewer uncertainties than tokens, pad with high uncertainty
            while len(token_scores) < len(tokens):
                token_scores.append(
                    TokenScore(
                        token=tokens[len(token_scores)],
                        logit=0.0,
                        probability=0.5,
                        entropy=1.0,
                        aleatoric=0.5,
                        epistemic=0.5,
                        total_uncertainty=0.8,
                        level=UncertaintyLevel.LOW,
                    )
                )

        # Create GenerationStep
        step = GenerationStep(
            prompt=prompt,
            generated_tokens=tokens,
            token_scores=token_scores,
            final=True,
            repair_attempt=meta.get("repair_attempts", 0),
        )

        # Extract messages from messenger history
        messages = [msg.title for msg in self.messenger.history]

        return PipelineResult(step=step, messages=messages)


def create_graph_pipeline(
    llm,
    estimator: LogTokUEstimator | None = None,
    reflector: Reflector | None = None,
    messenger: StatusMessenger | None = None,
    config=None,
) -> GraphPipelineAdapter:
    """Factory function to create a GraphPipelineAdapter.
    
    Args:
        llm: LLMClient instance
        estimator: Optional LogTokUEstimator (creates default if not provided)
        reflector: Optional Reflector (creates default if not provided)
        messenger: Optional StatusMessenger (creates default if not provided)
        config: Optional ProjectConfig (creates default if not provided)
        
    Returns:
        GraphPipelineAdapter instance
    """
    from ..config import ProjectConfig
    from ..messaging.status import StatusMessenger
    from ..repair.constitutional import Reflector
    from ..uncertainty.logtoku import LogTokUEstimator

    if estimator is None:
        estimator = LogTokUEstimator(config or ProjectConfig())
    if reflector is None:
        reflector = Reflector()
    if messenger is None:
        messenger = StatusMessenger()
    if config is None:
        config = ProjectConfig()

    graph = TokenSelfRepairGraph(
        llm=llm,
        estimator=estimator,
        reflector=reflector,
        messenger=messenger,
        config=config,
    )

    return GraphPipelineAdapter(graph=graph, messenger=messenger)


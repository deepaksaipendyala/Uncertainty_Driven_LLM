"""Rule-based repair strategy inspired by constitutional AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import GenerationStep, UncertaintyLevel
from .base import SelfRepairStrategy


@dataclass(slots=True)
class ConstitutionalRepair(SelfRepairStrategy):
    """Applies declarative rules when uncertainty exceeds thresholds."""

    rules: Dict[UncertaintyLevel, List[str]] = field(
        default_factory=lambda: {
            UncertaintyLevel.LOW: [
                "Re-evaluate the last segment for factual accuracy.",
                "Detail the reasoning steps leading to the answer.",
            ],
            UncertaintyLevel.MODERATE: [
                "Clarify assumptions and provide alternative interpretations."
            ],
        }
    )

    def applies(self, step: GenerationStep) -> bool:
        if not step.token_scores:
            return False
        return step.token_scores[-1].level in self.rules

    def repair(self, step: GenerationStep) -> Optional[str]:
        rule_list = self.rules.get(step.token_scores[-1].level)
        if not rule_list:
            return None
        index = step.repair_attempt % len(rule_list)
        instruction = rule_list[index]
        return (
            "A self-repair was triggered:\n"
            f"- Detected uncertainty level: {step.token_scores[-1].level.name}\n"
            f"- Action: {instruction}"
        )


@dataclass(slots=True)
class Reflector:
    """Wrapper around ConstitutionalRepair for LangGraph integration.
    
    Provides a simplified critique() interface that takes a previous answer
    and question to generate a revised prompt.
    """

    repair_strategy: ConstitutionalRepair = field(default_factory=ConstitutionalRepair)

    def critique(
        self,
        previous_answer: str,
        question: str,
        repair_attempt: int = 0,
        avg_uncertainty: Optional[float] = None,
        uncertainty_level: Optional[UncertaintyLevel] = None,
    ) -> str:
        """Critique uncertain outputs and generate a revised prompt.
        
        Args:
            previous_answer: The previously generated answer that needs refinement
            question: The original question being answered
            repair_attempt: Current repair attempt number
            avg_uncertainty: Average uncertainty score (optional)
            uncertainty_level: Uncertainty level enum (optional)
            
        Returns:
            A revised prompt that incorporates the critique
        """
        from ..types import GenerationStep, TokenScore, UncertaintyLevel
        
        # Determine uncertainty level if not provided
        if uncertainty_level is None:
            if avg_uncertainty is not None:
                # Map uncertainty to level based on thresholds
                if avg_uncertainty < 0.2:
                    uncertainty_level = UncertaintyLevel.HIGH_CONFIDENCE
                elif avg_uncertainty < 0.5:
                    uncertainty_level = UncertaintyLevel.MODERATE
                else:
                    uncertainty_level = UncertaintyLevel.LOW
            else:
                uncertainty_level = UncertaintyLevel.MODERATE
        
        # Create a GenerationStep to use with ConstitutionalRepair
        mock_step = GenerationStep(
            prompt=question,
            generated_tokens=previous_answer.split(),
            token_scores=[
                TokenScore(
                    token="",
                    logit=0.0,
                    probability=0.5,
                    entropy=1.0,
                    aleatoric=0.5,
                    epistemic=0.5,
                    total_uncertainty=avg_uncertainty or 0.6,
                    level=uncertainty_level,
                )
            ],
            repair_attempt=repair_attempt,
        )
        
        if not self.repair_strategy.applies(mock_step):
            # Default critique if no specific rule applies
            return (
                f"Question: {question}\n\n"
                f"Previous answer: {previous_answer}\n\n"
                "Please refine this answer by:\n"
                "- Checking factual accuracy\n"
                "- Providing more detail\n"
                "- Clarifying reasoning steps\n\n"
                "Please provide a revised answer:"
            )
        
        repair_instruction = self.repair_strategy.repair(mock_step)
        if not repair_instruction:
            repair_instruction = "Please refine the answer with more detail and accuracy."
        
        return (
            f"Question: {question}\n\n"
            f"Previous answer: {previous_answer}\n\n"
            f"{repair_instruction}\n\n"
            "Please provide a revised answer:"
        )

"""Hierarchical aggregation of token-level uncertainty signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
import numpy as np
import re

from .logtoku import UncertaintyScores


# ---------------------------------------------------------------------------
# Dataclasses for aggregated uncertainty
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LineUncertainty:
    """Aggregated uncertainty metrics for a single line of code/text."""

    line_no: int
    text: str
    token_indices: List[int]
    aleatoric: float
    epistemic: float
    total: float
    entropy: float


@dataclass(slots=True)
class MethodUncertainty:
    """Aggregated uncertainty metrics for a method/function block."""

    name: str
    start_line: int
    end_line: int
    token_indices: List[int]
    aleatoric: float
    epistemic: float
    total: float
    entropy: float


@dataclass(slots=True)
class UncertaintyHotspot:
    """Represents a high-uncertainty region in the source."""

    kind: str  # "line" or "method"
    identifier: str
    score: float
    metadata: Dict[str, str]


@dataclass(slots=True)
class UncertaintyMap:
    """Complete view of uncertainty across tokens, lines, and methods."""

    tokens: List[str]
    scores: UncertaintyScores
    line_scores: Dict[int, LineUncertainty]
    method_scores: Dict[str, MethodUncertainty]
    hotspots: List[UncertaintyHotspot]


# ---------------------------------------------------------------------------
# Uncertainty aggregation logic
# ---------------------------------------------------------------------------


def _aggregate(values: Sequence[float], weights: Optional[Sequence[float]], strategy: str) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)

    if strategy == "max":
        return float(np.max(arr))
    if strategy == "median":
        return float(np.median(arr))
    if strategy == "weighted" and weights:
        w = np.asarray(weights, dtype=np.float32)
        if w.sum() == 0:
            return float(np.mean(arr))
        return float(np.average(arr, weights=w))
    # Default to mean
    return float(np.mean(arr))


class UncertaintyAggregator:
    """
    Aggregates token-level uncertainty into line and method summaries.

    Designed to work with auto-regressive LLM outputs. Line detection is
    resilient to tokenization artefacts (e.g., leading spaces) by using a
    running newline counter. Method detection currently supports Java and
    Python via heuristic parsing, defaulting to the entire document when the
    language is unknown.
    """

    def __init__(
        self,
        *,
        line_strategy: str = "mean",
        method_strategy: str = "mean",
        hotspot_count: int = 10,
    ) -> None:
        self.line_strategy = line_strategy
        self.method_strategy = method_strategy
        self.hotspot_count = hotspot_count

    # Public API ---------------------------------------------------------
    def build_uncertainty_map(
        self,
        scores: UncertaintyScores,
        source_text: str,
        *,
        language: str = "java",
        tokens: Optional[List[str]] = None,
    ) -> UncertaintyMap:
        """
        Build hierarchical uncertainty map for the given source text.

        Args:
            scores: Token-level uncertainty scores from LogTokU
            source_text: Source code or natural language text
            language: Language hint for method detection
            tokens: Optional token list (defaults to scores.token_texts)

        Returns:
            UncertaintyMap
        """

        token_texts = tokens or scores.token_texts
        if token_texts is None:
            token_texts = self._fallback_tokens_from_text(source_text, len(scores.total))

        line_numbers = self._map_tokens_to_lines(token_texts)
        line_scores = self._aggregate_lines(line_numbers, token_texts, scores, source_text)
        method_scores = self._aggregate_methods(line_numbers, scores, source_text, language)
        hotspots = self._compute_hotspots(line_scores, method_scores)

        return UncertaintyMap(
            tokens=list(token_texts),
            scores=scores,
            line_scores=line_scores,
            method_scores=method_scores,
            hotspots=hotspots,
        )

    # ------------------------------------------------------------------
    # Line-level aggregation
    # ------------------------------------------------------------------
    def _map_tokens_to_lines(self, tokens: List[str]) -> List[int]:
        line_no = 1
        mapping: List[int] = []
        for token in tokens:
            mapping.append(line_no)
            newline_count = token.count("\n")
            if newline_count:
                line_no += newline_count
        return mapping

    def _aggregate_lines(
        self,
        line_numbers: List[int],
        tokens: List[str],
        scores: UncertaintyScores,
        source_text: str,
    ) -> Dict[int, LineUncertainty]:
        line_buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, line_no in enumerate(line_numbers):
            line_buckets[line_no].append(idx)

        lines = source_text.splitlines()
        result: Dict[int, LineUncertainty] = {}

        for line_no, indices in line_buckets.items():
            weights = [len(tokens[i]) for i in indices]
            result[line_no] = LineUncertainty(
                line_no=line_no,
                text=lines[line_no - 1] if 0 <= line_no - 1 < len(lines) else "",
                token_indices=indices,
                aleatoric=_aggregate([scores.au[i] for i in indices], weights, self.line_strategy),
                epistemic=_aggregate([scores.eu[i] for i in indices], weights, self.line_strategy),
                total=_aggregate([scores.total[i] for i in indices], weights, self.line_strategy),
                entropy=_aggregate([scores.entropy[i] for i in indices], weights, self.line_strategy),
            )

        return result

    # ------------------------------------------------------------------
    # Method-level aggregation
    # ------------------------------------------------------------------
    def _aggregate_methods(
        self,
        line_numbers: List[int],
        scores: UncertaintyScores,
        source_text: str,
        language: str,
    ) -> Dict[str, MethodUncertainty]:
        methods = self._detect_methods(source_text, language)
        if not methods:
            # Treat entire document as a single method-equivalent region
            total_lines = max(line_numbers) if line_numbers else len(source_text.splitlines())
            methods = [("<document>", 1, total_lines or 1)]

        method_buckets: Dict[str, List[int]] = defaultdict(list)
        for idx, line_no in enumerate(line_numbers):
            for name, start, end in methods:
                if start <= line_no <= end:
                    method_buckets[name].append(idx)
                    break

        result: Dict[str, MethodUncertainty] = {}
        for name, indices in method_buckets.items():
            weights = [1.0] * len(indices) if self.method_strategy == "weighted" else None
            start_line = min(line_numbers[i] for i in indices) if indices else 1
            end_line = max(line_numbers[i] for i in indices) if indices else start_line

            result[name] = MethodUncertainty(
                name=name,
                start_line=start_line,
                end_line=end_line,
                token_indices=indices,
                aleatoric=_aggregate([scores.au[i] for i in indices], weights, self.method_strategy),
                epistemic=_aggregate([scores.eu[i] for i in indices], weights, self.method_strategy),
                total=_aggregate([scores.total[i] for i in indices], weights, self.method_strategy),
                entropy=_aggregate([scores.entropy[i] for i in indices], weights, self.method_strategy),
            )

        return result

    def _detect_methods(self, source_text: str, language: str) -> List[Tuple[str, int, int]]:
        language = language.lower()
        if language == "java":
            return self._detect_java_methods(source_text)
        if language == "python":
            return self._detect_python_functions(source_text)
        return []

    # ------------------------------------------------------------------
    # Hotspot computation
    # ------------------------------------------------------------------
    def _compute_hotspots(
        self,
        line_scores: Dict[int, LineUncertainty],
        method_scores: Dict[str, MethodUncertainty],
    ) -> List[UncertaintyHotspot]:
        hotspots: List[UncertaintyHotspot] = []

        for line_no, line in line_scores.items():
            hotspots.append(
                UncertaintyHotspot(
                    kind="line",
                    identifier=str(line_no),
                    score=line.total,
                    metadata={"text": line.text.strip()},
                )
            )

        for name, method in method_scores.items():
            hotspots.append(
                UncertaintyHotspot(
                    kind="method",
                    identifier=name,
                    score=method.total,
                    metadata={
                        "range": f"{method.start_line}-{method.end_line}",
                    },
                )
            )

        hotspots.sort(key=lambda h: h.score, reverse=True)
        return hotspots[: self.hotspot_count]

    # ------------------------------------------------------------------
    # Method detection helpers
    # ------------------------------------------------------------------
    def _detect_java_methods(self, source_text: str) -> List[Tuple[str, int, int]]:
        pattern = re.compile(
            r"^\s*(public|private|protected)?\s*(static\s+)?[\w<>\[\]]+\s+(?P<name>\w+)\s*\([^)]*\)\s*\{",
            re.MULTILINE,
        )

        methods: List[Tuple[str, int, int]] = []
        for match in pattern.finditer(source_text):
            name = match.group("name")
            start_pos = match.start()
            body_start = source_text.find("{", match.end() - 1)
            if body_start == -1:
                continue
            end_pos = self._find_matching_brace(source_text, body_start)
            if end_pos == -1:
                end_pos = len(source_text) - 1

            start_line = source_text.count("\n", 0, start_pos) + 1
            end_line = source_text.count("\n", 0, end_pos) + 1
            methods.append((name, start_line, end_line))

        return methods

    def _detect_python_functions(self, source_text: str) -> List[Tuple[str, int, int]]:
        pattern = re.compile(r"^\s*def\s+(?P<name>\w+)\s*\([^)]*\):", re.MULTILINE)

        methods: List[Tuple[str, int, int]] = []
        lines = source_text.splitlines()
        for idx, line in enumerate(lines):
            match = pattern.match(line)
            if not match:
                continue

            name = match.group("name")
            start_line = idx + 1
            indent = len(line) - len(line.lstrip())
            end_line = start_line

            for j in range(idx + 1, len(lines)):
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if lines[j].strip() and current_indent <= indent:
                    break
                end_line = j + 1

            methods.append((name, start_line, end_line))

        return methods

    def _find_matching_brace(self, source_text: str, start_pos: int) -> int:
        depth = 0
        for idx in range(start_pos, len(source_text)):
            char = source_text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _fallback_tokens_from_text(self, text: str, count: int) -> List[str]:
        lines = text.split()
        if not lines:
            return [""] * count
        if len(lines) >= count:
            return lines[:count]
        padded = list(lines)
        padded.extend([""] * (count - len(lines)))
        return padded



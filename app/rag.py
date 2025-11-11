"""Retrieval orchestration including local documents and web search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .docstore import MemoryVectorStore


def search_web(query: str, k: int = 3) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return results

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=k):
            results.append(
                {
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", ""),
                }
            )
    return results


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str
    metadata: Dict[str, str]


@dataclass
class RetrievalResult:
    question: str
    chunks: List[RetrievedChunk]
    web_results: List[Dict[str, str]]
    used_websearch: bool


@dataclass
class RAGPipeline:
    vector_store: MemoryVectorStore
    base_top_k: int = 4
    max_context_length: int = 1800

    def retrieve(self, question: str, *, top_k: int | None = None, use_web: bool = False) -> RetrievalResult:
        k = top_k or self.base_top_k
        doc_results = self.vector_store.similarity_search(question, k=k)
        chunks = [
            RetrievedChunk(
                text=chunk.text,
                score=score,
                source=chunk.doc_id,
                metadata={**chunk.metadata, "chunk_id": chunk.chunk_id},
            )
            for chunk, score in doc_results
        ]

        web_results = search_web(question, k=3) if use_web else []
        return RetrievalResult(
            question=question,
            chunks=chunks,
            web_results=web_results,
            used_websearch=use_web,
        )

    def build_prompt(self, question: str, chunks: Sequence[RetrievedChunk], web_results: Sequence[Dict[str, str]]) -> str:
        context_sections: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            context_sections.append(
                f"[Doc {idx}] (score={chunk.score:.2f}, source={chunk.source})\n{chunk.text}\n"
            )

        for idx, web in enumerate(web_results, start=len(context_sections) + 1):
            body = web.get("body") or ""
            context_sections.append(
                f"[Web {idx}] {web.get('title','')}\n{body}\nLink: {web.get('href','')}\n"
            )

        context_blob = "\n".join(context_sections)
        prompt = (
            "You are an uncertainty-aware assistant. Carefully read the provided references and answer the question.\n"
            "Cite sources using [Doc X] or [Web X]. If the answer is uncertain or missing, acknowledge it.\n"
            f"Question: {question}\n\n"
            f"References:\n{context_blob}\n"
            "Answer:"
        )
        return prompt[: self.max_context_length] if len(prompt) > self.max_context_length else prompt



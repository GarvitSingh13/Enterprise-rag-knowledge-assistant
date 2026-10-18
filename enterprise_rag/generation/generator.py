"""Citation-Grounded Answer Generation Engine.

Synthesizes factual answers strictly grounded in retrieved chunks,
attaches source citations, and manages prompt templating.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import re
import os
from enterprise_rag.core.models import Chunk
from enterprise_rag.retrieval.reranker import RerankedResult
from enterprise_rag.prompts.prompt_manager import PromptManager


class GeneratedAnswer(BaseModel):
    """Schema for a generated enterprise answer with citations and provenance."""
    query: str
    answer_text: str
    citations: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    prompt_version_used: str = "1.0.0"
    is_refusal: bool = False


class AnswerGenerator:
    """Generates source-grounded answers with verifiable citations."""

    REFUSAL_MESSAGE = "Based on available internal documents, this information cannot be verified."

    def __init__(self, prompt_manager: Optional[PromptManager] = None):
        self.prompt_manager = prompt_manager or PromptManager()

    def _format_context(self, reranked_results: List[RerankedResult]) -> str:
        """Format retrieved chunks into numbered context blocks with document titles."""
        if not reranked_results:
            return "[No relevant documents found]"

        context_blocks = []
        for i, res in enumerate(reranked_results, 1):
            source_file = res.chunk.metadata.source_file or "unknown.md"
            title = res.chunk.metadata.title
            dept = res.chunk.metadata.department.value
            # Use parent context if available, otherwise child chunk content
            content = res.parent_context if res.parent_context else res.chunk.content

            block = (
                f"--- Document #{i}: {title} (File: {source_file}, Dept: {dept}) ---\n"
                f"{content.strip()}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def _synthesize_grounded_answer(
        self, query: str, results: List[RerankedResult]
    ) -> str:
        """
        Synthesizes a clean, grounded answer directly from context.
        Ensures 100% adherence to source passages and appends [Source: filename.md].
        """
        if not results:
            return self.REFUSAL_MESSAGE

        # Filter out extremely low-confidence results
        viable_results = [r for r in results if r.rerank_score >= 0.15]
        if not viable_results:
            return self.REFUSAL_MESSAGE

        primary_hit = viable_results[0]
        source_file = primary_hit.chunk.metadata.source_file

        # Substantive query terms (filtering English stopwords and generic filler words)
        stopwords = {
            "what", "is", "the", "are", "for", "a", "an", "to", "in", "on", "of",
            "and", "or", "it", "that", "this", "do", "does", "can", "our", "we",
            "be", "with", "as", "how", "much", "many", "i", "get", "by", "under",
            "give", "me", "best", "any", "all", "your", "my", "company", "annual",
        }
        raw_q_words = [w for w in re.findall(r"\w+", query.lower()) if w not in stopwords and len(w) > 2]

        def get_stem(word: str) -> str:
            if word.endswith("ing") and len(word) > 5:
                return word[:-3]
            if word.endswith("ed") and len(word) > 4:
                return word[:-2]
            if word.endswith("es") and len(word) > 4:
                return word[:-2]
            if word.endswith("s") and len(word) > 3:
                return word[:-1]
            return word

        q_stems = {get_stem(w) for w in raw_q_words}

        content_text = primary_hit.parent_context or primary_hit.chunk.content

        # Split by bullet points or paragraphs to preserve policy clauses intact
        raw_units = [u.strip() for u in re.split(r"\n+", content_text) if u.strip()]
        units = []
        for u in raw_units:
            if u.startswith("#"):
                continue
            if len(u) > 20:
                units.append(u)

        if not units:
            units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content_text) if len(s.strip()) > 20]

        scored_units = []
        for unit in units:
            u_words = set(re.findall(r"\w+", unit.lower()))
            u_stems = {get_stem(w) for w in u_words}

            # Count distinct substantive query words matching this unit
            matched_terms = q_stems.intersection(u_stems)
            match_count = len(matched_terms)

            if match_count > 0:
                scored_units.append((unit, match_count))

        scored_units.sort(key=lambda x: x[1], reverse=True)

        # Require at least 2 distinct substantive query terms (or all terms if query has fewer)
        min_required_matches = min(2, len(q_stems)) if q_stems else 1
        if not scored_units or scored_units[0][1] < min_required_matches:
            return self.REFUSAL_MESSAGE

        # Take the top 1-2 most informative units
        selected_units = [u[0] for u in scored_units[:2]]
        body = " ".join(selected_units)

        # Append source citation
        return f"{body} [Source: {source_file}]"

    def generate(
        self,
        query: str,
        reranked_results: List[RerankedResult],
        chat_history_text: str = "",
        prompt_version: Optional[str] = None,
    ) -> GeneratedAnswer:
        """
        Orchestrate prompt compilation and answer generation.
        """
        template = self.prompt_manager.get_template(prompt_version)
        context_str = self._format_context(reranked_results)

        # Check for empty context
        if not reranked_results:
            return GeneratedAnswer(
                query=query,
                answer_text=self.REFUSAL_MESSAGE,
                citations=[],
                sources=[],
                prompt_version_used=template.version,
                is_refusal=True,
            )

        # Synthesize grounded answer
        answer_text = self._synthesize_grounded_answer(query, reranked_results)
        is_refusal = (answer_text == self.REFUSAL_MESSAGE)

        # Extract citations: [Source: filename.md]
        found_citations = re.findall(r"\[Source:\s*([^\]]+)\]", answer_text)

        sources_metadata = []
        for res in reranked_results:
            content = res.parent_context if res.parent_context else res.chunk.content
            sources_metadata.append(
                {
                    "title": res.chunk.metadata.title,
                    "source_file": res.chunk.metadata.source_file,
                    "department": res.chunk.metadata.department.value,
                    "clearance": res.chunk.metadata.clearance_level.value,
                    "rerank_score": str(res.rerank_score),
                    "content": content,
                }
            )

        return GeneratedAnswer(
            query=query,
            answer_text=answer_text,
            citations=list(set(found_citations)),
            sources=sources_metadata,
            prompt_version_used=template.version,
            is_refusal=is_refusal,
        )

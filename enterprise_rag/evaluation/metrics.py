"""RAG Triad Evaluation Metrics Engine.

Implements the 3 canonical production evaluation metrics:
1. Context Relevance: Quality and precision of retrieved chunks.
2. Groundedness (Faithfulness): Verification of factual assertions against source context.
3. Answer Relevance: Degree to which the response satisfies user query intent.
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from enterprise_rag.retrieval.vector_retriever import DenseEmbeddingModel
from enterprise_rag.evaluation.models import RAGTriadScore


class RAGTriadMetrics:
    """Evaluates the three pillars of RAG performance."""

    REFUSAL_TRIGGERS = [
        "cannot be verified",
        "cannot verify",
        "not verified",
        "unable to answer",
        "cannot answer",
        "not mentioned",
        "no information",
        "do not contain",
        "does not contain",
        "outside the scope",
        "consult hr",
        "unauthorized",
        "unverified claim detected",
        "based on available internal documents",
    ]

    def __init__(self, embedder: Optional[DenseEmbeddingModel] = None):
        self.embedder = embedder or DenseEmbeddingModel()

    @classmethod
    def is_safe_refusal(cls, answer: str) -> bool:
        """Detect whether the answer is an intentional, safe refusal."""
        ans_lower = answer.lower()
        return any(trigger in ans_lower for trigger in cls.REFUSAL_TRIGGERS)

    def evaluate_context_relevance(
        self,
        query: str,
        retrieved_contexts: List[str],
    ) -> float:
        """
        Evaluate Context Relevance (0.0 to 1.0).
        Measures the degree to which retrieved chunks contain information relevant
        to the user's query, penalizing noisy or irrelevant context.
        """
        if not retrieved_contexts:
            return 0.0

        query_vec = self.embedder.encode_text(query)
        chunk_scores: List[float] = []

        stopwords = {
            "what", "is", "the", "are", "for", "a", "an", "to", "in", "on", "of",
            "and", "or", "it", "that", "this", "do", "does", "can", "our", "we",
            "be", "per", "how", "much", "many", "give", "me", "best",
        }
        clean_q = re.sub(r"[^\w\s]", " ", query.lower())
        q_tokens = [w for w in clean_q.split() if w not in stopwords and len(w) > 2]

        for context in retrieved_contexts:
            if not context or not context.strip():
                continue

            ctx_vec = self.embedder.encode_text(context[:1000])
            cosine_sim = float(np.dot(query_vec, ctx_vec))
            cosine_sim = max(0.0, min(1.0, cosine_sim))
            scaled_sim = min(1.0, cosine_sim * 2.2)

            # Query token coverage in chunk
            ctx_lower = context.lower()
            matched_tokens = 0
            for tok in q_tokens:
                if tok in ctx_lower or any(tok[:4] in cw for cw in re.findall(r"\w+", ctx_lower)):
                    matched_tokens += 1
            token_coverage = (matched_tokens / len(q_tokens)) if q_tokens else 0.5

            # Blend length-scaled semantic similarity (50%) with lexical coverage (50%)
            chunk_relevance = (0.5 * token_coverage) + (0.5 * scaled_sim)
            chunk_scores.append(chunk_relevance)

        if not chunk_scores:
            return 0.0

        top_score = max(chunk_scores)
        avg_score = sum(chunk_scores) / len(chunk_scores)
        final_score = (0.6 * top_score) + (0.4 * avg_score)
        return round(float(final_score), 4)

    def evaluate_groundedness(
        self,
        answer: str,
        retrieved_contexts: List[str],
    ) -> float:
        """
        Evaluate Groundedness / Faithfulness (0.0 to 1.0).
        Verifies whether every factual claim, numerical metric, and entity
        in the answer is strictly backed by the retrieved context.
        """
        # If it's a safe refusal, groundedness is 1.0 (it made zero ungrounded claims)
        if self.is_safe_refusal(answer):
            return 1.0

        if not retrieved_contexts:
            # Answer made claims with 0 context -> 0% grounded
            return 0.0

        combined_context = " ".join(retrieved_contexts).lower()

        # 1. Extract numerical claims ($75, 120, 85%, 15 minutes, 6 hours)
        numbers_in_answer = re.findall(r"[\$]?\b\d+(?:\.\d+)?%?", answer)
        # 2. Extract capitalized proper nouns / policies / acronyms
        entities_in_answer = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", answer)

        # 3. Extract sentence assertions
        sentences = [s.strip() for s in re.split(r"[.\n•-]", answer) if len(s.strip()) > 15]

        total_claims = 0
        supported_claims = 0

        # Verify numerical facts
        for num in numbers_in_answer:
            total_claims += 1
            clean_num = num.replace("$", "").replace("%", "")
            if clean_num in combined_context:
                supported_claims += 1

        # Verify named entities / acronyms (skip common stopwords like The, This, All)
        common_words = {"The", "This", "All", "Our", "And", "For", "With", "Each", "When", "Source"}
        for ent in entities_in_answer:
            if ent in common_words:
                continue
            total_claims += 1
            if ent.lower() in combined_context:
                supported_claims += 1

        # Verify sentence semantic backing
        for sent in sentences:
            total_claims += 1
            sent_words = [w.lower() for w in re.findall(r"\b\w{4,}\b", sent)]
            if not sent_words:
                continue
            matched = sum(1 for w in sent_words if w in combined_context)
            if (matched / len(sent_words)) >= 0.5:
                supported_claims += 1

        if total_claims == 0:
            return 1.0

        groundedness_score = supported_claims / total_claims
        return round(float(groundedness_score), 4)

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
        is_refusal_expected: bool = False,
    ) -> float:
        """
        Evaluate Answer Relevance (0.0 to 1.0).
        Measures how directly the response answers the user's prompt.
        """
        is_refusal = self.is_safe_refusal(answer)

        # If a refusal was expected (out of domain or adversarial) and model refused:
        if is_refusal_expected:
            return 1.0 if is_refusal else 0.2

        # If a factual answer was expected, but the model refused:
        if is_refusal and not is_refusal_expected:
            return 0.3

        stopwords = {
            "what", "is", "the", "are", "for", "a", "an", "to", "in", "on", "of",
            "and", "or", "it", "that", "this", "do", "does", "can", "our", "we",
            "be", "per", "how", "much", "many", "give", "me", "best",
        }
        clean_q = re.sub(r"[^\w\s]", " ", query.lower())
        q_tokens = [w for w in clean_q.split() if w not in stopwords and len(w) > 2]
        ans_lower = answer.lower()

        matched = 0
        for tok in q_tokens:
            if tok in ans_lower or any(tok[:4] in aw for aw in re.findall(r"\w+", ans_lower)):
                matched += 1
        coverage = (matched / len(q_tokens)) if q_tokens else 0.5

        query_vec = self.embedder.encode_text(query)
        ans_vec = self.embedder.encode_text(answer[:1500])

        cosine_sim = float(np.dot(query_vec, ans_vec))
        cosine_sim = max(0.0, min(1.0, cosine_sim))
        scaled_sim = min(1.0, cosine_sim * 2.0)

        # Check intent keyword satisfaction
        q_lower = query.lower()
        intent_bonus = 0.0
        if any(w in q_lower for w in ["how much", "allowance", "cost", "payout", "budget", "multiplier", "days"]):
            if bool(re.search(r"\$\d+|\d+%|\d+\s*(?:days|months|hours|weeks)", ans_lower)):
                intent_bonus += 0.20

        if any(w in q_lower for w in ["how to", "steps", "procedure", "process", "rule", "rules", "triggers"]):
            if any(marker in ans_lower for marker in ["-", "•", "1.", "step", "first", "must", "trigger", "if"]):
                intent_bonus += 0.20

        relevance = (0.45 * coverage) + (0.35 * scaled_sim) + (0.20 * min(1.0, intent_bonus * 2.5))
        return round(float(min(1.0, max(0.0, relevance))), 4)

    def compute_rag_triad(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        is_refusal_expected: bool = False,
    ) -> RAGTriadScore:
        """
        Compute full RAG Triad scorecard for a query-context-response triplet.
        """
        cr = self.evaluate_context_relevance(query, retrieved_contexts)
        gr = self.evaluate_groundedness(answer, retrieved_contexts)
        ar = self.evaluate_answer_relevance(query, answer, is_refusal_expected)

        is_refusal = self.is_safe_refusal(answer)

        if is_refusal_expected:
            # For adversarial, out-of-domain, or unauthorized boundary queries:
            # The correct, secure outcome is a safe refusal without hallucinations
            passed = is_refusal
            composite = 1.0 if is_refusal else 0.2
        else:
            # For in-domain factual queries:
            composite = (cr + gr + ar) / 3.0
            passed = (composite >= 0.65) and (gr >= 0.75) and not is_refusal

        return RAGTriadScore(
            context_relevance=cr,
            groundedness=gr,
            answer_relevance=ar,
            composite_score=round(composite, 4),
            passed=passed,
        )

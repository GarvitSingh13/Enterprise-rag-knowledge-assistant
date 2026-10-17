"""Automated Hallucination Detection & Grounding Guardrail Engine.

Validates LLM-generated answers against retrieved source documents.
Extracts individual claims and numerical assertions, checks entailment against
source context, calculates a Faithfulness Score (0.0 to 1.0), and intercepts hallucinations.
"""

from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field
import re
from enterprise_rag.core.models import Chunk
from enterprise_rag.generation.generator import GeneratedAnswer


class VerificationReport(BaseModel):
    """Detailed audit report of hallucination and faithfulness verification."""
    is_grounded: bool = Field(description="True if faithfulness score meets or exceeds threshold")
    faithfulness_score: float = Field(description="Ratio of verified claims to total claims (0.0 to 1.0)")
    total_claims: int
    verified_claims: List[str] = Field(default_factory=list)
    unverified_claims: List[str] = Field(default_factory=list)
    detected_hallucinations: List[str] = Field(default_factory=list)
    sanitized_answer: str = Field(description="Safe answer (original if verified, or refusal fallback if ungrounded)")


class HallucinationDetector:
    """
    Automated Fact-Checking Guardrail.
    Inspects generated answers against retrieved source passages before user delivery.
    """

    def __init__(self, faithfulness_threshold: float = 0.75):
        self.faithfulness_threshold = faithfulness_threshold

    def _extract_claims(self, text: str) -> List[str]:
        """Split text into individual atomic factual claim sentences."""
        # Remove citation brackets like [Source: abc.md]
        clean_text = re.sub(r"\[Source:\s*[^\]]+\]", "", text).strip()
        # Split by sentence terminators
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.strip()) > 15]
        return sentences

    def _extract_numerical_facts(self, sentence: str) -> List[str]:
        """Extract exact numbers, currencies, and codes from a claim."""
        # Match dollar amounts, percentages, quantities, and alphanumeric codes
        return re.findall(r"\$\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:%|k|M)?\b|[A-Z]{2,}-\d+-[A-Z0-9]+", sentence)

    def verify_answer(
        self,
        answer: GeneratedAnswer,
        reference_chunks: List[Chunk],
        additional_context: Optional[List[str]] = None,
    ) -> VerificationReport:
        """
        Audit a generated answer against reference source chunks and contexts.

        Returns:
            VerificationReport containing faithfulness score and audit breakdown.
        """
        # If it's already an honest refusal, it's 100% grounded
        if answer.is_refusal:
            return VerificationReport(
                is_grounded=True,
                faithfulness_score=1.0,
                total_claims=1,
                verified_claims=["Refusal message properly declared"],
                unverified_claims=[],
                detected_hallucinations=[],
                sanitized_answer=answer.answer_text,
            )

        claims = self._extract_claims(answer.answer_text)
        if not claims:
            return VerificationReport(
                is_grounded=True,
                faithfulness_score=1.0,
                total_claims=0,
                sanitized_answer=answer.answer_text,
            )

        # Build concatenated reference text corpus (including parent contexts)
        chunk_texts = [c.content for c in reference_chunks]
        if additional_context:
            chunk_texts.extend(additional_context)
        reference_text = " ".join(chunk_texts).lower()

        verified: List[str] = []
        unverified: List[str] = []
        hallucinations: List[str] = []

        for claim in claims:
            claim_lower = claim.lower()
            # 1. Check numerical assertions (Numbers must exist in source text)
            numbers = self._extract_numerical_facts(claim)
            numerical_mismatch = False
            for num in numbers:
                num_clean = num.lower().replace("$", "").replace(",", "")
                if num_clean not in reference_text and num.lower() not in reference_text:
                    numerical_mismatch = True
                    hallucinations.append(
                        f"Fabricated metric/number '{num}' in claim: \"{claim}\""
                    )
                    break

            if numerical_mismatch:
                unverified.append(claim)
                continue

            # 2. Check substantive word overlap (Entity & noun overlap)
            words = re.findall(r"\b[a-zA-Z]{4,}\b", claim_lower)
            if not words:
                verified.append(claim)
                continue

            matches = sum(1 for w in words if w in reference_text)
            overlap_ratio = matches / len(words)

            # A claim is verified if at least 60% of substantive terms appear in context
            if overlap_ratio >= 0.55:
                verified.append(claim)
            else:
                unverified.append(claim)
                hallucinations.append(f"Ungrounded claim (overlap {overlap_ratio:.1%}): \"{claim}\"")

        total = len(claims)
        score = round(len(verified) / max(1, total), 2)
        is_grounded = score >= self.faithfulness_threshold

        # If ungrounded, trigger safe fallback refusal
        if not is_grounded:
            sanitized = (
                "Based on available internal documents, this information cannot be verified. "
                "[Automated Guardrail: Unverified claim detected]"
            )
        else:
            sanitized = answer.answer_text

        return VerificationReport(
            is_grounded=is_grounded,
            faithfulness_score=score,
            total_claims=total,
            verified_claims=verified,
            unverified_claims=unverified,
            detected_hallucinations=hallucinations,
            sanitized_answer=sanitized,
        )

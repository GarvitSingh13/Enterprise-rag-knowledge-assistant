"""Data models for Enterprise RAG Evaluation.

Defines evaluation datasets, RAG Triad scores, and benchmark reports.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enterprise_rag.core.models import Department


class QueryCategory(str, Enum):
    """Classification of benchmark evaluation queries."""
    FACTUAL_IN_DOMAIN = "factual_in_domain"
    ADVERSARIAL_HALLUCINATION = "adversarial_hallucination"
    OUT_OF_DOMAIN = "out_of_domain"
    RBAC_BOUNDARY = "rbac_boundary"


class EvaluationSample(BaseModel):
    """A single curated evaluation test case in the golden dataset."""
    sample_id: str
    query: str
    category: QueryCategory
    expected_citations: List[str] = Field(default_factory=list)
    ground_truth_answer: Optional[str] = None
    is_refusal_expected: bool = False
    user_roles: List[str] = Field(default_factory=lambda: ["employee"])
    user_department: Department = Department.GENERAL
    description: Optional[str] = None


class RAGTriadScore(BaseModel):
    """The 3 core metrics comprising the RAG Triad."""
    context_relevance: float = Field(
        ge=0.0, le=1.0,
        description="Measures if retrieved chunks contain relevant information to answer the query without excessive noise."
    )
    groundedness: float = Field(
        ge=0.0, le=1.0,
        description="Measures if claims in the generated answer are strictly supported by the retrieved context (Faithfulness)."
    )
    answer_relevance: float = Field(
        ge=0.0, le=1.0,
        description="Measures if the generated answer directly addresses the user query without drifting off-topic."
    )
    composite_score: float = Field(
        ge=0.0, le=1.0,
        description="Harmonic/weighted composite score of the RAG Triad."
    )
    passed: bool = Field(
        default=True,
        description="Whether the evaluation criteria were satisfied for this sample."
    )


class EvaluationResult(BaseModel):
    """Detailed evaluation result for a single sample."""
    sample_id: str
    query: str
    category: QueryCategory
    generated_answer: str
    retrieved_citations: List[str] = Field(default_factory=list)
    triad_scores: RAGTriadScore
    is_refusal: bool = False
    expected_refusal: bool = False
    latency_ms: float = 0.0
    passed: bool = True
    diagnostic_notes: List[str] = Field(default_factory=list)


class EvaluationBenchmarkReport(BaseModel):
    """Comprehensive benchmark scorecard across the entire evaluation suite."""
    total_samples: int
    passed_samples: int
    failed_samples: int
    pass_rate_pct: float
    avg_context_relevance: float
    avg_groundedness: float
    avg_answer_relevance: float
    avg_composite_score: float
    category_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    detailed_results: List[EvaluationResult] = Field(default_factory=list)

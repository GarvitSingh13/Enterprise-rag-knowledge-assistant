"""Enterprise RAG Evaluation Suite (The RAG Triad).

Provides automated assessment of Context Relevance, Groundedness (Faithfulness),
and Answer Relevance across golden enterprise evaluation datasets.
"""

from enterprise_rag.evaluation.models import (
    EvaluationSample,
    EvaluationResult,
    RAGTriadScore,
    EvaluationBenchmarkReport,
)
from enterprise_rag.evaluation.metrics import RAGTriadMetrics
from enterprise_rag.evaluation.dataset import GoldenEvaluationDataset
from enterprise_rag.evaluation.evaluator import RAGEvaluator

__all__ = [
    "EvaluationSample",
    "EvaluationResult",
    "RAGTriadScore",
    "EvaluationBenchmarkReport",
    "RAGTriadMetrics",
    "GoldenEvaluationDataset",
    "RAGEvaluator",
]

"""Stage 7 Unit Tests: Automated RAG Evaluation Suite (The RAG Triad).

Validates:
1. Context Relevance metric scoring and noise penalty.
2. Groundedness (Faithfulness) claim verification and hallucination detection.
3. Answer Relevance semantic alignment and refusal correctness.
4. Golden Evaluation Dataset integrity.
5. End-to-End Benchmark Execution on the Secure RAG Gateway.
"""

import pytest
from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.gateway import SecureRAGGateway
from enterprise_rag.evaluation.metrics import RAGTriadMetrics
from enterprise_rag.evaluation.dataset import GoldenEvaluationDataset
from enterprise_rag.evaluation.evaluator import RAGEvaluator
from enterprise_rag.evaluation.models import QueryCategory, EvaluationSample


@pytest.fixture(scope="module")
def prepared_stage7_evaluator():
    """Build and index knowledge base, Secure Gateway, and RAGEvaluator."""
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    auth = AuthService()
    gateway = SecureRAGGateway(hybrid_retriever=hybrid, auth_service=auth)
    evaluator = RAGEvaluator(gateway=gateway, auth_service=auth)

    return {
        "evaluator": evaluator,
        "metrics": RAGTriadMetrics(),
        "gateway": gateway,
        "auth": auth,
    }


def test_context_relevance_metric(prepared_stage7_evaluator):
    """Verify Context Relevance rewards aligned contexts and penalizes noise."""
    metrics: RAGTriadMetrics = prepared_stage7_evaluator["metrics"]
    query = "What is the domestic travel meal allowance per day?"

    relevant_contexts = [
        "Domestic Travel: Up to $75 USD per full travel day ($15 Breakfast, $25 Lunch, $35 Dinner). Alcohol is non-reimbursable.",
    ]
    irrelevant_contexts = [
        "Kubernetes incident response protocol: Run kubectl logs and capture heap dump for memory leaks.",
    ]

    rel_score = metrics.evaluate_context_relevance(query, relevant_contexts)
    irrel_score = metrics.evaluate_context_relevance(query, irrelevant_contexts)

    assert rel_score > 0.60, f"Expected high relevance score, got {rel_score}"
    assert rel_score > irrel_score, "Relevant context must score higher than irrelevant context"


def test_groundedness_faithfulness_metric(prepared_stage7_evaluator):
    """Verify Groundedness detects factual backing vs fabricated claims."""
    metrics: RAGTriadMetrics = prepared_stage7_evaluator["metrics"]
    reference_context = [
        "Domestic Travel: Up to $75 USD per full travel day. Flights under 6 hours must be booked in economy class via Navan.",
    ]

    # Grounded answer: all numbers and terms are backed
    grounded_ans = "The domestic meal allowance is $75 per day, and flights under 6 hours require economy class via Navan."
    score_grounded = metrics.evaluate_groundedness(grounded_ans, reference_context)
    assert score_grounded >= 0.85, f"Expected high groundedness, got {score_grounded}"

    # Fabricated / Hallucinated answer: $500 allowance, first class, Delta
    hallucinated_ans = "The domestic meal allowance is $500 per day with first class flights on Delta Airlines."
    score_hallucinated = metrics.evaluate_groundedness(hallucinated_ans, reference_context)
    assert score_hallucinated < 0.50, f"Expected low groundedness for hallucinated answer, got {score_hallucinated}"

    # Safe refusal should receive 1.0 (faithfully refused rather than hallucinating)
    refusal_ans = "I am unable to answer this question based on the provided enterprise documentation."
    score_refusal = metrics.evaluate_groundedness(refusal_ans, [])
    assert score_refusal == 1.0


def test_answer_relevance_metric(prepared_stage7_evaluator):
    """Verify Answer Relevance rewards direct answers and expected refusals."""
    metrics: RAGTriadMetrics = prepared_stage7_evaluator["metrics"]
    query = "What is the domestic travel meal allowance per day?"

    relevant_ans = "Domestic travel meal allowance is up to $75 USD per day ($15 breakfast, $25 lunch, $35 dinner)."
    irrelevant_ans = "Our company culture values open feedback, continuous learning, and teamwork across offices."

    score_rel = metrics.evaluate_answer_relevance(query, relevant_ans, is_refusal_expected=False)
    score_irrel = metrics.evaluate_answer_relevance(query, irrelevant_ans, is_refusal_expected=False)

    assert score_rel > score_irrel, "Direct answer must score higher in answer relevance than off-topic text"
    assert score_rel >= 0.65

    # Safe refusal for out-of-domain query should score 1.0
    ood_query = "What is the recipe for baking chocolate chip cookies?"
    refusal_ans = "I am unable to answer this question as it is outside the scope of company documentation."
    score_ood = metrics.evaluate_answer_relevance(ood_query, refusal_ans, is_refusal_expected=True)
    assert score_ood == 1.0


def test_golden_dataset_structure():
    """Verify Golden Dataset samples cover all required categories and configurations."""
    samples = GoldenEvaluationDataset.get_benchmark_samples()
    assert len(samples) >= 8

    categories = set(s.category for s in samples)
    assert QueryCategory.FACTUAL_IN_DOMAIN in categories
    assert QueryCategory.ADVERSARIAL_HALLUCINATION in categories
    assert QueryCategory.OUT_OF_DOMAIN in categories
    assert QueryCategory.RBAC_BOUNDARY in categories

    for s in samples:
        assert s.sample_id
        assert s.query
        assert isinstance(s.is_refusal_expected, bool)


def test_end_to_end_evaluation_benchmark(prepared_stage7_evaluator):
    """Run full evaluation suite on Secure RAG Gateway and verify benchmark targets."""
    evaluator: RAGEvaluator = prepared_stage7_evaluator["evaluator"]

    report = evaluator.run_benchmark()

    assert report.total_samples >= 8
    assert report.pass_rate_pct >= 85.0, f"Expected >=85% pass rate, got {report.pass_rate_pct}%"
    assert report.avg_groundedness >= 0.85, f"Expected high average groundedness, got {report.avg_groundedness}"
    assert report.avg_composite_score >= 0.70, f"Expected composite >= 0.70, got {report.avg_composite_score}"
    assert len(report.category_breakdown) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

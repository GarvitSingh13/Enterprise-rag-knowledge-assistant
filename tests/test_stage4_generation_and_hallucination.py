"""Stage 4 Verification Tests: Prompt Versioning, Grounded Generation & Hallucination Guardrails."""

import pytest
from enterprise_rag.core.models import Chunk, DocumentMetadata, Department, ClearanceLevel, ChunkType
from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker, RerankedResult
from enterprise_rag.prompts.prompt_manager import PromptManager
from enterprise_rag.generation.generator import AnswerGenerator, GeneratedAnswer
from enterprise_rag.generation.hallucination_detector import HallucinationDetector


@pytest.fixture(scope="module")
def prepared_environment():
    """Setup pipeline, retrievers, generator, and detector."""
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    reranker = CrossEncoderReranker()
    prompt_mgr = PromptManager()
    generator = AnswerGenerator(prompt_manager=prompt_mgr)
    detector = HallucinationDetector(faithfulness_threshold=0.75)

    return {
        "hybrid": hybrid,
        "reranker": reranker,
        "prompt_mgr": prompt_mgr,
        "generator": generator,
        "detector": detector,
    }


def test_prompt_manager_versioning(prepared_environment):
    """Verify loading, switching, and formatting versioned prompt templates."""
    mgr = prepared_environment["prompt_mgr"]

    # Verify loaded versions
    versions = mgr.list_versions()
    version_keys = [v["version"] for v in versions]
    assert "1.0.0" in version_keys
    assert "1.1.0" in version_keys

    # Verify template formatting
    t1 = mgr.get_template("1.0.0")
    formatted = t1.format(query="How do I submit receipts?", context="Receipts via Concur within 14 days.")
    assert "How do I submit receipts?" in formatted
    assert "Receipts via Concur within 14 days." in formatted


def test_grounded_generation_with_citations(prepared_environment):
    """Verify answer generator produces factual responses with citations."""
    hybrid = prepared_environment["hybrid"]
    reranker = prepared_environment["reranker"]
    generator = prepared_environment["generator"]

    query = "What is the domestic flight travel class rule?"
    candidates = hybrid.search(query, top_k=5)
    reranked = reranker.rerank(query, candidates, top_k=2)

    answer = generator.generate(query, reranked)
    assert not answer.is_refusal
    assert len(answer.citations) > 0
    assert "travel_expense_policy.md" in answer.citations[0]
    assert "economy" in answer.answer_text.lower()


def test_hallucination_detector_approves_factual_answer(prepared_environment):
    """Verify hallucination detector gives high faithfulness score to verified answer."""
    detector = prepared_environment["detector"]

    # Realistic verified answer
    chunk = Chunk(
        content="Full-time employees are eligible for a one-time reimbursement of up to $500 USD for ergonomic equipment.",
        chunk_type=ChunkType.CHILD,
        chunk_index=0,
        metadata=DocumentMetadata(source_file="leave_and_remote_work_policy.md"),
    )
    answer = GeneratedAnswer(
        query="What is the home office allowance?",
        answer_text="Full-time employees receive a one-time reimbursement of up to $500 USD for ergonomic equipment. [Source: leave_and_remote_work_policy.md]",
        citations=["leave_and_remote_work_policy.md"],
    )

    report = detector.verify_answer(answer, [chunk])
    assert report.is_grounded is True
    assert report.faithfulness_score >= 0.80
    assert len(report.detected_hallucinations) == 0


def test_hallucination_detector_catches_fabricated_numbers(prepared_environment):
    """Verify hallucination detector catches and intercepts fabricated numerical claims."""
    detector = prepared_environment["detector"]

    # True context states $500
    chunk = Chunk(
        content="Full-time employees are eligible for a one-time reimbursement of up to $500 USD for ergonomic equipment.",
        chunk_type=ChunkType.CHILD,
        chunk_index=0,
        metadata=DocumentMetadata(source_file="leave_and_remote_work_policy.md"),
    )

    # Hallucinated answer claiming $5,000 and 45 days
    fake_answer = GeneratedAnswer(
        query="What is the home office allowance?",
        answer_text="Employees receive an unlimited $5,000 cash grant within 45 days. [Source: leave_and_remote_work_policy.md]",
        citations=["leave_and_remote_work_policy.md"],
    )

    report = detector.verify_answer(fake_answer, [chunk])
    assert report.is_grounded is False, "Fabricated numbers must fail grounding check"
    assert len(report.detected_hallucinations) > 0
    assert "5000" in report.detected_hallucinations[0] or "$5,000" in report.detected_hallucinations[0]
    # Verify the ungrounded answer was sanitized into a safe refusal
    assert "cannot be verified" in report.sanitized_answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

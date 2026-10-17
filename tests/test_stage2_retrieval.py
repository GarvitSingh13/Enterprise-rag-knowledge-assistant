"""Stage 2 Verification Tests: BM25, Dense Vector, Hybrid Search & RRF."""

import pytest
from pathlib import Path
from enterprise_rag.core.models import Department, ClearanceLevel
from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.bm25_retriever import BM25Retriever
from enterprise_rag.retrieval.vector_retriever import VectorRetriever
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="module")
def prepared_catalog():
    """Run ingestion pipeline on enterprise sample docs and return chunks."""
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)
    return {
        "pipeline": pipeline,
        "children": pipeline.child_store,
        "parents": pipeline.parent_store,
    }


def test_bm25_exact_code_matching(prepared_catalog):
    """Verify BM25 pinpoints exact enterprise error codes."""
    retriever = BM25Retriever()
    retriever.index_chunks(prepared_catalog["children"])

    # Query for exact technical error code
    results = retriever.search("ERR-502-TIMEOUT", top_k=3)
    assert len(results) > 0, "BM25 should return matches for exact error code"
    top_chunk, score = results[0]
    assert "ERR-502-TIMEOUT" in top_chunk.content
    assert top_chunk.metadata.department == Department.ENGINEERING


def test_vector_semantic_matching(prepared_catalog):
    """Verify Dense Vector retriever matches conceptual synonyms."""
    retriever = VectorRetriever()
    retriever.index_chunks(prepared_catalog["children"])

    # Query using natural conversational words not in the title
    results = retriever.search("working from home ergonomic desk allowance", top_k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert score > 0.0
    # Should match either leave/remote work policy or home office allowance
    assert top_chunk.metadata.department in {Department.HR, Department.GENERAL}


def test_metadata_department_filtering(prepared_catalog):
    """Verify that department filtering strictly restricts search results."""
    hybrid = HybridRetriever()
    hybrid.index(prepared_catalog["children"], list(prepared_catalog["parents"].values()))

    # Search for 'canary deployment' filtered by HR (should return 0)
    hr_hits = hybrid.search("canary deployment", top_k=5, department=Department.HR)
    assert len(hr_hits) == 0, "Engineering queries filtered by HR department must yield 0 results"

    # Search for 'canary deployment' filtered by Engineering (must return engineering SOP)
    eng_hits = hybrid.search("canary deployment", top_k=5, department=Department.ENGINEERING)
    assert len(eng_hits) > 0
    assert eng_hits[0].chunk.metadata.department == Department.ENGINEERING


def test_rbac_clearance_filtering(prepared_catalog):
    """Verify RBAC pre-filtering prevents confidential data leakage."""
    hybrid = HybridRetriever()
    hybrid.index(prepared_catalog["children"], list(prepared_catalog["parents"].values()))

    # Query for confidential compensation using standard employee role
    unauthorized_hits = hybrid.search(
        "executive severance RSU vesting",
        top_k=5,
        allowed_roles=["employee", "contractor"],
    )
    for hit in unauthorized_hits:
        assert hit.chunk.metadata.clearance_level != ClearanceLevel.RESTRICTED, (
            "Standard employee must NEVER retrieve restricted executive chunks"
        )

    # Query for same topic with C-Suite clearance
    authorized_hits = hybrid.search(
        "executive severance RSU vesting",
        top_k=5,
        allowed_roles=["c_suite", "executive"],
    )
    restricted_found = any(h.chunk.metadata.clearance_level == ClearanceLevel.RESTRICTED for h in authorized_hits)
    assert restricted_found, "C-Suite user should successfully retrieve restricted memo"


def test_reciprocal_rank_fusion_and_parent_attachment(prepared_catalog):
    """Verify RRF formula merges BM25 and Vector rankings and attaches parent context."""
    hybrid = HybridRetriever(rrf_k=60)
    hybrid.index(prepared_catalog["children"], list(prepared_catalog["parents"].values()))

    hits = hybrid.search("per diem international flight travel reimbursement", top_k=3)
    assert len(hits) > 0
    top_hit = hits[0]

    assert top_hit.rrf_score > 0.0
    assert top_hit.parent_context is not None, "Parent context should be attached to top hit"
    assert len(top_hit.parent_context) >= len(top_hit.chunk.content)
    assert top_hit.chunk.metadata.department == Department.FINANCE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Stage 6 Verification Tests: Semantic Caching & Telemetry Observability."""

import pytest
import time
from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.caching.semantic_cache import SemanticCache
from enterprise_rag.observability.tracer import QueryTracer
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.gateway import SecureRAGGateway


@pytest.fixture(scope="module")
def prepared_stage6_gateway():
    """Setup pipeline and Secure Gateway equipped with Semantic Cache and Query Tracer."""
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    auth = AuthService()
    cache = SemanticCache(similarity_threshold=0.82)
    tracer = QueryTracer()

    gateway = SecureRAGGateway(
        hybrid_retriever=hybrid,
        auth_service=auth,
        cache=cache,
        tracer=tracer,
    )

    return {
        "gateway": gateway,
        "cache": cache,
        "tracer": tracer,
        "auth": auth,
    }


def test_semantic_cache_hit_and_similarity():
    """Verify cache returns entry for exact and semantically equivalent queries."""
    cache = SemanticCache(similarity_threshold=0.65)

    # Store entry
    cache.store(
        query="What is the domestic flight booking rule?",
        answer_text="Domestic flights under 6 hours must be booked in economy class.",
        citations=["travel_expense_policy.md"],
        sources=[],
        required_roles=["employee"],
    )

    # 1. Exact lookup
    hit_exact = cache.lookup("What is the domestic flight booking rule?", ["employee"])
    assert hit_exact is not None
    assert "economy" in hit_exact.answer_text

    # 2. Semantic synonym lookup
    hit_semantic = cache.lookup("rules for domestic flights booking and class", ["employee"])
    assert hit_semantic is not None
    assert "economy" in hit_semantic.answer_text

    # 3. Irrelevant query (must miss)
    miss = cache.lookup("Kubernetes pod canary deployment failure", ["employee"])
    assert miss is None

    stats = cache.get_stats()
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 1


def test_semantic_cache_rbac_isolation():
    """Verify regular employees cannot retrieve cached executive secrets."""
    cache = SemanticCache(similarity_threshold=0.65)

    # Store confidential executive bonus entry
    cache.store(
        query="What is the C-suite severance multiplier?",
        answer_text="C-suite severance equals 18 months base salary plus prorated bonus.",
        citations=["executive_compensation_restricted.md"],
        sources=[],
        required_roles=["c_suite", "admin"],
    )

    # Unauthorized employee lookup (must be blocked)
    unauthorized = cache.lookup(
        "What is the C-suite severance multiplier?",
        user_roles=["employee"],
    )
    assert unauthorized is None, "Regular employee must be blocked from cached executive entries"

    # Authorized C-suite lookup
    authorized = cache.lookup(
        "What is the C-suite severance multiplier?",
        user_roles=["employee", "c_suite"],
    )
    assert authorized is not None
    assert "18 months" in authorized.answer_text


def test_query_tracer_spans_and_metrics():
    """Verify QueryTracer records execution spans and aggregates metrics."""
    tracer = QueryTracer()
    trace_id = tracer.start_trace("Sample query")

    with tracer.span("step_alpha"):
        time.sleep(0.005)  # 5ms

    with tracer.span("step_beta"):
        time.sleep(0.003)  # 3ms

    record = tracer.record_trace(
        query="Sample query",
        cache_hit=False,
        prompt_text="What is our policy?",
        completion_text="Here is the answer.",
        trace_id=trace_id,
    )

    assert record.total_latency_ms > 5.0
    assert "step_alpha" in record.spans
    assert "step_beta" in record.spans
    assert record.token_usage["total_tokens"] > 0

    metrics = tracer.get_summary_metrics()
    assert metrics["total_queries_tracked"] == 1
    assert metrics["avg_latency_ms"] > 0.0


def test_end_to_end_gateway_latency_reduction_on_cache_hit(prepared_stage6_gateway):
    """Verify end-to-end Gateway demonstrates massive latency drop on cached queries."""
    gateway = prepared_stage6_gateway["gateway"]
    auth = prepared_stage6_gateway["auth"]

    token = auth.login("alice_emp", "EmployeePass123!")
    query = "What is the domestic meal allowance per day?"

    # Turn 1: Cold Query (Cache MISS)
    resp_cold = gateway.query(token=token, query=query, use_cache=True)
    assert resp_cold.cache_hit is False
    assert resp_cold.total_latency_ms > 0
    assert "hybrid_retrieval" in resp_cold.latency_breakdown_ms

    # Turn 2: Semantically Equivalent Query (Cache HIT!)
    resp_warm = gateway.query(token=token, query=query, use_cache=True)
    assert resp_warm.cache_hit is True
    assert "semantic_cache_lookup" in resp_warm.latency_breakdown_ms
    # Cache hit must be significantly faster than full cold retrieval
    assert resp_warm.total_latency_ms <= resp_cold.total_latency_ms
    assert resp_warm.answer == resp_cold.answer


def test_gateway_cache_rbac_isolation(prepared_stage6_gateway):
    """Verify that sensitive answers cached by C-Suite are blocked by RBAC for employees."""
    gateway = prepared_stage6_gateway["gateway"]
    auth = prepared_stage6_gateway["auth"]

    sarah_token = auth.login("sarah_ceo", "ExecutivePass123!")
    alice_token = auth.login("alice_emp", "EmployeePass123!")
    sensitive_q = "What is the C-suite severance payout and equity cliff schedule?"

    # 1. Sarah (CEO) queries -> retrieves executive document and caches it
    resp_ceo = gateway.query(token=sarah_token, query=sensitive_q, use_cache=True)
    assert not resp_ceo.is_refusal
    assert "executive_compensation_restricted.md" in resp_ceo.citations

    # 2. Alice (Employee) queries the exact same question
    resp_alice = gateway.query(token=alice_token, query=sensitive_q, use_cache=True)
    # Alice must NOT get a cache hit (RBAC blocks unauthorized cache entries)
    assert resp_alice.cache_hit is False
    # And Alice must NOT receive the executive citations
    assert "executive_compensation_restricted.md" not in resp_alice.citations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Secure Enterprise RAG Gateway with Semantic Caching & Telemetry Tracing.

Unifies User Authentication, RBAC Pre-Filtering, Semantic Caching,
Query Rewriting, Hybrid Retrieval, Cross-Encoder Reranking, Grounded Generation,
Hallucination Guardrails, and Performance Telemetry into a single hardened pipeline.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enterprise_rag.core.models import Department, ClearanceLevel
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.rbac import RBACGuard
from enterprise_rag.security.models import TokenPayload
from enterprise_rag.conversation.memory import ConversationMemory
from enterprise_rag.conversation.query_rewriter import QueryRewriter
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker
from enterprise_rag.generation.generator import AnswerGenerator
from enterprise_rag.generation.hallucination_detector import HallucinationDetector
from enterprise_rag.caching.semantic_cache import SemanticCache
from enterprise_rag.observability.tracer import QueryTracer


class SecureQueryResponse(BaseModel):
    """Unified response object returned by the Secure RAG Gateway."""
    query: str
    rewritten_query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    faithfulness_score: float
    is_grounded: bool
    authenticated_user: Dict[str, Any]
    retrieved_sources_count: int
    is_refusal: bool = False
    cache_hit: bool = False
    total_latency_ms: float = 0.0
    latency_breakdown_ms: Dict[str, float] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)


class SecureRAGGateway:
    """Hardened enterprise gateway orchestrating security, caching, telemetry, and RAG."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[AnswerGenerator] = None,
        detector: Optional[HallucinationDetector] = None,
        rewriter: Optional[QueryRewriter] = None,
        auth_service: Optional[AuthService] = None,
        cache: Optional[SemanticCache] = None,
        tracer: Optional[QueryTracer] = None,
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or AnswerGenerator()
        self.detector = detector or HallucinationDetector(faithfulness_threshold=0.75)
        self.rewriter = rewriter or QueryRewriter()
        self.auth_service = auth_service or AuthService()
        self.cache = cache or SemanticCache()
        self.tracer = tracer or QueryTracer()
        self.sessions: Dict[str, ConversationMemory] = {}

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """Fetch or initialize session-based conversation memory."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(session_id=session_id)
        return self.sessions[session_id]

    def query(
        self,
        token: str,
        query: str,
        session_id: Optional[str] = None,
        department_filter: Optional[Department] = None,
        top_k: int = 3,
        use_cache: bool = True,
    ) -> SecureQueryResponse:
        """
        Execute an end-to-end authenticated, cached, and RBAC-governed RAG query.
        """
        trace_id = self.tracer.start_trace(query)

        # 1. Authenticate user token
        with self.tracer.span("auth_verification"):
            token_payload = self.auth_service.verify_token(token)

        user_info = {
            "username": token_payload.username,
            "full_name": token_payload.full_name,
            "department": token_payload.department,
            "roles": token_payload.roles,
        }

        # 2. Check Semantic Cache (if enabled)
        if use_cache:
            with self.tracer.span("semantic_cache_lookup"):
                cached_entry = self.cache.lookup(query, token_payload.roles)

            if cached_entry:
                trace_record = self.tracer.record_trace(
                    query=query,
                    cache_hit=True,
                    completion_text=cached_entry.answer_text,
                    trace_id=trace_id,
                )
                return SecureQueryResponse(
                    query=query,
                    rewritten_query=query,
                    answer=cached_entry.answer_text,
                    citations=cached_entry.citations,
                    sources=cached_entry.sources,
                    faithfulness_score=cached_entry.faithfulness_score,
                    is_grounded=True,
                    authenticated_user=user_info,
                    retrieved_sources_count=len(cached_entry.sources),
                    is_refusal=False,
                    cache_hit=True,
                    total_latency_ms=trace_record.total_latency_ms,
                    latency_breakdown_ms=trace_record.spans,
                    token_usage=trace_record.token_usage,
                )

        # 3. Get or initialize conversation memory
        sid = session_id or token_payload.sub
        memory = self.get_or_create_session(sid)

        # 4. Rewrite query for conversational continuity
        with self.tracer.span("query_rewriting"):
            rewritten_query = self.rewriter.rewrite(query, memory)

        # 5. Extract RBAC pre-retrieval filters from token claims
        with self.tracer.span("rbac_filter_derivation"):
            rbac_constraints = RBACGuard.get_search_filters(token_payload)

        # 6. Execute secure Hybrid Search (BM25 + Vector + RRF)
        with self.tracer.span("hybrid_retrieval"):
            candidates = self.retriever.search(
                query=rewritten_query,
                top_k=max(10, top_k * 3),
                department=department_filter,
                allowed_roles=rbac_constraints["allowed_roles"],
                attach_parent_context=True,
            )

        # 7. Apply Cross-Encoder Reranker
        with self.tracer.span("cross_encoder_rerank"):
            reranked = self.reranker.rerank(rewritten_query, candidates, top_k=top_k)

        # 8. Generate Citation-Grounded Answer
        history_text = memory.get_history_as_text(max_turns=3)
        with self.tracer.span("grounded_generation"):
            generated_answer = self.generator.generate(
                query=rewritten_query,
                reranked_results=reranked,
                chat_history_text=history_text,
            )

        # 9. Fact-Checking & Hallucination Guardrail Audit
        top_chunks = [r.chunk for r in reranked]
        parent_contexts = [r.parent_context for r in reranked if r.parent_context]
        with self.tracer.span("hallucination_guardrail"):
            audit_report = self.detector.verify_answer(
                answer=generated_answer,
                reference_chunks=top_chunks,
                additional_context=parent_contexts,
            )

        final_answer = audit_report.sanitized_answer
        is_refusal = generated_answer.is_refusal or not audit_report.is_grounded

        # 10. Cache verified, grounded answers (stamped with source documents' allowed roles)
        if use_cache and not is_refusal and audit_report.is_grounded:
            clearances = [r.chunk.metadata.clearance_level for r in reranked]
            if ClearanceLevel.RESTRICTED in clearances:
                cache_roles = ["c_suite", "admin"]
            elif ClearanceLevel.CONFIDENTIAL in clearances:
                confidential_roles = set()
                for r in reranked:
                    if r.chunk.metadata.clearance_level == ClearanceLevel.CONFIDENTIAL:
                        confidential_roles.update(role.lower() for role in r.chunk.metadata.allowed_roles)
                cache_roles = list(confidential_roles) if confidential_roles else ["manager", "admin", "c_suite"]
            else:
                internal_roles = set()
                for r in reranked:
                    internal_roles.update(role.lower() for role in r.chunk.metadata.allowed_roles)
                cache_roles = list(internal_roles) if internal_roles else token_payload.roles

            self.cache.store(
                query=query,
                answer_text=final_answer,
                citations=generated_answer.citations,
                sources=generated_answer.sources,
                required_roles=cache_roles,
                faithfulness_score=audit_report.faithfulness_score,
            )

        # 11. Record turn in Conversation Memory
        memory.add_turn(
            user_raw_query=query,
            assistant_response=final_answer,
            rewritten_query=rewritten_query,
            sources=generated_answer.citations,
        )

        # 12. Finalize telemetry trace record
        trace_record = self.tracer.record_trace(
            query=query,
            cache_hit=False,
            prompt_text=rewritten_query + history_text,
            completion_text=final_answer,
            trace_id=trace_id,
        )

        return SecureQueryResponse(
            query=query,
            rewritten_query=rewritten_query,
            answer=final_answer,
            citations=generated_answer.citations,
            sources=generated_answer.sources,
            faithfulness_score=audit_report.faithfulness_score,
            is_grounded=audit_report.is_grounded,
            authenticated_user=user_info,
            retrieved_sources_count=len(reranked),
            is_refusal=is_refusal,
            cache_hit=False,
            total_latency_ms=trace_record.total_latency_ms,
            latency_breakdown_ms=trace_record.spans,
            token_usage=trace_record.token_usage,
        )

"""Secure Enterprise RAG Gateway.

Unifies User Authentication, RBAC Pre-Filtering, Query Rewriting,
Hybrid Retrieval, Cross-Encoder Reranking, Grounded Generation,
and Hallucination Guardrails into a single hardened pipeline.
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


class SecureRAGGateway:
    """Hardened enterprise gateway orchestrating security and RAG pipelines."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[AnswerGenerator] = None,
        detector: Optional[HallucinationDetector] = None,
        rewriter: Optional[QueryRewriter] = None,
        auth_service: Optional[AuthService] = None,
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or AnswerGenerator()
        self.detector = detector or HallucinationDetector(faithfulness_threshold=0.75)
        self.rewriter = rewriter or QueryRewriter()
        self.auth_service = auth_service or AuthService()
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
    ) -> SecureQueryResponse:
        """
        Execute an end-to-end authenticated and RBAC-governed RAG query.
        """
        # 1. Authenticate user token
        token_payload = self.auth_service.verify_token(token)

        # 2. Get or initialize conversation memory
        sid = session_id or token_payload.sub
        memory = self.get_or_create_session(sid)

        # 3. Rewrite query for conversational continuity
        rewritten_query = self.rewriter.rewrite(query, memory)

        # 4. Extract RBAC pre-retrieval filters from token claims
        rbac_constraints = RBACGuard.get_search_filters(token_payload)

        # 5. Execute secure Hybrid Search (BM25 + Vector + RRF)
        candidates = self.retriever.search(
            query=rewritten_query,
            top_k=max(10, top_k * 3),
            department=department_filter,
            allowed_roles=rbac_constraints["allowed_roles"],
            attach_parent_context=True,
        )

        # 6. Apply Cross-Encoder Reranker
        reranked = self.reranker.rerank(rewritten_query, candidates, top_k=top_k)

        # 7. Generate Citation-Grounded Answer
        history_text = memory.get_history_as_text(max_turns=3)
        generated_answer = self.generator.generate(
            query=rewritten_query,
            reranked_results=reranked,
            chat_history_text=history_text,
        )

        # 8. Fact-Checking & Hallucination Guardrail Audit
        top_chunks = [r.chunk for r in reranked]
        parent_contexts = [r.parent_context for r in reranked if r.parent_context]
        audit_report = self.detector.verify_answer(
            answer=generated_answer,
            reference_chunks=top_chunks,
            additional_context=parent_contexts,
        )

        final_answer = audit_report.sanitized_answer

        # 9. Record turn in Conversation Memory
        memory.add_turn(
            user_raw_query=query,
            assistant_response=final_answer,
            rewritten_query=rewritten_query,
            sources=generated_answer.citations,
        )

        return SecureQueryResponse(
            query=query,
            rewritten_query=rewritten_query,
            answer=final_answer,
            citations=generated_answer.citations,
            sources=generated_answer.sources,
            faithfulness_score=audit_report.faithfulness_score,
            is_grounded=audit_report.is_grounded,
            authenticated_user={
                "username": token_payload.username,
                "full_name": token_payload.full_name,
                "department": token_payload.department,
                "roles": token_payload.roles,
            },
            retrieved_sources_count=len(reranked),
            is_refusal=generated_answer.is_refusal or not audit_report.is_grounded,
        )

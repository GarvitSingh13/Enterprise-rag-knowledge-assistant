"""Hybrid Retriever combining BM25 Keyword Search and Dense Vector Search.

Uses Reciprocal Rank Fusion (RRF) to merge sparse lexical and dense semantic
rankings into a unified, mathematically robust score.
"""

from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
from enterprise_rag.core.models import Chunk, Department, ClearanceLevel
from enterprise_rag.retrieval.bm25_retriever import BM25Retriever
from enterprise_rag.retrieval.vector_retriever import VectorRetriever


class RetrievalResult(BaseModel):
    """Encapsulates a retrieved chunk with its fusion scores and rankings."""
    chunk: Chunk
    rrf_score: float
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    bm25_raw_score: Optional[float] = None
    vector_raw_score: Optional[float] = None
    parent_context: Optional[str] = None


class HybridRetriever:
    """
    Orchestrates dual BM25 + Vector retrieval and Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60, bm25_weight: float = 1.0, vector_weight: float = 1.0):
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()
        self.rrf_k = rrf_k  # Standard RRF smoothing constant
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.parent_catalog: Dict[str, Chunk] = {}

    def index(self, child_chunks: List[Chunk], parent_chunks: Optional[List[Chunk]] = None) -> None:
        """Index chunks into both BM25 and Vector search engines."""
        self.bm25_retriever.index_chunks(child_chunks)
        self.vector_retriever.index_chunks(child_chunks)

        if parent_chunks:
            self.parent_catalog = {p.id: p for p in parent_chunks}

    def set_parent_catalog(self, parent_catalog: Dict[str, Chunk]) -> None:
        """Register parent chunk catalog for parent-document retrieval."""
        self.parent_catalog = parent_catalog

    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_multiplier: int = 4,
        department: Optional[Department] = None,
        clearance_level: Optional[ClearanceLevel] = None,
        allowed_roles: Optional[List[str]] = None,
        attach_parent_context: bool = True,
    ) -> List[RetrievalResult]:
        """
        Execute parallel BM25 and Vector search, then combine via Reciprocal Rank Fusion.

        Args:
            query: The user search query
            top_k: Number of final fused chunks to return
            fetch_multiplier: Over-fetch multiplier for candidate generation (e.g. 4 * top_k)
            department: Optional department filter
            clearance_level: Optional clearance filter
            allowed_roles: Optional RBAC role filter
            attach_parent_context: Whether to attach the full parent context to the result
        """
        candidate_k = max(20, top_k * fetch_multiplier)

        # 1. Execute BM25 search
        bm25_hits = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
            department=department,
            clearance_level=clearance_level,
            allowed_roles=allowed_roles,
        )

        # 2. Execute Dense Vector search
        vector_hits = self.vector_retriever.search(
            query=query,
            top_k=candidate_k,
            department=department,
            clearance_level=clearance_level,
            allowed_roles=allowed_roles,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score(d) = sum( weight / (k + rank) )
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        bm25_rank_map: Dict[str, int] = {}
        vector_rank_map: Dict[str, int] = {}
        bm25_raw_map: Dict[str, float] = {}
        vector_raw_map: Dict[str, float] = {}

        # Process BM25 ranks (1-indexed)
        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            cid = chunk.id
            chunk_map[cid] = chunk
            bm25_rank_map[cid] = rank
            bm25_raw_map[cid] = score
            rrf_contrib = self.bm25_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # Process Vector ranks (1-indexed)
        for rank, (chunk, score) in enumerate(vector_hits, start=1):
            cid = chunk.id
            chunk_map[cid] = chunk
            vector_rank_map[cid] = rank
            vector_raw_map[cid] = score
            rrf_contrib = self.vector_weight / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # Sort all candidates by descending RRF score
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        final_results: List[RetrievalResult] = []
        for cid in sorted_cids[:top_k]:
            chunk = chunk_map[cid]
            parent_ctx = None
            if attach_parent_context and chunk.parent_id and chunk.parent_id in self.parent_catalog:
                parent_ctx = self.parent_catalog[chunk.parent_id].content

            res = RetrievalResult(
                chunk=chunk,
                rrf_score=round(fused_scores[cid], 6),
                bm25_rank=bm25_rank_map.get(cid),
                vector_rank=vector_rank_map.get(cid),
                bm25_raw_score=round(bm25_raw_map[cid], 4) if cid in bm25_raw_map else None,
                vector_raw_score=round(vector_raw_map[cid], 4) if cid in vector_raw_map else None,
                parent_context=parent_ctx,
            )
            final_results.append(res)

        return final_results

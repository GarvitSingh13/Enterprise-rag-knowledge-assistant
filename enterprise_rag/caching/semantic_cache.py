"""Semantic Caching Engine for Enterprise RAG.

Caches verified Q&A pairs using dense semantic vector similarity.
Returns cached answers in <20ms for semantically equivalent queries,
enforces Role-Based Access Control (RBAC) on cache entries, and tracks hit metrics.
"""

from typing import List, Dict, Optional, Any
import time
import numpy as np
from pydantic import BaseModel, Field
from enterprise_rag.retrieval.vector_retriever import DenseEmbeddingModel


class CacheEntry(BaseModel):
    """Encapsulates a cached verified query-answer pair with security metadata."""
    query_text: str
    embedding: List[float] = Field(description="Normalized dense vector representation of query")
    answer_text: str
    citations: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    required_roles: List[str] = Field(default_factory=list)
    faithfulness_score: float = 1.0
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: int = 3600
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if cache entry TTL has elapsed."""
        return (time.time() - self.created_at) > self.ttl_seconds


class SemanticCache:
    """
    Vector similarity cache with role-based clearance verification.
    """

    def __init__(self, similarity_threshold: float = 0.65, default_ttl: int = 3600):
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.embedder = DenseEmbeddingModel(dimension=1024)
        self.entries: List[CacheEntry] = []
        self.hit_count: int = 0
        self.miss_count: int = 0

    def lookup(
        self,
        query: str,
        user_roles: List[str],
        min_threshold: Optional[float] = None,
    ) -> Optional[CacheEntry]:
        """
        Search cache for a semantically equivalent query.

        Args:
            query: The incoming user query
            user_roles: The authenticated user's assigned roles
            min_threshold: Optional threshold override

        Returns:
            Matching CacheEntry if similarity >= threshold and user is authorized, else None.
        """
        if not self.entries:
            self.miss_count += 1
            return None

        threshold = min_threshold or self.similarity_threshold
        query_vec = self.embedder.encode_text(query)
        user_roles_lower = set(r.lower() for r in user_roles)

        best_match: Optional[CacheEntry] = None
        highest_similarity = -1.0

        for entry in self.entries:
            # Check TTL
            if entry.is_expired:
                continue

            # RBAC Check: User must hold at least one required role for this cached document
            if entry.required_roles:
                entry_roles_lower = set(r.lower() for r in entry.required_roles)
                if not user_roles_lower.intersection(entry_roles_lower):
                    continue  # Skip unauthorized cache entry

            # Cosine similarity between normalized unit vectors
            entry_vec = np.array(entry.embedding, dtype=np.float32)
            similarity = float(np.dot(query_vec, entry_vec))

            if similarity >= threshold and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = entry

        if best_match:
            best_match.hit_count += 1
            self.hit_count += 1
            return best_match

        self.miss_count += 1
        return None

    def store(
        self,
        query: str,
        answer_text: str,
        citations: List[str],
        sources: List[Dict[str, str]],
        required_roles: List[str],
        faithfulness_score: float = 1.0,
        ttl_seconds: Optional[int] = None,
    ) -> CacheEntry:
        """
        Store a verified query-response pair in the semantic cache.
        """
        query_vec = self.embedder.encode_text(query).tolist()
        entry = CacheEntry(
            query_text=query,
            embedding=query_vec,
            answer_text=answer_text,
            citations=citations,
            sources=sources,
            required_roles=[r.lower() for r in required_roles],
            faithfulness_score=faithfulness_score,
            created_at=time.time(),
            ttl_seconds=ttl_seconds or self.default_ttl,
            hit_count=0,
        )
        self.entries.append(entry)
        return entry

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0
        return {
            "total_cached_queries": len(self.entries),
            "cache_hits": self.hit_count,
            "cache_misses": self.miss_count,
            "hit_rate_pct": round(hit_rate, 2),
        }

    def clear(self) -> None:
        """Flush cache entries and reset counters."""
        self.entries.clear()
        self.hit_count = 0
        self.miss_count = 0

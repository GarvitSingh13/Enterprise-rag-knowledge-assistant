"""BM25 (Best Matching 25) Sparse Keyword Retriever.

Indexes document chunks using an inverted word index with Okapi BM25 scoring.
Specially tuned to preserve enterprise identifiers (acronyms, hyphenated error codes, SKUs).
"""

from typing import List, Tuple, Dict, Any, Optional
import re
from rank_bm25 import BM25Okapi
from enterprise_rag.core.models import Chunk, Department, ClearanceLevel


class BM25Retriever:
    """Sparse keyword retriever implementing BM25Okapi."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into search tokens.
        Preserves hyphenated terms, error codes (e.g. ERR-502-TIMEOUT),
        and alphanumeric codes (e.g. SKU-PROD-CLUSTER-A4).
        """
        # Match alphanumeric sequences with optional internal hyphens or underscores
        raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower())
        # Filter single character punctuation artifacts
        return [t for t in raw_tokens if len(t) > 1]

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build the BM25 inverted index across provided chunks."""
        self.chunks = chunks
        self.corpus_tokens = [self.tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        department: Optional[Department] = None,
        clearance_level: Optional[ClearanceLevel] = None,
        allowed_roles: Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Execute BM25 keyword search with metadata pre-filtering.

        Returns:
            List of (Chunk, bm25_score) tuples sorted in descending order of score.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Raw scores from BM25Okapi
        scores = self.bm25.get_scores(query_tokens)

        results: List[Tuple[Chunk, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue

            chunk = self.chunks[idx]

            # 1. Department filter
            if department and chunk.metadata.department != department:
                continue

            # 2. Clearance filter
            if clearance_level and chunk.metadata.clearance_level != clearance_level:
                continue

            # 3. RBAC role filter (chunk's allowed roles must intersect user's allowed roles)
            if allowed_roles is not None:
                if not any(role in chunk.metadata.allowed_roles for role in allowed_roles):
                    continue

            results.append((chunk, float(score)))

        # Sort descending by BM25 score
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

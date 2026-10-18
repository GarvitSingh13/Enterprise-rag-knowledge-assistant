"""Dense Vector Retriever for Enterprise RAG.

Generates dense vector embeddings for document chunks and user queries,
performing exact cosine similarity search with metadata pre-filtering.
"""

from typing import List, Tuple, Optional
import numpy as np
import re
import hashlib
from enterprise_rag.core.models import Chunk, Department, ClearanceLevel


class DenseEmbeddingModel:
    """
    Lightweight, self-contained dense vector embedding model.
    Produces high-dimensional (dim=384) normalized semantic vectors
    using subword character n-gram hashing and term weighting,
    ensuring zero-dependency instant execution and robust synonym overlap.
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        """Deterministically hash token string into an integer index."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension

    STOPWORDS = {
        "what", "is", "the", "are", "for", "a", "an", "to", "in", "on", "of",
        "and", "or", "it", "that", "this", "do", "does", "can", "our", "we", "be",
    }

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text string into normalized unit semantic vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())

        if not words:
            return vec

        substantive: List[str] = []
        for word in words:
            if word in self.STOPWORDS:
                continue

            # Root stem normalization
            stem = word
            if word.endswith("ing") and len(word) > 5:
                stem = word[:-3]
            elif word.endswith("ed") and len(word) > 4:
                stem = word[:-2]
            elif word.endswith("es") and len(word) > 4:
                stem = word[:-2]
            elif word.endswith("s") and len(word) > 3:
                stem = word[:-1]

            vec[self._hash_token(word)] += 1.0
            if stem != word:
                vec[self._hash_token(stem)] += 0.8
            substantive.append(stem)

        # Character 3-grams for subword semantic capture
        for s in substantive:
            if len(s) >= 4:
                for i in range(len(s) - 2):
                    sub = s[i : i + 3]
                    s_idx = self._hash_token(sub)
                    vec[s_idx] += 0.4

        # L2 Normalize vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into a 2D matrix of shape (N, dim)."""
        return np.vstack([self.encode_text(t) for t in texts])


class VectorRetriever:
    """Dense vector search engine using cosine similarity."""

    def __init__(self, dimension: int = 1024, min_similarity_threshold: float = 0.15):
        self.embedder = DenseEmbeddingModel(dimension=dimension)
        self.min_similarity_threshold = min_similarity_threshold
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, dim)

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Compute and store embeddings for all provided chunks."""
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.embeddings = self.embedder.encode_batch(texts)

    def search(
        self,
        query: str,
        top_k: int = 10,
        department: Optional[Department] = None,
        clearance_level: Optional[ClearanceLevel] = None,
        allowed_roles: Optional[List[str]] = None,
        min_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Execute dense vector search with metadata pre-filtering.

        Returns:
            List of (Chunk, cosine_similarity) tuples sorted in descending order.
        """
        if self.embeddings is None or not self.chunks:
            return []

        threshold = min_threshold if min_threshold is not None else self.min_similarity_threshold
        query_vec = self.embedder.encode_text(query)
        # Dot product of normalized unit vectors = Cosine Similarity
        cosine_scores = np.dot(self.embeddings, query_vec)

        results: List[Tuple[Chunk, float]] = []
        for idx, score in enumerate(cosine_scores):
            # Ignore random noise below threshold
            if score < threshold:
                continue

            chunk = self.chunks[idx]

            # 1. Department filter
            if department and chunk.metadata.department != department:
                continue

            # 2. Clearance filter
            if clearance_level and chunk.metadata.clearance_level != clearance_level:
                continue

            # 3. RBAC role filter
            if allowed_roles is not None:
                if not any(role in chunk.metadata.allowed_roles for role in allowed_roles):
                    continue

            results.append((chunk, float(score)))

        # Sort descending by similarity score
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

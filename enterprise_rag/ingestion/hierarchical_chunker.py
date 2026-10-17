"""Hierarchical (Parent-Child) Chunker for Enterprise RAG.

Splits documents into:
1. Large Parent Chunks (1000-1500 chars): Retains rich context, tables, and narrative flow.
2. Small Child Chunks (250-400 chars): Optimized for precise keyword (BM25) and vector retrieval.

Each child chunk maintains an explicit pointer (parent_id) to its parent chunk.
"""

from typing import List, Tuple
import re
from enterprise_rag.core.models import RawDocument, Chunk, ChunkType
from enterprise_rag.core.config import ChunkingConfig


class HierarchicalChunker:
    """Creates parent-child chunk hierarchies with boundary-aware text splitting."""

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()

    def _split_into_units(self, text: str) -> List[str]:
        """Split text by paragraphs, markdown headers, and sentence boundaries."""
        # Split by double newline first (paragraphs)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        units: List[str] = []

        for para in paragraphs:
            if len(para) <= self.config.child_chunk_size:
                units.append(para)
            else:
                # Split large paragraphs into sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    s_clean = sentence.strip()
                    if s_clean:
                        units.append(s_clean)

        return units if units else [text]

    def _create_sliding_chunks(self, text: str, target_size: int, overlap: int) -> List[str]:
        """Generate overlapping chunks using natural semantic units (paragraphs/sentences)."""
        units = self._split_into_units(text)
        chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)
            if current_len + unit_len + 1 <= target_size:
                current_chunk_parts.append(unit)
                current_len += unit_len + 1
            else:
                if current_chunk_parts:
                    chunk_text = " ".join(current_chunk_parts).strip()
                    if len(chunk_text) >= self.config.min_chunk_length:
                        chunks.append(chunk_text)

                # Overlap logic: keep trailing units that fit in the overlap window
                overlap_parts: List[str] = []
                overlap_len = 0
                for part in reversed(current_chunk_parts):
                    if overlap_len + len(part) + 1 <= overlap:
                        overlap_parts.insert(0, part)
                        overlap_len += len(part) + 1
                    else:
                        break

                current_chunk_parts = overlap_parts + [unit]
                current_len = sum(len(p) for p in current_chunk_parts) + len(current_chunk_parts)

        if current_chunk_parts:
            chunk_text = " ".join(current_chunk_parts).strip()
            if len(chunk_text) >= self.config.min_chunk_length:
                chunks.append(chunk_text)

        # Fallback if text was not split
        if not chunks and len(text.strip()) >= self.config.min_chunk_length:
            chunks.append(text.strip())

        return chunks

    def chunk_document(self, document: RawDocument) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Produce hierarchical chunks for a document.

        Returns:
            Tuple[List[Chunk], List[Chunk]]: (parent_chunks, child_chunks)
        """
        parent_texts = self._create_sliding_chunks(
            text=document.content,
            target_size=self.config.parent_chunk_size,
            overlap=self.config.parent_chunk_overlap,
        )

        parent_chunks: List[Chunk] = []
        child_chunks: List[Chunk] = []
        child_global_index = 0

        for parent_idx, p_text in enumerate(parent_texts):
            parent_chunk = Chunk(
                content=p_text,
                chunk_type=ChunkType.PARENT,
                chunk_index=parent_idx,
                metadata=document.metadata.model_copy(),
                parent_id=None,
            )
            parent_chunks.append(parent_chunk)

            # Sub-chunk the parent chunk into child chunks
            child_texts = self._create_sliding_chunks(
                text=p_text,
                target_size=self.config.child_chunk_size,
                overlap=self.config.child_chunk_overlap,
            )

            for c_text in child_texts:
                child_chunk = Chunk(
                    content=c_text,
                    chunk_type=ChunkType.CHILD,
                    chunk_index=child_global_index,
                    metadata=document.metadata.model_copy(),
                    parent_id=parent_chunk.id,  # Explicit link to parent context
                )
                child_chunks.append(child_chunk)
                child_global_index += 1

        return parent_chunks, child_chunks

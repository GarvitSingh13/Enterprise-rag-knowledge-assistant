"""Ingestion Pipeline Coordinator.

Orchestrates loading, metadata extraction, hierarchical chunking,
and parent-child cataloging for enterprise knowledge corpora.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from enterprise_rag.core.models import RawDocument, Chunk, ChunkType
from enterprise_rag.core.config import settings, ChunkingConfig
from enterprise_rag.ingestion.document_loader import DocumentLoader
from enterprise_rag.ingestion.hierarchical_chunker import HierarchicalChunker


class IngestionPipeline:
    """Manages the full offline document ingestion lifecycle."""

    def __init__(self, chunking_config: ChunkingConfig = None):
        self.chunker = HierarchicalChunker(config=chunking_config or settings.chunking)
        self.parent_store: Dict[str, Chunk] = {}
        self.child_store: List[Chunk] = []
        self.raw_documents: List[RawDocument] = []

    def process_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Ingest all enterprise documents from a directory."""
        self.raw_documents = DocumentLoader.load_directory(dir_path)
        all_parents: List[Chunk] = []
        all_children: List[Chunk] = []

        dept_stats: Dict[str, int] = {}
        clearance_stats: Dict[str, int] = {}

        for doc in self.raw_documents:
            # Stats tracking
            dept = doc.metadata.department.value
            clearance = doc.metadata.clearance_level.value
            dept_stats[dept] = dept_stats.get(dept, 0) + 1
            clearance_stats[clearance] = clearance_stats.get(clearance, 0) + 1

            # Chunking
            parents, children = self.chunker.chunk_document(doc)
            all_parents.extend(parents)
            all_children.extend(children)

            for p in parents:
                self.parent_store[p.id] = p

        self.child_store = all_children

        # Export catalog for persistence
        self._save_catalog(settings.processed_chunks_dir)

        return {
            "total_documents": len(self.raw_documents),
            "total_parent_chunks": len(all_parents),
            "total_child_chunks": len(all_children),
            "department_breakdown": dept_stats,
            "clearance_breakdown": clearance_stats,
            "average_children_per_parent": round(len(all_children) / max(1, len(all_parents)), 2),
        }

    def _save_catalog(self, output_dir: Path) -> None:
        """Persist parent and child chunks into JSON catalogs."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save parent map: {parent_id: chunk_dict}
        parent_export = {
            pid: chunk.model_dump() for pid, chunk in self.parent_store.items()
        }
        with open(output_dir / "parent_catalog.json", "w", encoding="utf-8") as f:
            json.dump(parent_export, f, indent=2)

        # Save child list: [child_dict, ...]
        child_export = [c.model_dump() for c in self.child_store]
        with open(output_dir / "child_catalog.json", "w", encoding="utf-8") as f:
            json.dump(child_export, f, indent=2)

    def get_parent_context(self, parent_id: str) -> str:
        """Fetch the full rich context of a parent chunk by ID."""
        chunk = self.parent_store.get(parent_id)
        return chunk.content if chunk else ""

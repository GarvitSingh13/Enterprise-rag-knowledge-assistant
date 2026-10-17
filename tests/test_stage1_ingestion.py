"""Stage 1 Verification Tests: Ingestion, Metadata Extraction & Hierarchical Chunking."""

import pytest
from pathlib import Path
from enterprise_rag.core.models import Department, ClearanceLevel, ChunkType, RawDocument, DocumentMetadata
from enterprise_rag.core.config import settings, ChunkingConfig
from enterprise_rag.ingestion.document_loader import DocumentLoader
from enterprise_rag.ingestion.metadata_extractor import MetadataExtractor
from enterprise_rag.ingestion.hierarchical_chunker import HierarchicalChunker
from enterprise_rag.ingestion.pipeline import IngestionPipeline


def test_document_loader_parses_frontmatter():
    """Verify frontmatter parsing and clean body extraction."""
    sample_text = """---
title: "Custom Security SOP"
department: "engineering"
clearance: "internal"
author: "Security Team"
tags: ["auth", "security"]
---
# Security Guide
This is the main document body with critical technical instructions.
"""
    frontmatter, body = DocumentLoader._parse_frontmatter(sample_text)
    assert frontmatter["title"] == "Custom Security SOP"
    assert frontmatter["department"] == "engineering"
    assert frontmatter["author"] == "Security Team"
    assert "auth" in frontmatter["tags"]
    assert body.startswith("# Security Guide")


def test_metadata_extractor_from_path():
    """Verify department and clearance level inference from path heuristics."""
    fake_path = Path("data/raw_docs/engineering/api_deployment_sop.md")
    meta = MetadataExtractor.extract_from_path(fake_path)

    assert meta.department == Department.ENGINEERING
    assert meta.clearance_level == ClearanceLevel.INTERNAL
    assert meta.document_type == "sop"
    assert "engineering" in meta.tags


def test_metadata_extractor_restricted_clearance():
    """Verify restricted clearance identification."""
    fake_path = Path("data/raw_docs/executive/executive_compensation_restricted.md")
    meta = MetadataExtractor.extract_from_path(fake_path)

    assert meta.department == Department.EXECUTIVE
    assert meta.clearance_level == ClearanceLevel.RESTRICTED
    assert "c_suite" in meta.allowed_roles


def test_hierarchical_chunker_parent_child_linking():
    """Verify hierarchical chunking generates parent chunks and correctly linked child chunks."""
    long_content = (
        "Enterprise API Gateway Deployment Standard Operating Procedure. " * 30
        + "\n\n"
        + "Incident Management Protocol and Severities. " * 30
    )
    doc = RawDocument(
        content=long_content,
        metadata=DocumentMetadata(
            title="Gateway SOP",
            department=Department.ENGINEERING,
            clearance_level=ClearanceLevel.INTERNAL,
        ),
    )

    chunker = HierarchicalChunker(
        config=ChunkingConfig(
            parent_chunk_size=600,
            parent_chunk_overlap=100,
            child_chunk_size=200,
            child_chunk_overlap=40,
            min_chunk_length=30,
        )
    )

    parents, children = chunker.chunk_document(doc)

    assert len(parents) > 0, "Should generate at least one parent chunk"
    assert len(children) > len(parents), "Should generate more child chunks than parent chunks"

    parent_ids = {p.id for p in parents}
    for child in children:
        assert child.chunk_type == ChunkType.CHILD
        assert child.parent_id in parent_ids, f"Child {child.id} must point to a valid parent"
        assert child.metadata.department == Department.ENGINEERING


def test_full_ingestion_pipeline():
    """Verify end-to-end ingestion pipeline across the sample enterprise repository."""
    pipeline = IngestionPipeline()
    stats = pipeline.process_directory(settings.raw_docs_dir)

    assert stats["total_documents"] >= 5, "Expected at least 5 enterprise documents"
    assert stats["total_parent_chunks"] > 0
    assert stats["total_child_chunks"] > 0
    assert "engineering" in stats["department_breakdown"]
    assert "hr" in stats["department_breakdown"]
    assert "finance" in stats["department_breakdown"]
    assert "executive" in stats["department_breakdown"]

    # Verify catalog persistence
    parent_catalog = settings.processed_chunks_dir / "parent_catalog.json"
    child_catalog = settings.processed_chunks_dir / "child_catalog.json"
    assert parent_catalog.exists(), "Parent catalog file must exist"
    assert child_catalog.exists(), "Child catalog file must exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

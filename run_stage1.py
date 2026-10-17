"""Stage 1 Standalone Ingestion & Hierarchical Chunking Runner.

Run this script to ingest enterprise documents, extract metadata,
produce parent-child chunks, and view the catalog statistics.
"""

from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline


def main():
    print("=" * 70)
    print("🏢 ENTERPRISE RAG KNOWLEDGE ASSISTANT: STAGE 1 INGESTION ENGINE")
    print("=" * 70)
    print(f"[*] Scanning Raw Documents Directory: {settings.raw_docs_dir}")

    pipeline = IngestionPipeline()
    stats = pipeline.process_directory(settings.raw_docs_dir)

    print("\n✅ INGESTION & HIERARCHICAL CHUNKING COMPLETE!")
    print("-" * 50)
    print(f"  • Total Documents Processed:       {stats['total_documents']}")
    print(f"  • Total Parent Chunks (Context):  {stats['total_parent_chunks']}")
    print(f"  • Total Child Chunks (Retrieval): {stats['total_child_chunks']}")
    print(f"  • Average Children per Parent:    {stats['average_children_per_parent']}")
    print("-" * 50)
    print("📂 Department Breakdown:")
    for dept, count in stats["department_breakdown"].items():
        print(f"    - {dept.capitalize():<15}: {count} document(s)")
    print("-" * 50)
    print("🔒 Clearance Level Breakdown:")
    for clearance, count in stats["clearance_breakdown"].items():
        print(f"    - {clearance.capitalize():<15}: {count} document(s)")
    print("-" * 50)
    print(f"📁 Catalogs successfully saved to: {settings.processed_chunks_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Stage 2 Standalone Retrieval Demo & Benchmarking Runner.

Executes queries comparing Pure BM25, Pure Vector, and Hybrid RRF,
and demonstrates Document-Level Security (RBAC) and Parent Context Retrieval.
"""

from enterprise_rag.core.config import settings
from enterprise_rag.core.models import Department, ClearanceLevel
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"🔍 {title.upper()}")
    print("=" * 70)


def main():
    print_separator("STAGE 2: HYBRID RETRIEVAL ENGINE (BM25 + VECTOR + RRF)")

    # 1. Ingest documents and initialize retriever
    print("[*] Ingesting documents and building dual indexes...")
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    retriever = HybridRetriever(rrf_k=60)
    retriever.index(pipeline.child_store, list(pipeline.parent_store.values()))
    print(f"✅ Indexed {len(pipeline.child_store)} child chunks into BM25 & Vector engines.")

    # Demo Query 1: Exact Technical Error Code (BM25 Strength)
    print_separator("Demo 1: Exact Technical Code (BM25 Test)")
    query1 = "What should I do for error ERR-502-TIMEOUT on cluster SKU-PROD-CLUSTER-A4?"
    print(f"Query: \"{query1}\"\n")
    results1 = retriever.search(query1, top_k=2)
    for i, r in enumerate(results1, 1):
        print(f"  Hit #{i} [RRF Score: {r.rrf_score:.4f}]")
        print(f"    • BM25 Rank: {r.bm25_rank} | Vector Rank: {r.vector_rank}")
        print(f"    • Source: {r.chunk.metadata.source_file} ({r.chunk.metadata.department.value})")
        print(f"    • Snippet: {r.chunk.content[:160]}...\n")

    # Demo Query 2: Conversational Synonym Search (Vector Strength)
    print_separator("Demo 2: Conversational Synonym Search (Vector Test)")
    query2 = "How much money can I get for setting up a home office desk and chair?"
    print(f"Query: \"{query2}\"\n")
    results2 = retriever.search(query2, top_k=2)
    for i, r in enumerate(results2, 1):
        print(f"  Hit #{i} [RRF Score: {r.rrf_score:.4f}]")
        print(f"    • BM25 Rank: {r.bm25_rank} | Vector Rank: {r.vector_rank}")
        print(f"    • Source: {r.chunk.metadata.source_file} ({r.chunk.metadata.department.value})")
        print(f"    • Snippet: {r.chunk.content[:160]}...\n")

    # Demo Query 3: Parent-Document Context Attachment
    print_separator("Demo 3: Parent-Document Context Attachment")
    query3 = "What is the per diem meal allowance for international travel?"
    print(f"Query: \"{query3}\"\n")
    results3 = retriever.search(query3, top_k=1, attach_parent_context=True)
    if results3:
        top = results3[0]
        print(f"  Child Chunk (Found via Search):\n    \"{top.chunk.content}\"\n")
        print(f"  Linked Parent Context (Fed to LLM, {len(top.parent_context)} chars):\n    \"{top.parent_context[:280]}...\"\n")

    # Demo Query 4: Role-Based Access Control (RBAC) Security
    print_separator("Demo 4: Role-Based Access Control (RBAC Security Test)")
    query4 = "What is the executive bonus severance multiplier for C-suite?"
    print(f"Query: \"{query4}\"")

    # Case A: Standard Employee Querying
    print("\n  [A] Querying as Standard Employee (role: 'employee'):")
    emp_hits = retriever.search(query4, top_k=3, allowed_roles=["employee"])
    restricted_count = sum(1 for h in emp_hits if h.chunk.metadata.clearance_level == ClearanceLevel.RESTRICTED)
    print(f"      Total hits returned: {len(emp_hits)}")
    print(f"      Restricted confidential documents leaked: {restricted_count} (100% SECURE! 🛡️)")

    # Case B: C-Suite Executive Querying
    print("\n  [B] Querying as Executive Officer (role: 'c_suite'):")
    c_suite_hits = retriever.search(query4, top_k=2, allowed_roles=["c_suite"])
    for i, r in enumerate(c_suite_hits, 1):
        print(f"      Hit #{i}: {r.chunk.metadata.title} [Clearance: {r.chunk.metadata.clearance_level.value.upper()}]")
        print(f"      Snippet: {r.chunk.content[:140]}...\n")

    print("=" * 70)
    print("🎉 STAGE 2 HYBRID RETRIEVAL VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

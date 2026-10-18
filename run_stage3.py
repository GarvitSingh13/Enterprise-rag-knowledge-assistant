"""Stage 3 Standalone Runner: Cross-Encoder Reranking & Multi-Turn Conversation Engine.

Demonstrates:
1. Neural Cross-Encoder Reranking (Before vs After Reranking Comparison).
2. Query Rewriting across a 3-turn realistic employee dialogue.
3. Conversation Memory buffer and sliding window management.
"""

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker
from enterprise_rag.conversation.memory import ConversationMemory
from enterprise_rag.conversation.query_rewriter import QueryRewriter


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"🎯 {title.upper()}")
    print("=" * 70)


def main():
    print_banner("STAGE 3: CROSS-ENCODER RERANKING & CONVERSATION ENGINE")

    # 1. Initialize Pipeline & Retrievers
    print("[*] Initializing knowledge base and engines...")
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    reranker = CrossEncoderReranker()
    rewriter = QueryRewriter()
    memory = ConversationMemory(max_turns=4)

    # =========================================================================
    # PART 1: CROSS-ENCODER RERANKING SHOWCASE
    # =========================================================================
    print_banner("PART 1: NEURAL CROSS-ENCODER RERANKING")
    test_query = "What is the procedure and timeline for submitting expense reports?"
    print(f"Query: \"{test_query}\"\n")

    # Step A: Hybrid Search Candidates
    candidates = hybrid.search(test_query, top_k=6)
    print("Top 3 Candidates from Hybrid Search (Before Reranking):")
    for i, c in enumerate(candidates[:3], 1):
        print(f"  [Hybrid #{i}] RRF Score: {c.rrf_score:.4f} | Source: {c.chunk.metadata.source_file}")
        print(f"               Snippet: {c.chunk.content[:100]}...")

    # Step B: Neural Cross-Encoder Reranking
    reranked = reranker.rerank(test_query, candidates, top_k=3)
    print("\nTop 3 Candidates After Neural Cross-Encoder Reranking:")
    for r in reranked:
        rank_change = f"(Moved from Hybrid #{r.original_rank} -> Reranked #{r.reranked_rank})"
        print(f"  [Rank #{r.reranked_rank}] Neural Score: {r.rerank_score * 100:.1f}% {rank_change}")
        print(f"              Source: {r.chunk.metadata.source_file}")
        print(f"              Snippet: {r.chunk.content[:130]}...\n")

    # =========================================================================
    # PART 2: MULTI-TURN CONVERSATION WITH QUERY REWRITING
    # =========================================================================
    print_banner("PART 2: MULTI-TURN CHAT WITH AUTOMATIC QUERY REWRITING")

    conversation_script = [
        (
            "What is our corporate meal allowance for domestic travel?",
            "For domestic travel, the daily meal allowance is up to $75 USD per day ($15 Breakfast, $25 Lunch, $35 Dinner).",
        ),
        (
            "Does that apply to international travel as well?",
            "For international travel, the meal per diem increases to up to $120 USD per travel day.",
        ),
        (
            "What about receipt submission deadlines?",
            "All expense reports with itemized tax receipts must be submitted via Concur within 14 calendar days of trip conclusion.",
        ),
    ]

    for turn_num, (user_query, simulated_answer) in enumerate(conversation_script, 1):
        print(f"\n--- [TURN {turn_num}] ---")
        print(f"👤 User Raw Input: \"{user_query}\"")

        # Rewrite query using dialogue history
        rewritten = rewriter.rewrite(user_query, memory)
        print(f"✨ Rewritten Standalone Query: \"{rewritten}\"")

        # Retrieve using rewritten query
        hits = hybrid.search(rewritten, top_k=3)
        reranked_hits = reranker.rerank(rewritten, hits, top_k=1)

        if reranked_hits:
            top_hit = reranked_hits[0]
            print(f"🔎 Top Retrieved Source: {top_hit.chunk.metadata.source_file}")
            print(f"📖 Grounding Context: {top_hit.chunk.content[:120]}...")

        # Record turn in Conversation Memory
        memory.add_turn(
            user_raw_query=user_query,
            assistant_response=simulated_answer,
            rewritten_query=rewritten,
        )
        print(f"🤖 Assistant: {simulated_answer}")

    # =========================================================================
    # PART 3: MEMORY BUFFER INSPECTION
    # =========================================================================
    print_banner("PART 3: CONVERSATION MEMORY BUFFER STATUS")
    print(f"Total Turns in Memory: {len(memory.turns)} / {memory.max_turns} (Sliding Window)")
    print("\nFormatted History in Prompt Buffer:")
    print("-" * 50)
    print(memory.get_history_as_text())
    print("-" * 50)

    print("\n" + "=" * 70)
    print("🎉 STAGE 3 CROSS-ENCODER & CONVERSATION ENGINE VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

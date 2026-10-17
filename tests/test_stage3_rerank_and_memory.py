"""Stage 3 Verification Tests: Cross-Encoder Reranker, Conversation Memory & Query Rewriting."""

import pytest
from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.retrieval.reranker import CrossEncoderReranker
from enterprise_rag.conversation.memory import ConversationMemory
from enterprise_rag.conversation.query_rewriter import QueryRewriter


@pytest.fixture(scope="module")
def prepared_system():
    """Ingest documents and initialize Hybrid Retriever and Reranker."""
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    reranker = CrossEncoderReranker()
    rewriter = QueryRewriter()

    return {
        "hybrid": hybrid,
        "reranker": reranker,
        "rewriter": rewriter,
    }


def test_cross_encoder_reranks_and_scores(prepared_system):
    """Verify that Cross-Encoder scores candidates and outputs RerankedResult objects."""
    hybrid = prepared_system["hybrid"]
    reranker = prepared_system["reranker"]

    query = "What is the domestic flight travel policy and booking standard?"
    candidates = hybrid.search(query, top_k=6)
    assert len(candidates) > 0

    reranked = reranker.rerank(query, candidates, top_k=3)
    assert len(reranked) <= 3
    assert len(reranked) > 0

    top = reranked[0]
    assert top.rerank_score > 0.0
    assert top.reranked_rank == 1
    assert "travel" in top.chunk.metadata.tags or "finance" in top.chunk.metadata.tags


def test_conversation_memory_sliding_window():
    """Verify sliding window drops oldest turn when max_turns is exceeded."""
    mem = ConversationMemory(max_turns=2)

    mem.add_turn("Question 1", "Answer 1")
    mem.add_turn("Question 2", "Answer 2")
    assert len(mem.turns) == 2

    mem.add_turn("Question 3", "Answer 3")
    assert len(mem.turns) == 2
    # Oldest turn (Question 1) must have been dropped
    assert mem.turns[0].user_raw_query == "Question 2"
    assert mem.turns[1].user_raw_query == "Question 3"

    history_text = mem.get_history_as_text()
    assert "Question 1" not in history_text
    assert "Question 2" in history_text
    assert "Question 3" in history_text


def test_query_rewriter_resolves_pronouns(prepared_system):
    """Verify that QueryRewriter transforms ambiguous follow-up questions."""
    rewriter = prepared_system["rewriter"]
    memory = ConversationMemory()

    # Turn 1
    memory.add_turn(
        user_raw_query="What is the travel meal per diem allowance?",
        assistant_response="Domestic travel meal per diem is up to $75 USD per day.",
    )

    # Turn 2: User asks follow-up with pronoun 'that'
    raw_follow_up = "Does that apply to international travel as well?"
    rewritten = rewriter.rewrite(raw_follow_up, memory)

    assert "travel" in rewritten.lower() or "meal" in rewritten.lower()
    assert "international" in rewritten.lower()
    assert "that" not in rewritten.lower(), "Pronoun 'that' should be replaced by topic"


def test_end_to_end_conversational_retrieval(prepared_system):
    """Verify multi-turn interaction: follow-up query retrieves correct document."""
    hybrid = prepared_system["hybrid"]
    reranker = prepared_system["reranker"]
    rewriter = prepared_system["rewriter"]
    memory = ConversationMemory()

    # Turn 1: Ask about dental insurance
    t1_query = "What does our health insurance say about dental cleanings?"
    memory.add_turn(
        user_raw_query=t1_query,
        assistant_response="Dental benefits cover 100% of routine cleanings twice per year.",
    )

    # Turn 2: Vague follow-up
    t2_query = "What is the maximum annual dollar coverage for that?"
    t2_rewritten = rewriter.rewrite(t2_query, memory)

    # Search with rewritten query
    candidates = hybrid.search(t2_rewritten, top_k=5)
    reranked = reranker.rerank(t2_rewritten, candidates, top_k=2)

    assert len(reranked) > 0
    top_chunk = reranked[0].chunk
    assert "dental" in top_chunk.content.lower() or "health" in top_chunk.metadata.tags
    assert "2,000" in top_chunk.content or "maximum" in top_chunk.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

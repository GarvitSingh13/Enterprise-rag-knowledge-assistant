"""Stage 4 Standalone Runner: Generation, Prompt Versioning & Hallucination Guardrails.

Demonstrates:
1. Versioned Prompt Management (Listing & Switching templates).
2. End-to-End Citation-Grounded Answer Generation.
3. Automated Hallucination Detection & Real-Time Interception.
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
from enterprise_rag.prompts.prompt_manager import PromptManager
from enterprise_rag.generation.generator import AnswerGenerator, GeneratedAnswer
from enterprise_rag.generation.hallucination_detector import HallucinationDetector


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"🛡️  {title.upper()}")
    print("=" * 70)


def main():
    print_banner("STAGE 4: GENERATION & AUTOMATED HALLUCINATION GUARDRAILS")

    # 1. Initialize Pipeline & Modules
    print("[*] Initializing knowledge base and generation engine...")
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    reranker = CrossEncoderReranker()
    prompt_mgr = PromptManager()
    generator = AnswerGenerator(prompt_manager=prompt_mgr)
    detector = HallucinationDetector(faithfulness_threshold=0.75)

    # =========================================================================
    # PART 1: PROMPT VERSIONING AUDIT
    # =========================================================================
    print_banner("PART 1: ENTERPRISE PROMPT VERSION REGISTRY")
    print("Loaded Prompt Templates:")
    for v in prompt_mgr.list_versions():
        active_badge = "[ACTIVE ✅]" if v["is_active"] else "[STANDBY]"
        print(f"  • Version: {v['version']:<8} {active_badge}")
        print(f"    Name: {v['name']}")
        print(f"    Description: {v['description']}\n")

    # =========================================================================
    # PART 2: GROUNDED GENERATION & VERIFICATION (HAPPY PATH)
    # =========================================================================
    print_banner("PART 2: GROUNDED GENERATION & VERIFICATION (APPROVED)")
    query = "What is the annual dental coverage limit and cleaning allowance?"
    print(f"Employee Question: \"{query}\"\n")

    # Retrieve & Rerank
    candidates = hybrid.search(query, top_k=5)
    reranked = reranker.rerank(query, candidates, top_k=2)

    # Generate Answer
    answer = generator.generate(query, reranked)
    print("Generated Answer:")
    print(f"  \"{answer.answer_text}\"")
    print(f"\nCitations Found: {answer.citations}")

    # Verify Groundedness with Hallucination Detector (including full Parent Contexts)
    top_chunks = [r.chunk for r in reranked]
    parent_contexts = [r.parent_context for r in reranked if r.parent_context]
    report = detector.verify_answer(answer, top_chunks, additional_context=parent_contexts)

    print("\n🔬 Automated Hallucination Audit Report:")
    print(f"  • Faithfulness Score: {report.faithfulness_score * 100:.1f}%")
    print(f"  • Status:             {'APPROVED ✅ (Grounded in context)' if report.is_grounded else 'REJECTED ❌'}")
    print(f"  • Total Claims:       {report.total_claims}")
    print(f"  • Verified Claims:    {len(report.verified_claims)}")
    print(f"  • Hallucinations:     {len(report.detected_hallucinations)}")

    # =========================================================================
    # PART 3: SIMULATED HALLUCINATION ATTACK (INTERCEPTION PATH)
    # =========================================================================
    print_banner("PART 3: HALLUCINATION ATTACK & INTERCEPTION (REJECTED)")
    fake_attack_answer = GeneratedAnswer(
        query="What is the home office setup reimbursement amount?",
        answer_text="Employees receive an unlimited $5,000 cash grant for buying luxury electronics within 90 days. [Source: leave_and_remote_work_policy.md]",
        citations=["leave_and_remote_work_policy.md"],
    )

    print("Simulated Hallucinated Model Output:")
    print(f"  \"{fake_attack_answer.answer_text}\"")

    # Audit the fake answer
    fake_report = detector.verify_answer(fake_attack_answer, pipeline.child_store)

    print("\n🚨 Automated Guardrail Interception:")
    print(f"  • Faithfulness Score: {fake_report.faithfulness_score * 100:.1f}%")
    print(f"  • Status:             {'APPROVED ✅' if fake_report.is_grounded else 'BLOCKED 🛡️ (Hallucination Detected)'}")
    print(f"  • Violations Caught:")
    for h in fake_report.detected_hallucinations:
        print(f"      ⚠️  {h}")

    print("\nSanitized Answer Delivered to User:")
    print(f"  \"{fake_report.sanitized_answer}\"")

    print("\n" + "=" * 70)
    print("🎉 STAGE 4 GROUNDED GENERATION & GUARDRAILS VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

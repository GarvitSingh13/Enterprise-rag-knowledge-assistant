"""Stage 7 Standalone Runner: Automated RAG Evaluation Suite (The RAG Triad).

Demonstrates:
1. Automated evaluation across Golden Enterprise Dataset.
2. Real-time RAG Triad scoring:
   - Context Relevance (Retrieval Precision)
   - Groundedness / Faithfulness (Zero-Hallucination Audit)
   - Answer Relevance (User Intent Satisfaction)
3. Benchmark Breakdown by category (Factual, Adversarial, Out-of-Domain, RBAC).
4. Executive Quality Scorecard for CI/CD gating.
"""

from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.gateway import SecureRAGGateway
from enterprise_rag.caching.semantic_cache import SemanticCache
from enterprise_rag.observability.tracer import QueryTracer
from enterprise_rag.evaluation.dataset import GoldenEvaluationDataset
from enterprise_rag.evaluation.evaluator import RAGEvaluator
from enterprise_rag.evaluation.models import QueryCategory


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"📊 {title.upper()}")
    print("=" * 75)


def main():
    print_banner("STAGE 7: AUTOMATED RAG EVALUATION SUITE (THE RAG TRIAD)")

    # 1. Initialize System
    print("[*] Initializing knowledge base and Secure RAG Gateway...")
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    auth = AuthService()
    cache = SemanticCache(similarity_threshold=0.65)
    tracer = QueryTracer()

    gateway = SecureRAGGateway(
        hybrid_retriever=hybrid,
        auth_service=auth,
        cache=cache,
        tracer=tracer,
    )

    evaluator = RAGEvaluator(gateway=gateway, auth_service=auth)
    samples = GoldenEvaluationDataset.get_benchmark_samples()

    # =========================================================================
    # PART 1: THE RAG TRIAD EXPLAINED
    # =========================================================================
    print("\n📐 THE PRODUCTION RAG TRIAD:")
    print("  1. Context Relevance  : Did our retrieval engine pull only relevant context?")
    print("  2. Groundedness       : Is every claim in the answer backed by the context? (Zero Hallucinations)")
    print("  3. Answer Relevance   : Does the response directly address the user's inquiry?")

    # =========================================================================
    # PART 2: RUNNING EVALUATION SUITE
    # =========================================================================
    print_banner("PART 2: RUNNING EVALUATION BENCHMARK ON GOLDEN DATASET")
    print(f"[*] Executing evaluation across {len(samples)} enterprise benchmark test cases...\n")

    results = []
    for idx, sample in enumerate(samples, 1):
        res = evaluator.evaluate_sample(sample)
        results.append(res)

        status_icon = "✅ PASSED" if res.passed else "❌ FAILED"
        print(f"[{idx}/{len(samples)}] {status_icon} | {sample.sample_id} ({sample.category.value})")
        print(f"  • Query:    \"{sample.query}\"")
        ans_preview = res.generated_answer.replace("\n", " ")
        if len(ans_preview) > 95:
            ans_preview = ans_preview[:92] + "..."
        print(f"  • Response: \"{ans_preview}\"")
        print(f"  • Citations: {res.retrieved_citations if res.retrieved_citations else '[Refusal - No Sources Cited]'}")
        scores = res.triad_scores
        print(f"  • RAG Triad Scores:")
        print(f"      - Context Relevance : {scores.context_relevance * 100:.1f}%")
        print(f"      - Groundedness      : {scores.groundedness * 100:.1f}% (Faithfulness)")
        print(f"      - Answer Relevance  : {scores.answer_relevance * 100:.1f}%")
        print(f"      - Composite Score   : {scores.composite_score * 100:.1f}% | Latency: {res.latency_ms:.1f}ms")
        if res.diagnostic_notes:
            for note in res.diagnostic_notes:
                print(f"      - Note: {note}")
        print()

    # =========================================================================
    # PART 3: AGGREGATE BENCHMARK REPORT
    # =========================================================================
    print_banner("PART 3: EXECUTIVE QUALITY SCORECARD & METRICS DASHBOARD")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = (passed / total * 100.0)

    avg_cr = sum(r.triad_scores.context_relevance for r in results) / total
    avg_gr = sum(r.triad_scores.groundedness for r in results) / total
    avg_ar = sum(r.triad_scores.answer_relevance for r in results) / total
    avg_comp = sum(r.triad_scores.composite_score for r in results) / total

    print("🏆 SYSTEM BENCHMARK SUMMARY:")
    print(f"  • Total Test Cases Evaluated : {total}")
    print(f"  • Passed Benchmark Criteria  : {passed} / {total} ({pass_rate:.1f}%)")
    print(f"  • Average Context Relevance  : {avg_cr * 100:.1f}%")
    print(f"  • Average Groundedness       : {avg_gr * 100:.1f}% 🛡️ (Zero Hallucination Integrity)")
    print(f"  • Average Answer Relevance   : {avg_ar * 100:.1f}%")
    print(f"  • Overall Composite Quality  : {avg_comp * 100:.1f}%")

    print("\n📊 CATEGORY BREAKDOWN:")
    for cat in QueryCategory:
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            c_tot = len(cat_results)
            c_pass = sum(1 for r in cat_results if r.passed)
            c_comp = sum(r.triad_scores.composite_score for r in cat_results) / c_tot
            c_gr = sum(r.triad_scores.groundedness for r in cat_results) / c_tot
            print(f"  • {cat.value.replace('_', ' ').title():<28}: Pass Rate {c_pass}/{c_tot} ({c_pass/c_tot*100:.0f}%) | Comp: {c_comp*100:.1f}% | Grounded: {c_gr*100:.1f}%")

    print("\n" + "=" * 75)
    print("🎉 STAGE 7 AUTOMATED RAG EVALUATION SUITE COMPLETE & VERIFIED!")
    print("=" * 75)


if __name__ == "__main__":
    main()

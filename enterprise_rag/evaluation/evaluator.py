"""Automated RAG Evaluator & Benchmark Runner.

Executes test suites against the Secure RAG Gateway, calculates RAG Triad scores,
and generates production benchmark scorecards.
"""

import time
from typing import List, Dict, Optional, Any
from enterprise_rag.security.gateway import SecureRAGGateway
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.models import UserRole
from enterprise_rag.evaluation.models import (
    EvaluationSample,
    EvaluationResult,
    EvaluationBenchmarkReport,
    QueryCategory,
)
from enterprise_rag.evaluation.metrics import RAGTriadMetrics
from enterprise_rag.evaluation.dataset import GoldenEvaluationDataset


class RAGEvaluator:
    """Orchestrates automated evaluation benchmarks across the RAG system."""

    def __init__(
        self,
        gateway: SecureRAGGateway,
        auth_service: AuthService,
        metrics: Optional[RAGTriadMetrics] = None,
    ):
        self.gateway = gateway
        self.auth_service = auth_service
        self.metrics = metrics or RAGTriadMetrics()

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single test sample through the Secure RAG Gateway."""
        # 1. Resolve or create authentication token matching sample persona
        username = f"eval_{sample.sample_id.lower().replace('-', '_')}"
        role_enums = []
        for r in sample.user_roles:
            try:
                role_enums.append(UserRole(r) if isinstance(r, str) else r)
            except ValueError:
                role_enums.append(UserRole.EMPLOYEE)

        if username not in self.auth_service.users_db:
            user = self.auth_service.register_user(
                username=username,
                password="EvalTestPassword123!",
                full_name=f"Eval User ({sample.sample_id})",
                department=sample.user_department,
                roles=role_enums,
            )
        else:
            user = self.auth_service.users_db[username]

        token = self.auth_service.create_access_token(user)

        # 2. Execute query through Secure Gateway (bypass cache to measure live pipeline)
        start_time = time.time()
        response = self.gateway.query(
            token=token,
            query=sample.query,
            use_cache=False,
            session_id=f"eval_session_{sample.sample_id}",
        )
        elapsed_ms = (time.time() - start_time) * 1000.0

        # 3. Retrieve raw text contexts for context relevance and groundedness checks
        retrieved_contexts = [src.get("content", "") for src in response.sources]

        # 4. Calculate RAG Triad Metrics
        triad_score = self.metrics.compute_rag_triad(
            query=sample.query,
            retrieved_contexts=retrieved_contexts,
            answer=response.answer,
            is_refusal_expected=sample.is_refusal_expected,
        )

        # 5. Diagnostic checks
        notes = []
        is_refusal = response.is_refusal or RAGTriadMetrics.is_safe_refusal(response.answer)

        # Check refusal expectations
        if sample.is_refusal_expected and not is_refusal:
            notes.append("FAILED: Expected safe refusal, but system generated a response.")
            triad_score.passed = False
        elif not sample.is_refusal_expected and is_refusal:
            notes.append("FAILED: Expected factual answer, but system generated a refusal.")
            triad_score.passed = False

        # Check citations for factual samples
        if sample.expected_citations and not sample.is_refusal_expected:
            missing_cites = [
                exp for exp in sample.expected_citations
                if not any(exp in c for c in response.citations)
            ]
            if missing_cites:
                notes.append(f"WARNING: Missing expected citation(s): {missing_cites}")

        return EvaluationResult(
            sample_id=sample.sample_id,
            query=sample.query,
            category=sample.category,
            generated_answer=response.answer,
            retrieved_citations=response.citations,
            triad_scores=triad_score,
            is_refusal=is_refusal,
            expected_refusal=sample.is_refusal_expected,
            latency_ms=round(elapsed_ms, 2),
            passed=triad_score.passed,
            diagnostic_notes=notes,
        )

    def run_benchmark(
        self,
        samples: Optional[List[EvaluationSample]] = None,
    ) -> EvaluationBenchmarkReport:
        """
        Execute full benchmark across all evaluation samples and aggregate metrics.
        """
        eval_samples = samples or GoldenEvaluationDataset.get_benchmark_samples()
        results: List[EvaluationResult] = []

        for sample in eval_samples:
            res = self.evaluate_sample(sample)
            results.append(res)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100.0) if total > 0 else 0.0

        avg_cr = sum(r.triad_scores.context_relevance for r in results) / total if total > 0 else 0.0
        avg_gr = sum(r.triad_scores.groundedness for r in results) / total if total > 0 else 0.0
        avg_ar = sum(r.triad_scores.answer_relevance for r in results) / total if total > 0 else 0.0
        avg_comp = sum(r.triad_scores.composite_score for r in results) / total if total > 0 else 0.0

        # Category breakdown
        category_breakdown: Dict[str, Dict[str, float]] = {}
        for cat in QueryCategory:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                cat_count = len(cat_results)
                cat_pass = sum(1 for r in cat_results if r.passed)
                category_breakdown[cat.value] = {
                    "total": cat_count,
                    "passed": cat_pass,
                    "pass_rate_pct": round((cat_pass / cat_count * 100.0), 1),
                    "avg_composite": round(sum(r.triad_scores.composite_score for r in cat_results) / cat_count, 3),
                    "avg_groundedness": round(sum(r.triad_scores.groundedness for r in cat_results) / cat_count, 3),
                }

        return EvaluationBenchmarkReport(
            total_samples=total,
            passed_samples=passed,
            failed_samples=failed,
            pass_rate_pct=round(pass_rate, 2),
            avg_context_relevance=round(avg_cr, 3),
            avg_groundedness=round(avg_gr, 3),
            avg_answer_relevance=round(avg_ar, 3),
            avg_composite_score=round(avg_comp, 3),
            category_breakdown=category_breakdown,
            detailed_results=results,
        )

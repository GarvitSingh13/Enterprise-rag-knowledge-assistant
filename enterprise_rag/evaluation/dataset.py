"""Golden Enterprise Evaluation Dataset.

Curated benchmark dataset designed to stress-test enterprise RAG across:
1. In-domain factual accuracy
2. Adversarial hallucination resistance
3. Out-of-domain rejection
4. RBAC boundary enforcement
"""

from typing import List
from enterprise_rag.core.models import Department
from enterprise_rag.evaluation.models import EvaluationSample, QueryCategory


class GoldenEvaluationDataset:
    """Manages the benchmark evaluation suite samples."""

    @staticmethod
    def get_benchmark_samples() -> List[EvaluationSample]:
        """Return the standard enterprise evaluation dataset."""
        return [
            EvaluationSample(
                sample_id="EVAL-001-TRAVEL-MEALS",
                query="What is the domestic travel meal allowance per day?",
                category=QueryCategory.FACTUAL_IN_DOMAIN,
                expected_citations=["travel_expense_policy.md"],
                ground_truth_answer="Up to $75 USD per full travel day ($15 Breakfast, $25 Lunch, $35 Dinner).",
                is_refusal_expected=False,
                user_roles=["employee"],
                user_department=Department.HR,
                description="Checks factual retrieval of daily domestic per diem meal limits.",
            ),
            EvaluationSample(
                sample_id="EVAL-002-FLIGHT-BOOKING",
                query="What are the rules for booking domestic flights under 6 hours?",
                category=QueryCategory.FACTUAL_IN_DOMAIN,
                expected_citations=["travel_expense_policy.md"],
                ground_truth_answer="Must be booked via Navan at least 14 days in advance; economy class is mandatory for flights under 6 hours.",
                is_refusal_expected=False,
                user_roles=["employee"],
                user_department=Department.HR,
                description="Checks travel policy booking window and cabin class restrictions.",
            ),
            EvaluationSample(
                sample_id="EVAL-003-CANARY-DEPLOYMENT",
                query="What are the automated rollback triggers during an API canary deployment?",
                category=QueryCategory.FACTUAL_IN_DOMAIN,
                expected_citations=["api_deployment_sop.md"],
                ground_truth_answer="Rollback triggered if HTTP 5xx spikes above 0.5% over 3 minutes, readiness probes fail on >20% of pods, or P95 latency exceeds 500ms for 5 minutes.",
                is_refusal_expected=False,
                user_roles=["employee", "engineer"],
                user_department=Department.ENGINEERING,
                description="Checks technical SOP retrieval for microservice deployment rollback rules.",
            ),
            EvaluationSample(
                sample_id="EVAL-004-ANNUAL-LEAVE-CARRYOVER",
                query="What is the maximum number of unused annual leave days that can be carried over into the following year?",
                category=QueryCategory.FACTUAL_IN_DOMAIN,
                expected_citations=["leave_and_remote_work_policy.md"],
                ground_truth_answer="A maximum of 5 unused leave days may be carried over into the following year, expiring by March 31st.",
                is_refusal_expected=False,
                user_roles=["employee"],
                user_department=Department.HR,
                description="Checks HR policy retrieval on leave rollover entitlements.",
            ),
            EvaluationSample(
                sample_id="EVAL-005-ADVERSARIAL-GRANT",
                query="How do I claim the company $5,000 annual personal pet wellness stipend?",
                category=QueryCategory.ADVERSARIAL_HALLUCINATION,
                expected_citations=[],
                ground_truth_answer="Refusal expected; no such pet stipend exists in company policy.",
                is_refusal_expected=True,
                user_roles=["employee"],
                user_department=Department.GENERAL,
                description="Stress-tests hallucination detector against fabricated benefit claims.",
            ),
            EvaluationSample(
                sample_id="EVAL-006-OUT-OF-DOMAIN",
                query="Can you give me the best recipe for baking chocolate chip sourdough bread?",
                category=QueryCategory.OUT_OF_DOMAIN,
                expected_citations=[],
                ground_truth_answer="Refusal expected; baking recipes are outside enterprise documentation.",
                is_refusal_expected=True,
                user_roles=["employee"],
                user_department=Department.GENERAL,
                description="Tests out-of-domain safe rejection.",
            ),
            EvaluationSample(
                sample_id="EVAL-007-RBAC-BLOCKED-EMPLOYEE",
                query="What is the CEO severance multiplier and equity cliff schedule?",
                category=QueryCategory.RBAC_BOUNDARY,
                expected_citations=[],
                ground_truth_answer="Refusal expected; standard employees lack clearance for executive memos.",
                is_refusal_expected=True,
                user_roles=["employee"],
                user_department=Department.HR,
                description="Verifies that standard employee role is denied access to C-suite secrets.",
            ),
            EvaluationSample(
                sample_id="EVAL-008-RBAC-AUTHORIZED-CEO",
                query="What is the executive bonus severance multiplier and equity acceleration?",
                category=QueryCategory.RBAC_BOUNDARY,
                expected_citations=["executive_compensation_restricted.md"],
                ground_truth_answer="Severance payout equals 18 months base salary plus prorated bonus with double-trigger equity acceleration.",
                is_refusal_expected=False,
                user_roles=["c_suite", "admin", "employee"],
                user_department=Department.EXECUTIVE,
                description="Verifies that C-suite executive successfully retrieves restricted memos.",
            ),
        ]

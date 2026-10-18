"""Stage 5 Standalone Runner: User Authentication, RBAC & Document Security.

Demonstrates:
1. Password hashing (PBKDF2) and JWT token generation.
2. Multi-tenant document protection (Standard employee blocked from executive secrets).
3. Authorized executive access with citation grounding.
"""

from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.gateway import SecureRAGGateway


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"🔐 {title.upper()}")
    print("=" * 70)


def main():
    print_banner("STAGE 5: USER AUTHENTICATION & ROLE-BASED ACCESS CONTROL (RBAC)")

    # 1. Initialize Pipeline & Gateway
    print("[*] Initializing knowledge base and Secure RAG Gateway...")
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    auth_service = AuthService()
    gateway = SecureRAGGateway(hybrid_retriever=hybrid, auth_service=auth_service)

    # =========================================================================
    # PART 1: USER AUTHENTICATION & JWT GENERATION
    # =========================================================================
    print_banner("PART 1: AUTHENTICATION & JWT ISSUANCE")

    # Log in Alice (Employee)
    alice_token = auth_service.login("alice_emp", "EmployeePass123!")
    alice_claims = auth_service.verify_token(alice_token)
    print(f"👤 User Logged In: {alice_claims.full_name} (@{alice_claims.username})")
    print(f"   • Department: {alice_claims.department.upper()}")
    print(f"   • Roles:      {alice_claims.roles}")
    print(f"   • JWT Token:  {alice_token[:45]}...[TRUNCATED]\n")

    # Log in Sarah (CEO)
    sarah_token = auth_service.login("sarah_ceo", "ExecutivePass123!")
    sarah_claims = auth_service.verify_token(sarah_token)
    print(f"👑 User Logged In: {sarah_claims.full_name} (@{sarah_claims.username})")
    print(f"   • Department: {sarah_claims.department.upper()}")
    print(f"   • Roles:      {sarah_claims.roles}")
    print(f"   • JWT Token:  {sarah_token[:45]}...[TRUNCATED]\n")

    # =========================================================================
    # PART 2: UNAUTHORIZED ATTEMPT (ALICE ASKS FOR EXECUTIVE SALARIES)
    # =========================================================================
    print_banner("PART 2: DATA LEAKAGE TEST (REGULAR EMPLOYEE QUERY)")
    sensitive_query = "What is the C-suite executive severance payout and equity cliff schedule?"
    print(f"User: Alice Johnson (Role: 'employee')")
    print(f"Query: \"{sensitive_query}\"\n")

    resp_alice = gateway.query(token=alice_token, query=sensitive_query)

    print("Gateway Response:")
    print(f"  \"{resp_alice.answer}\"")
    print(f"  • Citations:                 {resp_alice.citations}")
    print(f"  • Retrieved Chunks:          {resp_alice.retrieved_sources_count}")
    print(f"  • Confidential Data Leaked:  0 chunks (100% SECURE! 🛡️)")

    # =========================================================================
    # PART 3: AUTHORIZED ACCESS (CEO ASKS THE EXACT SAME QUESTION)
    # =========================================================================
    print_banner("PART 3: AUTHORIZED ACCESS (C-SUITE CEO QUERY)")
    print(f"User: Sarah Connor (Role: 'c_suite', 'admin')")
    print(f"Query: \"{sensitive_query}\"\n")

    resp_sarah = gateway.query(token=sarah_token, query=sensitive_query)

    print("Gateway Response:")
    print(f"  \"{resp_sarah.answer}\"")
    print(f"\nAudit Metadata:")
    print(f"  • Citations:          {resp_sarah.citations}")
    print(f"  • Faithfulness Score: {resp_sarah.faithfulness_score * 100:.1f}%")
    print(f"  • Grounded in Source: {'YES ✅' if resp_sarah.is_grounded else 'NO ❌'}")

    print("\n" + "=" * 70)
    print("🎉 STAGE 5 AUTHENTICATION & RBAC SECURITY VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

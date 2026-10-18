"""Stage 5 Verification Tests: User Authentication, JWT Tokens, RBAC & Secure Gateway."""

import pytest
from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.models import UserRole
from enterprise_rag.security.gateway import SecureRAGGateway


@pytest.fixture(scope="module")
def prepared_secure_system():
    """Build and initialize end-to-end secure RAG gateway."""
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    auth_service = AuthService()
    gateway = SecureRAGGateway(hybrid_retriever=hybrid, auth_service=auth_service)

    return {
        "auth_service": auth_service,
        "gateway": gateway,
    }


def test_password_hashing_and_login_success(prepared_secure_system):
    """Verify PBKDF2 password hashing and valid JWT token issuance."""
    auth = prepared_secure_system["auth_service"]

    # Valid credentials
    token = auth.login("alice_emp", "EmployeePass123!")
    assert isinstance(token, str)
    assert len(token) > 20

    # Token verification
    payload = auth.verify_token(token)
    assert payload.username == "alice_emp"
    assert "employee" in payload.roles
    assert payload.department == "hr"


def test_authentication_failure_cases(prepared_secure_system):
    """Verify rejection of invalid credentials and tampered JWT tokens."""
    auth = prepared_secure_system["auth_service"]

    # Wrong password
    with pytest.raises(ValueError, match="Invalid username or password"):
        auth.login("alice_emp", "WrongPassword999!")

    # Non-existent user
    with pytest.raises(ValueError, match="Invalid username or password"):
        auth.login("ghost_user", "AnyPassword123!")

    # Tampered token
    valid_token = auth.login("alice_emp", "EmployeePass123!")
    tampered_token = valid_token[:-4] + "fake"
    with pytest.raises(ValueError, match="Invalid or tampered token"):
        auth.verify_token(tampered_token)


def test_rbac_isolation_standard_employee(prepared_secure_system):
    """Verify standard employee is blocked from accessing restricted executive memos."""
    auth = prepared_secure_system["auth_service"]
    gateway = prepared_secure_system["gateway"]

    # Log in as Alice (Regular Employee)
    alice_token = auth.login("alice_emp", "EmployeePass123!")

    # Attempt to ask about confidential executive compensation
    response = gateway.query(
        token=alice_token,
        query="What is the CEO and C-suite severance multiplier and equity cliff schedule?",
    )

    # Must NOT cite or retrieve executive restricted documents
    for cite in response.citations:
        assert "executive" not in cite.lower(), "Standard employee must never receive executive documents"
    assert "executive_compensation_restricted.md" not in response.citations


def test_rbac_authorization_c_suite_access(prepared_secure_system):
    """Verify C-Suite executive successfully retrieves confidential memos."""
    auth = prepared_secure_system["auth_service"]
    gateway = prepared_secure_system["gateway"]

    # Log in as Sarah (CEO / C-Suite)
    ceo_token = auth.login("sarah_ceo", "ExecutivePass123!")

    response = gateway.query(
        token=ceo_token,
        query="What is the executive bonus severance multiplier and equity acceleration?",
    )

    assert not response.is_refusal
    assert len(response.citations) > 0
    assert "executive_compensation_restricted.md" in response.citations
    assert "severance" in response.answer.lower() or "acceleration" in response.answer.lower()


def test_secure_gateway_conversational_continuity(prepared_secure_system):
    """Verify multi-turn query rewriting inside the Secure RAG Gateway."""
    auth = prepared_secure_system["auth_service"]
    gateway = prepared_secure_system["gateway"]

    token = auth.login("alice_emp", "EmployeePass123!")
    session_id = "test-session-alice-1"

    # Turn 1
    resp1 = gateway.query(
        token=token,
        query="What is our home office equipment allowance?",
        session_id=session_id,
    )
    assert "500" in resp1.answer

    # Turn 2: Follow-up question using pronoun 'that'
    resp2 = gateway.query(
        token=token,
        query="What is the deadline for submitting receipts for that?",
        session_id=session_id,
    )
    assert resp2.rewritten_query != "What is the deadline for submitting receipts for that?"
    assert "60 days" in resp2.answer or "receipts" in resp2.answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

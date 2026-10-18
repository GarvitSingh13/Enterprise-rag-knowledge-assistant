"""Stage 8 Unit & Integration Tests: Interactive Web Application & REST API Server.

Validates:
1. HTTP Server initialization and static asset delivery.
2. User Authentication and JWT token issuance (/api/auth/login).
3. End-to-end RAG chat queries with telemetry (/api/chat/query).
4. Multi-tenant RBAC clearance enforcement over HTTP.
5. Live observability metrics and RAG Triad benchmark endpoints.
"""

import threading
import time
import json
import urllib.request
import urllib.error
import pytest
from enterprise_rag.api.server import create_app_server, EnterpriseAPIServer


TEST_HOST = "127.0.0.1"
TEST_PORT = 8999


@pytest.fixture(scope="module")
def running_api_server():
    """Launch API server in background daemon thread for integration testing."""
    server: EnterpriseAPIServer = create_app_server(host=TEST_HOST, port=TEST_PORT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)  # Allow server to bind and start listening

    yield server

    server.shutdown()


def http_request(url: str, method: str = "GET", body: dict = None, headers: dict = None):
    """Helper sending HTTP requests and parsing JSON responses."""
    req_headers = {"User-Agent": "PytestClient"}
    if headers:
        req_headers.update(headers)

    data_bytes = None
    if body is not None:
        data_bytes = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            return status, content_type, content
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def test_api_server_startup_and_static_assets(running_api_server):
    """Verify HTTP server serves SPA index.html, style.css, and app.js."""
    base_url = running_api_server.url

    # 1. Root SPA HTML
    status, ctype, body = http_request(f"{base_url}/")
    assert status == 200
    assert "text/html" in ctype
    assert b"Enterprise RAG Assistant" in body

    # 2. CSS Stylesheet
    status, ctype, body = http_request(f"{base_url}/static/style.css")
    assert status == 200
    assert "text/css" in ctype
    assert b":root" in body

    # 3. JavaScript Controller
    status, ctype, body = http_request(f"{base_url}/static/app.js")
    assert status == 200
    assert "javascript" in ctype
    assert b"ENTERPRISE_PERSONAS" in body


def test_api_auth_login_endpoints(running_api_server):
    """Verify /api/auth/login handles credentials and issues signed JWTs."""
    base_url = running_api_server.url

    # Valid login for Alice
    status, _, body = http_request(
        f"{base_url}/api/auth/login",
        method="POST",
        body={"username": "alice_emp", "password": "EmployeePass123!"},
    )
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert "token" in data
    assert data["user"]["username"] == "alice_emp"
    assert "employee" in data["user"]["roles"]

    # Invalid login password
    status, _, body = http_request(
        f"{base_url}/api/auth/login",
        method="POST",
        body={"username": "alice_emp", "password": "WrongPassword!"},
    )
    assert status == 401

    # Get personas list
    status, _, body = http_request(f"{base_url}/api/auth/users")
    assert status == 200
    users_data = json.loads(body.decode("utf-8"))
    assert len(users_data["users"]) >= 4


def test_api_chat_query_authenticated_retrieval(running_api_server):
    """Verify /api/chat/query executes grounded RAG and semantic caching."""
    base_url = running_api_server.url

    # 1. Login as Alice
    _, _, login_body = http_request(
        f"{base_url}/api/auth/login",
        method="POST",
        body={"username": "alice_emp", "password": "EmployeePass123!"},
    )
    token = json.loads(login_body.decode("utf-8"))["token"]

    # 2. Cold Query (Cache Miss)
    status, _, q_body = http_request(
        f"{base_url}/api/chat/query",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
        body={"query": "What is the domestic travel meal allowance per day?", "use_cache": True},
    )
    assert status == 200
    res = json.loads(q_body.decode("utf-8"))
    assert "$75" in res["answer"]
    assert "travel_expense_policy.md" in res["citations"]
    assert res["is_grounded"] is True

    # 3. Warm Query (Cache Hit)
    status, _, q_body2 = http_request(
        f"{base_url}/api/chat/query",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
        body={"query": "What is the domestic travel meal allowance per day?", "use_cache": True},
    )
    assert status == 200
    res2 = json.loads(q_body2.decode("utf-8"))
    assert res2["cache_hit"] is True
    assert res2["total_latency_ms"] <= res["total_latency_ms"]


def test_api_chat_rbac_clearance_isolation(running_api_server):
    """Verify multi-tenant RBAC enforcement over the REST API."""
    base_url = running_api_server.url

    # Alice (Employee) Login
    _, _, a_login = http_request(
        f"{base_url}/api/auth/login",
        method="POST",
        body={"username": "alice_emp", "password": "EmployeePass123!"},
    )
    alice_token = json.loads(a_login.decode("utf-8"))["token"]

    # Sarah (CEO) Login
    _, _, s_login = http_request(
        f"{base_url}/api/auth/login",
        method="POST",
        body={"username": "sarah_ceo", "password": "ExecutivePass123!"},
    )
    sarah_token = json.loads(s_login.decode("utf-8"))["token"]

    sensitive_q = "What is the executive bonus severance multiplier and equity acceleration?"

    # 1. Alice queries executive secret -> Refused / Blocked by RBAC
    status, _, a_resp_body = http_request(
        f"{base_url}/api/chat/query",
        method="POST",
        headers={"Authorization": f"Bearer {alice_token}"},
        body={"query": sensitive_q, "use_cache": False},
    )
    assert status == 200
    a_resp = json.loads(a_resp_body.decode("utf-8"))
    assert a_resp["is_refusal"] is True
    assert "executive_compensation_restricted.md" not in a_resp["citations"]

    # 2. Sarah queries executive secret -> Authorized and Grounded
    status, _, s_resp_body = http_request(
        f"{base_url}/api/chat/query",
        method="POST",
        headers={"Authorization": f"Bearer {sarah_token}"},
        body={"query": sensitive_q, "use_cache": False},
    )
    assert status == 200
    s_resp = json.loads(s_resp_body.decode("utf-8"))
    assert s_resp["is_refusal"] is False
    assert "executive_compensation_restricted.md" in s_resp["citations"]


def test_api_telemetry_and_benchmark_endpoints(running_api_server):
    """Verify /api/observability/metrics, /api/corpus/documents, and /api/evaluation/benchmark."""
    base_url = running_api_server.url

    # 1. Telemetry metrics
    status, _, body = http_request(f"{base_url}/api/observability/metrics")
    assert status == 200
    tel = json.loads(body.decode("utf-8"))
    assert "telemetry" in tel
    assert "semantic_cache" in tel
    assert tel["telemetry"]["total_queries_tracked"] > 0

    # 2. Corpus catalog
    status, _, body = http_request(f"{base_url}/api/corpus/documents")
    assert status == 200
    corp = json.loads(body.decode("utf-8"))
    assert corp["total_documents"] >= 6
    assert corp["total_parent_chunks"] > 0

    # 3. RAG Triad benchmark
    status, _, body = http_request(f"{base_url}/api/evaluation/benchmark")
    assert status == 200
    bench = json.loads(body.decode("utf-8"))
    assert bench["total_samples"] >= 8
    assert bench["pass_rate_pct"] >= 85.0
    assert bench["avg_groundedness"] >= 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

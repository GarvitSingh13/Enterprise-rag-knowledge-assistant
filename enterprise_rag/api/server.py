"""Production-Grade Multi-Threaded HTTP API Server for Enterprise RAG.

Provides RESTful endpoints for:
1. Authentication & JWT issuing (/api/auth/login, /api/auth/users)
2. Secure RAG Query Execution (/api/chat/query)
3. Live Observability & Telemetry (/api/observability/metrics)
4. RAG Triad Benchmark Reports (/api/evaluation/benchmark)
5. Ingested Corpus Catalog (/api/corpus/documents)
6. Serving the Interactive Web Console Single Page Application
"""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import os
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.parse

from enterprise_rag.core.config import settings
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.retrieval.hybrid_retriever import HybridRetriever
from enterprise_rag.security.auth import AuthService
from enterprise_rag.security.gateway import SecureRAGGateway
from enterprise_rag.caching.semantic_cache import SemanticCache
from enterprise_rag.observability.tracer import QueryTracer
from enterprise_rag.evaluation.evaluator import RAGEvaluator
from enterprise_rag.evaluation.dataset import GoldenEvaluationDataset


STATIC_DIR = Path(__file__).parent / "static"


class EnterpriseRequestHandler(BaseHTTPRequestHandler):
    """Handles REST API requests and serves static web assets."""

    # Injected server-wide dependencies
    gateway: SecureRAGGateway = None
    auth_service: AuthService = None
    cache: SemanticCache = None
    tracer: QueryTracer = None
    evaluator: RAGEvaluator = None
    pipeline: IngestionPipeline = None

    def log_message(self, format, *args):
        """Custom request logging to keep console clean."""
        # Suppress standard logging for static files
        if args and any(str(args[0]).startswith(p) for p in ["GET /static", "GET /favicon"]):
            return
        super().log_message(format, *args)

    def _send_cors_headers(self):
        """Attach standard CORS headers to response."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        """Send formatted JSON payload."""
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, status_code: int, message: str):
        """Send standard JSON error envelope."""
        self._send_json_response(status_code, {"error": message, "status": "error"})

    def do_OPTIONS(self):
        """Respond to preflight CORS checks."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Route GET endpoints and serve static frontend."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # 1. API: List Enterprise Personas
        if path == "/api/auth/users":
            users = []
            for u in self.auth_service.users_db.values():
                users.append({
                    "username": u.username,
                    "full_name": u.full_name,
                    "department": u.department.value,
                    "roles": [r.value if hasattr(r, "value") else str(r) for r in u.roles],
                })
            self._send_json_response(200, {"users": users})
            return

        # 2. API: Ingested Corpus Catalog
        if path == "/api/corpus/documents":
            docs = []
            for doc in self.pipeline.raw_documents:
                docs.append({
                    "file_name": doc.metadata.source_file,
                    "title": doc.metadata.title,
                    "department": doc.metadata.department.value,
                    "clearance_level": doc.metadata.clearance_level.value,
                    "allowed_roles": doc.metadata.allowed_roles,
                    "character_count": doc.character_count,
                })
            self._send_json_response(200, {
                "total_documents": len(docs),
                "total_parent_chunks": len(self.pipeline.parent_store),
                "total_child_chunks": len(self.pipeline.child_store),
                "documents": docs,
            })
            return

        # 3. API: Observability Telemetry Metrics
        if path == "/api/observability/metrics":
            tracer_metrics = self.tracer.get_summary_metrics()
            cache_stats = self.cache.get_stats()
            self._send_json_response(200, {
                "telemetry": tracer_metrics,
                "semantic_cache": cache_stats,
            })
            return

        # 4. API: RAG Triad Evaluation Benchmark
        if path == "/api/evaluation/benchmark":
            report = self.evaluator.run_benchmark()
            self._send_json_response(200, report.model_dump())
            return

        # 5. Serve Web Application Static Files (HTML / CSS / JS)
        file_path: Path
        if path in ["/", "/index.html"]:
            file_path = STATIC_DIR / "index.html"
        elif path.startswith("/static/"):
            rel_name = path[len("/static/"):]
            file_path = STATIC_DIR / rel_name
        else:
            rel_name = path.lstrip("/")
            file_path = STATIC_DIR / rel_name

        if file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                self._send_error_json(500, f"Error reading asset: {str(e)}")
                return

        self._send_error_json(404, f"Endpoint or asset '{path}' not found.")

    def do_POST(self):
        """Route POST endpoints (authentication and query execution)."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Read JSON body
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_error_json(400, "Invalid JSON body.")
            return

        # 1. API: User Login
        if path == "/api/auth/login":
            username = body.get("username", "").strip()
            password = body.get("password", "")

            if not username or not password:
                self._send_error_json(400, "Fields 'username' and 'password' are required.")
                return

            try:
                token = self.auth_service.login(username, password)
                token_payload = self.auth_service.verify_token(token)
                self._send_json_response(200, {
                    "token": token,
                    "token_type": "Bearer",
                    "user": {
                        "username": token_payload.username,
                        "full_name": token_payload.full_name,
                        "department": token_payload.department,
                        "roles": token_payload.roles,
                    },
                })
                return
            except ValueError as e:
                self._send_error_json(401, str(e))
                return

        # 2. API: Secure RAG Chat Query
        if path == "/api/chat/query":
            # Extract Bearer token
            auth_header = self.headers.get("Authorization", "")
            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
            elif "token" in body:
                token = body.get("token", "")

            if not token:
                self._send_error_json(401, "Missing Authorization Bearer token.")
                return

            query_text = body.get("query", "").strip()
            if not query_text:
                self._send_error_json(400, "Field 'query' cannot be empty.")
                return

            use_cache = bool(body.get("use_cache", True))
            session_id = body.get("session_id")

            try:
                resp = self.gateway.query(
                    token=token,
                    query=query_text,
                    session_id=session_id,
                    use_cache=use_cache,
                )
                self._send_json_response(200, resp.model_dump())
                return
            except ValueError as e:
                self._send_error_json(401, f"Authentication/Security error: {str(e)}")
                return
            except Exception as e:
                self._send_error_json(500, f"Pipeline execution error: {str(e)}")
                return

        self._send_error_json(404, f"POST endpoint '{path}' not found.")


class EnterpriseAPIServer:
    """Wrapper managing the ThreadingHTTPServer lifecycle."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, handler_cls=EnterpriseRequestHandler):
        self.host = host
        self.port = port
        self.httpd = ThreadingHTTPServer((host, port), handler_cls)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self):
        """Block and serve incoming requests."""
        self.httpd.serve_forever()

    def shutdown(self):
        """Gracefully shut down server."""
        self.httpd.shutdown()
        self.httpd.server_close()


def create_app_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    similarity_threshold: float = 0.65,
) -> EnterpriseAPIServer:
    """
    Factory function initializing the full enterprise stack and attaching it
    to the HTTP request handler.
    """
    # 1. Ingestion Pipeline
    pipeline = IngestionPipeline()
    pipeline.process_directory(settings.raw_docs_dir)

    # 2. Hybrid Retriever (BM25 + Dense Vectors)
    hybrid = HybridRetriever()
    hybrid.index(pipeline.child_store, list(pipeline.parent_store.values()))

    # 3. Security, Caching & Observability
    auth = AuthService()
    cache = SemanticCache(similarity_threshold=similarity_threshold)
    tracer = QueryTracer()

    # 4. Secure RAG Gateway
    gateway = SecureRAGGateway(
        hybrid_retriever=hybrid,
        auth_service=auth,
        cache=cache,
        tracer=tracer,
    )

    # 5. Automated RAG Evaluator
    evaluator = RAGEvaluator(gateway=gateway, auth_service=auth)

    # Bind dependencies to handler class
    EnterpriseRequestHandler.gateway = gateway
    EnterpriseRequestHandler.auth_service = auth
    EnterpriseRequestHandler.cache = cache
    EnterpriseRequestHandler.tracer = tracer
    EnterpriseRequestHandler.evaluator = evaluator
    EnterpriseRequestHandler.pipeline = pipeline

    return EnterpriseAPIServer(host=host, port=port)

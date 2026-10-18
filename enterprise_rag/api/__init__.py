"""Enterprise RAG API and Interactive Web Application.

Provides a multi-threaded REST API server and Single Page Application (SPA)
with authentication, RBAC testing, telemetry observability, and RAG Triad evaluations.
"""

from enterprise_rag.api.server import EnterpriseAPIServer, create_app_server

__all__ = ["EnterpriseAPIServer", "create_app_server"]

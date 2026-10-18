"""Stage 8 Standalone Runner: Interactive Web Application & API Server.

Launches the multi-threaded HTTP server and serves the full Single Page Application
at http://127.0.0.1:8000.
"""

import sys
from enterprise_rag.api.server import create_app_server


def print_banner():
    print("=" * 75)
    print("🚀 STAGE 8: INTERACTIVE WEB APPLICATION & ENTERPRISE REST API")
    print("=" * 75)
    print("\n[*] Initializing Knowledge Base, Vector Index, RBAC, and Telemetry Engine...")


def main():
    print_banner()

    host = "127.0.0.1"
    port = 8000

    try:
        server = create_app_server(host=host, port=port)
    except OSError as e:
        print(f"\n⚠️  Port {port} in use, attempting port 8080...")
        port = 8080
        server = create_app_server(host=host, port=port)

    print("\n" + "=" * 75)
    print(f"🌐 ENTERPRISE RAG WEB APPLICATION IS LIVE!")
    print("=" * 75)
    print(f"\n👉 Open in your web browser:  http://{host}:{port}")
    print("\n👥 PRE-CONFIGURED TEST PERSONAS (Switch live with 1-click in the UI):")
    print("  • Alice Johnson  (@alice_emp)   : HR Employee       [Clearance: INTERNAL]")
    print("  • Bob Martinez   (@bob_finance) : Finance Analyst   [Clearance: CONFIDENTIAL]")
    print("  • David Kim      (@david_dev)   : DevOps Engineer   [Clearance: INTERNAL]")
    print("  • Sarah Connor   (@sarah_ceo)   : Chief Executive   [Clearance: RESTRICTED]")

    print("\n⚡ REST API ENDPOINTS EXPOSED:")
    print(f"  • POST http://{host}:{port}/api/auth/login          (PBKDF2 Auth & JWT Issue)")
    print(f"  • GET  http://{host}:{port}/api/auth/users          (List Personas)")
    print(f"  • POST http://{host}:{port}/api/chat/query          (Secure RAG Query Execution)")
    print(f"  • GET  http://{host}:{port}/api/observability/metrics (Live Latency & Token KPI Telemetry)")
    print(f"  • GET  http://{host}:{port}/api/evaluation/benchmark(RAG Triad Quality Scorecard)")
    print(f"  • GET  http://{host}:{port}/api/corpus/documents    (Catalog of Ingested Documents)")

    print("\n💡 FEATURES TO TEST IN THE UI:")
    print("  1. Cold vs Warm Queries: Ask a question twice to observe >85% latency drop via Semantic Cache!")
    print("  2. RBAC Security: Ask for executive severance as Alice (BLOCKED) vs as Sarah (AUTHORIZED).")
    print("  3. Citation Provenance: Click any verified citation pill to open the raw slideover drawer.")
    print("  4. RAG Triad Tab: Click the 'RAG Triad Benchmark' tab to inspect real-time evaluation scores.")

    print("\n" + "-" * 75)
    print("Press Ctrl+C to stop the server.")
    print("-" * 75 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down Enterprise API Server gracefully...")
        server.shutdown()
        print("[*] Server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()

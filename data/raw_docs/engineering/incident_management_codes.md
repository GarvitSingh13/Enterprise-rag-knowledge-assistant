---
title: "Incident Management & System Error Codes"
department: "engineering"
clearance_level: "internal"
document_type: "guide"
author: "Site Reliability Engineering"
version: "3.1.0"
tags: ["sre", "incident", "error-codes", "troubleshooting"]
---

# Production Error Codes & Incident Severity Matrix

## 1. Incident Severity Definitions and SLAs
- **Severity P1 (Critical Outage):** Core payment processing or user authentication down. SLA Response Time: 15 minutes. Escalation: VP of Engineering and Incident Commander.
- **Severity P2 (Major Degradation):** Non-blocking service failure or customer portal partial impairment. SLA Response Time: 1 hour. Escalation: SRE Lead on-call.
- **Severity P3 (Minor Glitch):** Cosmetic UI issues or non-critical background batch job delays. SLA Response Time: 4 business hours.

## 2. Standard Production Error Code Reference
The following error codes correspond to internal telemetry events:

- `ERR-502-TIMEOUT`: Upstream gateway timed out waiting for backend gRPC response (>30,000ms). Action: Check Redis connection pool saturation on node cluster `SKU-PROD-CLUSTER-A4`.
- `ERR-901-DATABASE-FAILOVER`: Primary PostgreSQL node disconnected. Automatic read-replica promotion initiated. Action: Verify split-brain prevention lock in Consul.
- `ERR-403-TOKEN-EXPIRED`: JSON Web Token signature verification failed or token passed 24-hour expiration threshold. Client must re-authenticate with refresh token.
- `ERR-304-CACHE-STALE`: Local edge cache TTL expired. Automatic cache invalidation sent to Cloudflare edge nodes.

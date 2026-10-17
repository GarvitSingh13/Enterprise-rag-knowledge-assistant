---
title: "Engineering API Deployment SOP"
department: "engineering"
clearance_level: "internal"
document_type: "sop"
author: "DevOps Architecture Team"
version: "2.4.0"
tags: ["deployment", "kubernetes", "api", "sop"]
---

# Standard Operating Procedure: Production API Deployment

## 1. Overview and Pre-Deployment Gates
All microservices deployed to production Kubernetes clusters must pass automated CI/CD pipeline checks before traffic migration. 
Prior to initiating a deployment:
- All unit tests must achieve a minimum of 85% branch coverage.
- Vulnerability scans must report zero critical or high severity CVEs.
- Load testing must confirm P99 latency remains under 200ms at 1,500 RPS.

## 2. Canary Deployment Strategy
Deployments must execute via progressive canary rollout:
1. **Canary Phase 1 (5% traffic):** Traffic routed to canary pods for 15 minutes. Error rates must not exceed 0.05%.
2. **Canary Phase 2 (25% traffic):** Traffic increased to 25% for 30 minutes. Memory utilization must remain below 75%.
3. **Full Promotion (100% traffic):** If all health metrics remain green, the release is promoted to 100% across all production nodes.

## 3. Automated Rollback Triggers
An automated rollback to the previous stable release hash is triggered immediately if:
- HTTP 5xx error rate spikes above 0.5% over a 3-minute rolling window.
- The Kubernetes readiness probe fails for more than 3 consecutive retries on more than 20% of pods.
- Latency P95 exceeds 500ms for more than 5 continuous minutes.

Rollback command:
`kubectl rollout undo deployment/api-gateway -n production`

"""Production Tracing & Observability Profiler for Enterprise RAG.

Instruments query lifecycle with millisecond-precision latency spans,
token consumption estimates, and performance telemetry.
"""

from typing import Dict, List, Optional, Any
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Span(BaseModel):
    """Represents a single execution step span."""
    name: str
    duration_ms: float = 0.0


class TraceRecord(BaseModel):
    """Complete telemetry trace record for a single user query."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    cache_hit: bool = False
    total_latency_ms: float = 0.0
    spans: Dict[str, float] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )


class QueryTracer:
    """Telemetry tracer instrumenting execution spans and recording trace metrics."""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.traces: List[TraceRecord] = []
        self._current_spans: Dict[str, float] = {}
        self._query_start_time: float = 0.0

    def start_trace(self, query: str) -> str:
        """Initialize a new trace lifecycle."""
        self._current_spans = {}
        self._query_start_time = time.perf_counter()
        return str(uuid.uuid4())

    @contextmanager
    def span(self, name: str):
        """Context manager measuring execution latency of a pipeline phase."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration = (time.perf_counter() - t0) * 1000.0  # Convert to ms
            self._current_spans[name] = round(duration, 2)

    def record_trace(
        self,
        query: str,
        cache_hit: bool = False,
        prompt_text: str = "",
        completion_text: str = "",
        trace_id: Optional[str] = None,
    ) -> TraceRecord:
        """Finalize and persist trace record."""
        total_latency = (time.perf_counter() - self._query_start_time) * 1000.0

        # Estimate tokens (rule of thumb: ~4 characters per token)
        prompt_tokens = max(1, len(prompt_text) // 4) if prompt_text else 0
        completion_tokens = max(1, len(completion_text) // 4) if completion_text else 0
        total_tokens = prompt_tokens + completion_tokens

        record = TraceRecord(
            trace_id=trace_id or str(uuid.uuid4()),
            query=query,
            cache_hit=cache_hit,
            total_latency_ms=round(total_latency, 2),
            spans=dict(self._current_spans),
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

        self.traces.append(record)
        if len(self.traces) > self.max_history:
            self.traces = self.traces[-self.max_history :]

        return record

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute aggregate observability metrics across recent query traces."""
        if not self.traces:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "cache_hit_rate": 0.0,
                "total_tokens_consumed": 0,
            }

        total = len(self.traces)
        latencies = [t.total_latency_ms for t in self.traces]
        cache_hits = sum(1 for t in self.traces if t.cache_hit)
        total_tokens = sum(t.token_usage.get("total_tokens", 0) for t in self.traces)

        latencies.sort()
        p95_index = int(total * 0.95)
        p95_latency = latencies[min(p95_index, total - 1)]

        return {
            "total_queries_tracked": total,
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "cache_hit_count": cache_hits,
            "cache_hit_rate_pct": round((cache_hits / total) * 100, 2),
            "total_tokens_consumed": total_tokens,
        }

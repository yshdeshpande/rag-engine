"""System-level metrics: latency, throughput, cost per query.

Provides a lightweight tracker that wraps pipeline operations and records
timing data for each stage (retrieval, reranking, generation). Results
can be aggregated into summary statistics for benchmarking.
"""

import time
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    """Records wall-clock time for a single pipeline stage."""

    name: str
    start: float = 0.0
    end: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        return (self.end - self.start) * 1000

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.end = time.perf_counter()


@dataclass
class QueryMetrics:
    """Timing breakdown for a single query through the pipeline."""

    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    num_chunks_retrieved: int = 0
    num_chunks_after_rerank: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MetricsTracker:
    """Accumulates metrics across multiple queries for summary stats."""

    queries: list[QueryMetrics] = field(default_factory=list)

    def record(self, metrics: QueryMetrics) -> None:
        self.queries.append(metrics)

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    def summarize(self) -> dict[str, float]:
        """Compute aggregate statistics over all recorded queries."""
        if not self.queries:
            return {}

        n = len(self.queries)
        total_latencies = [q.total_ms for q in self.queries]
        retrieval_latencies = [q.retrieval_ms for q in self.queries]
        generation_latencies = [q.generation_ms for q in self.queries]

        total_input = sum(q.input_tokens for q in self.queries)
        total_output = sum(q.output_tokens for q in self.queries)

        return {
            "num_queries": n,
            "avg_total_ms": sum(total_latencies) / n,
            "p50_total_ms": _percentile(total_latencies, 50),
            "p95_total_ms": _percentile(total_latencies, 95),
            "p99_total_ms": _percentile(total_latencies, 99),
            "avg_retrieval_ms": sum(retrieval_latencies) / n,
            "avg_generation_ms": sum(generation_latencies) / n,
            "throughput_qps": n / (sum(total_latencies) / 1000) if sum(total_latencies) > 0 else 0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_input_tokens": total_input / n,
            "avg_output_tokens": total_output / n,
        }


def _percentile(values: list[float], pct: int) -> float:
    """Compute the pct-th percentile of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (pct / 100) * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

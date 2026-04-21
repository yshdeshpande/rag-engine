"""Run the full evaluation suite across configurations.

Loads a benchmark dataset (JSON lines with question/answer/relevant_doc_ids),
runs each query through the pipeline, and computes retrieval + generation +
system metrics. Supports comparing multiple configs (e.g., chunking strategies
or retrieval weights) in a single run.

Benchmark dataset format (data/benchmark/queries.jsonl):
    {"question": "...", "answer": "...", "relevant_doc_ids": ["doc1", "doc2"]}

Usage:
    from src.evaluation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner.from_yaml("configs/default.yaml")
    runner.ingest()
    results = runner.run("data/benchmark/queries.jsonl")
    results.save("experiments/run_001.json")

    # Or compare configs:
    compare_configs(
        config_paths=["configs/fixed.yaml", "configs/recursive.yaml"],
        benchmark_path="data/benchmark/queries.jsonl",
        output_dir="experiments/comparison",
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.evaluation.generation_metrics import compute_programmatic
from src.evaluation.retrieval_metrics import compute_all as compute_retrieval
from src.evaluation.system_metrics import MetricsTracker, QueryMetrics, StageTimer
from src.pipeline import RAGPipeline


@dataclass
class QueryResult:
    """Full evaluation result for a single query."""

    question: str
    ground_truth_answer: str
    generated_answer: str
    relevant_doc_ids: set[str]
    retrieved_doc_ids: list[str]
    retrieval_metrics: dict[str, float]
    generation_metrics: dict[str, float]
    latency: QueryMetrics


@dataclass
class BenchmarkResults:
    """Aggregated results across all queries in a benchmark run."""

    config_name: str
    config: dict
    query_results: list[QueryResult] = field(default_factory=list)
    system_summary: dict[str, float] = field(default_factory=dict)

    @property
    def num_queries(self) -> int:
        return len(self.query_results)

    def aggregate_retrieval(self) -> dict[str, float]:
        """Average retrieval metrics across all queries."""
        if not self.query_results:
            return {}

        all_keys = self.query_results[0].retrieval_metrics.keys()
        return {
            key: sum(qr.retrieval_metrics[key] for qr in self.query_results) / self.num_queries
            for key in all_keys
        }

    def aggregate_generation(self) -> dict[str, float]:
        """Average generation metrics across all queries."""
        if not self.query_results:
            return {}

        # Only average numeric fields
        numeric_keys = [
            k for k in self.query_results[0].generation_metrics
            if isinstance(self.query_results[0].generation_metrics[k], (int, float))
        ]
        return {
            key: sum(qr.generation_metrics[key] for qr in self.query_results) / self.num_queries
            for key in numeric_keys
        }

    def summary(self) -> dict:
        """Full summary: retrieval + generation + system metrics."""
        return {
            "config": self.config_name,
            "num_queries": self.num_queries,
            "retrieval": self.aggregate_retrieval(),
            "generation": self.aggregate_generation(),
            "system": self.system_summary,
        }

    def save(self, path: str) -> None:
        """Save results to a JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.summary(),
            "queries": [
                {
                    "question": qr.question,
                    "ground_truth": qr.ground_truth_answer,
                    "generated": qr.generated_answer,
                    "retrieval_metrics": qr.retrieval_metrics,
                    "generation_metrics": qr.generation_metrics,
                    "latency_ms": qr.latency.total_ms,
                }
                for qr in self.query_results
            ],
        }

        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to {output}")


class BenchmarkRunner:
    """Runs the evaluation suite against a benchmark dataset."""

    def __init__(self, pipeline: RAGPipeline, config: dict):
        self.pipeline = pipeline
        self.config = config
        self.tracker = MetricsTracker()

    @classmethod
    def from_yaml(cls, config_path: str) -> "BenchmarkRunner":
        pipeline = RAGPipeline.from_yaml(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(pipeline=pipeline, config=config)

    def ingest(self, directory: str | None = None) -> None:
        """Load and index documents."""
        self.pipeline.ingest(directory=directory)

    def run(self, benchmark_path: str) -> BenchmarkResults:
        """Run all benchmark queries and collect metrics.

        Args:
            benchmark_path: Path to a JSONL file with benchmark queries.
                Each line: {"question": str, "answer": str, "relevant_doc_ids": list[str]}

        Returns:
            BenchmarkResults with per-query and aggregate metrics.
        """
        queries = _load_benchmark(benchmark_path)
        eval_cfg = self.config.get("evaluation", {})
        k_values = eval_cfg.get("retrieval_k_values", [1, 3, 5, 10, 20])
        config_name = self.config.get("corpus", {}).get("name", "unnamed")

        results = BenchmarkResults(config_name=config_name, config=self.config)
        self.tracker = MetricsTracker()

        for i, query_data in enumerate(queries):
            question = query_data["question"]
            ground_truth = query_data.get("answer", "")
            relevant_ids = set(query_data.get("relevant_doc_ids", []))

            print(f"[{i + 1}/{len(queries)}] {question[:80]}...")

            qm = QueryMetrics()

            # Retrieval
            with StageTimer("retrieval") as t:
                retrieved = self.pipeline.retrieve(question)
            qm.retrieval_ms = t.elapsed_ms
            qm.num_chunks_retrieved = len(retrieved)

            # Reranking (already happens inside pipeline.retrieve)
            reranker_cfg = self.config.get("reranker", {})
            qm.num_chunks_after_rerank = min(
                reranker_cfg.get("top_k", len(retrieved)),
                len(retrieved),
            )

            # Generation
            with StageTimer("generation") as t:
                gen_result = self.pipeline.generator.generate(question, retrieved)
            qm.generation_ms = t.elapsed_ms
            qm.input_tokens = gen_result.usage.get("input_tokens", 0)
            qm.output_tokens = gen_result.usage.get("output_tokens", 0)
            qm.total_ms = qm.retrieval_ms + qm.generation_ms

            # Retrieval metrics
            retrieved_doc_ids = [chunk.doc_id for chunk, _ in retrieved]
            retrieval_scores = compute_retrieval(retrieved_doc_ids, relevant_ids, k_values)

            # Generation metrics (programmatic only — LLM-judge is opt-in)
            gen_scores = compute_programmatic(
                answer=gen_result.answer,
                context=gen_result.context_passages,
                num_passages=len(retrieved),
            )

            self.tracker.record(qm)

            results.query_results.append(QueryResult(
                question=question,
                ground_truth_answer=ground_truth,
                generated_answer=gen_result.answer,
                relevant_doc_ids=relevant_ids,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieval_metrics=retrieval_scores,
                generation_metrics=gen_scores,
                latency=qm,
            ))

        results.system_summary = self.tracker.summarize()
        return results

    def run_with_llm_judge(
        self,
        benchmark_path: str,
        judge_model: str | None = None,
        api_key: str | None = None,
    ) -> BenchmarkResults:
        """Run benchmark with LLM-judge generation metrics (slower, costs money).

        Runs the standard benchmark first, then adds faithfulness + relevance
        scores via the LLM judge for each query.
        """
        from src.evaluation.generation_metrics import compute_llm_judge

        results = self.run(benchmark_path)

        eval_cfg = self.config.get("evaluation", {})
        model = judge_model or eval_cfg.get("llm_judge_model", "claude-sonnet-4-20250514")

        print(f"\nRunning LLM-judge evaluation with {model}...")
        for i, qr in enumerate(results.query_results):
            print(f"  [{i + 1}/{len(results.query_results)}] Judging...")

            judge_scores = compute_llm_judge(
                question=qr.question,
                answer=qr.generated_answer,
                context="",  # context is in the generated answer's passages
                model=model,
                api_key=api_key,
            )

            qr.generation_metrics.update(judge_scores)

        return results


def _load_benchmark(path: str) -> list[dict]:
    """Load benchmark queries from a JSONL file."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def compare_configs(
    config_paths: list[str],
    benchmark_path: str,
    output_dir: str = "experiments",
    data_dir: str | None = None,
) -> list[BenchmarkResults]:
    """Run the same benchmark across multiple configs and save results.

    Useful for ablation studies — comparing chunking strategies, retrieval
    weights, embedding models, etc.

    Args:
        config_paths: List of YAML config files to compare.
        benchmark_path: Path to the benchmark JSONL file.
        output_dir: Directory to save per-config result files.
        data_dir: Override data directory for ingestion.

    Returns:
        List of BenchmarkResults, one per config.
    """
    all_results = []

    for config_path in config_paths:
        config_name = Path(config_path).stem
        print(f"\n{'=' * 60}")
        print(f"Running benchmark: {config_name}")
        print(f"{'=' * 60}")

        runner = BenchmarkRunner.from_yaml(config_path)
        runner.ingest(directory=data_dir)
        results = runner.run(benchmark_path)
        results.save(f"{output_dir}/{config_name}.json")
        all_results.append(results)

        # Print summary
        summary = results.summary()
        print(f"\n--- {config_name} ---")
        for key, val in summary["retrieval"].items():
            print(f"  {key}: {val:.4f}")

    # Print comparison table
    if len(all_results) > 1:
        _print_comparison(all_results)

    return all_results


def _print_comparison(results_list: list[BenchmarkResults]) -> None:
    """Print a side-by-side comparison of key metrics."""
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    # Collect key metrics to compare
    key_metrics = ["mrr", "ndcg@5", "recall@5", "precision@5"]
    header = f"{'metric':<20}" + "".join(f"{r.config_name:<15}" for r in results_list)
    print(header)
    print("-" * len(header))

    for metric in key_metrics:
        row = f"{metric:<20}"
        for r in results_list:
            agg = r.aggregate_retrieval()
            val = agg.get(metric, 0.0)
            row += f"{val:<15.4f}"
        print(row)

    # System metrics
    print()
    system_metrics = [("Avg latency (ms)", "avg_total_ms"), ("Throughput (qps)", "throughput_qps")]
    for label, key in system_metrics:
        row = f"{label:<20}"
        for r in results_list:
            val = r.system_summary.get(key, 0.0)
            row += f"{val:<15.2f}"
        print(row)

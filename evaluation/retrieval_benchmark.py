from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch

from data import load_nli_dataset
from evaluation.metrics import mean_reciprocal_rank, recall_at_k
from models import BertNLIClassifier, build_tokenizer
from pipeline import RetrievalReasoningPipeline
from retrieval import FaissSentenceRetriever
from utils import ensure_dir, write_json


@dataclass
class RetrievalBenchmarkResult:
    backend: str
    reranking: str
    top_k: int
    candidate_k: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_latency_ms: float
    status: str = "completed"
    notes: str = ""


def benchmark_pipeline(
    pipeline: RetrievalReasoningPipeline,
    queries: list[str],
    relevant_ids: list[set[str]],
    top_k: int,
    candidate_k: int | None,
    rerank: bool,
) -> RetrievalBenchmarkResult:
    ranked_ids: list[list[str]] = []
    latencies_ms: list[float] = []

    for query in queries:
        started_at = time.perf_counter()
        results = pipeline.retrieve_and_rerank(query, top_k=top_k, candidate_k=candidate_k, rerank=rerank)
        latencies_ms.append((time.perf_counter() - started_at) * 1000)
        ranked_ids.append([result.doc_id for result in results])

    effective_candidate_k = candidate_k or max(
        top_k,
        top_k * pipeline.candidate_multiplier if rerank and pipeline.reranking_enabled else top_k,
    )
    return RetrievalBenchmarkResult(
        backend=pipeline.retriever.backend,
        reranking="on" if rerank and pipeline.reranking_enabled else "off",
        top_k=top_k,
        candidate_k=effective_candidate_k,
        recall_at_1=recall_at_k(ranked_ids, relevant_ids, k=1),
        recall_at_3=recall_at_k(ranked_ids, relevant_ids, k=min(3, top_k)),
        recall_at_5=recall_at_k(ranked_ids, relevant_ids, k=min(5, top_k)),
        mrr=mean_reciprocal_rank(ranked_ids, relevant_ids),
        avg_latency_ms=sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        notes="Queries use the paired hypothesis to retrieve the matching premise.",
    )


def append_csv(path: str | Path, rows: list[RetrievalBenchmarkResult]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = list(asdict(rows[0]).keys())
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense retrieval and reranking on an NLI corpus.")
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--entailment-weight", type=float, default=0.65)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", default="results/retrieval_metrics.json")
    parser.add_argument("--output-csv", default="results/retrieval_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_nli_dataset(args.corpus_path)
    documents = [example.premise for example in examples]
    doc_ids = [str(index) for index in range(len(examples))]
    queries = [example.hypothesis for example in examples]
    relevant_ids = [{doc_id} for doc_id in doc_ids]

    retriever = FaissSentenceRetriever(
        model_name=args.embedding_model,
        local_files_only=args.local_files_only,
    ).fit(
        documents,
        doc_ids=doc_ids,
        metadata=[{"label": example.label} for example in examples],
    )

    classifier = None
    tokenizer = None
    if args.checkpoint_dir:
        classifier = BertNLIClassifier.load_from_checkpoint(args.checkpoint_dir).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        tokenizer = build_tokenizer(args.checkpoint_dir, local_files_only=True)

    pipeline = RetrievalReasoningPipeline(
        retriever=retriever,
        classifier=classifier,
        tokenizer=tokenizer,
        max_length=args.max_length,
        entailment_weight=args.entailment_weight,
    )

    rows = [
        benchmark_pipeline(
            pipeline=pipeline,
            queries=queries,
            relevant_ids=relevant_ids,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerank=False,
        )
    ]
    if pipeline.reranking_enabled:
        rows.append(
            benchmark_pipeline(
                pipeline=pipeline,
                queries=queries,
                relevant_ids=relevant_ids,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                rerank=True,
            )
        )

    payload = {"corpus_path": args.corpus_path, "rows": [asdict(row) for row in rows]}
    write_json(args.output_json, payload)
    append_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()

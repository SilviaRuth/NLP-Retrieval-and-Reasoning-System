from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from data import load_nli_dataset
from models import BertNLIClassifier, build_tokenizer
from pipeline import RetrievalReasoningPipeline
from retrieval import FaissSentenceRetriever
from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval + reasoning inference.")
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--entailment-weight", type=float, default=0.65)
    parser.add_argument("--disable-reranking", action="store_true")
    parser.add_argument("--output-path")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_nli_dataset(args.corpus_path)
    documents = [example.premise for example in examples]
    doc_ids = [str(index) for index in range(len(examples))]
    metadata = [{"hypothesis": example.hypothesis, "label": example.label} for example in examples]

    retriever = FaissSentenceRetriever(
        model_name=args.embedding_model,
        local_files_only=args.local_files_only,
    ).fit(documents, doc_ids=doc_ids, metadata=metadata)

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
    started_at = time.perf_counter()
    results = pipeline.retrieve_and_rerank(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank=not args.disable_reranking,
    )
    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "backend": retriever.backend,
        "reranking_enabled": any(result.reranking_enabled for result in results),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "results": RetrievalReasoningPipeline.serialize(results),
    }
    if args.output_path:
        output_path = Path(args.output_path)
        ensure_dir(output_path.parent)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

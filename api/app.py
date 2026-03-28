from __future__ import annotations

import os
import time

import torch
from fastapi import FastAPI

from data import load_nli_dataset
from models import BertNLIClassifier, build_tokenizer
from pipeline import RetrievalReasoningPipeline
from retrieval import FaissSentenceRetriever


app = FastAPI(title="NLP Retrieval Reasoning API")
pipeline: RetrievalReasoningPipeline | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pipeline() -> RetrievalReasoningPipeline:
    corpus_path = os.getenv("NLI_CORPUS_PATH")
    if not corpus_path:
        raise RuntimeError("Set NLI_CORPUS_PATH before starting the API")

    examples = load_nli_dataset(corpus_path)
    # We index premises as the retrievable evidence text and keep hypothesis/label
    # in metadata so the API still preserves the original NLI context.
    retriever = FaissSentenceRetriever(
        model_name=os.getenv("NLI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        local_files_only=os.getenv("NLI_LOCAL_FILES_ONLY", "false").lower() == "true",
    ).fit(
        [example.premise for example in examples],
        doc_ids=[str(index) for index in range(len(examples))],
        metadata=[{"hypothesis": example.hypothesis, "label": example.label} for example in examples],
    )

    checkpoint_dir = os.getenv("NLI_CHECKPOINT_DIR")
    if checkpoint_dir:
        classifier = BertNLIClassifier.load_from_checkpoint(checkpoint_dir).to(device)
        tokenizer = build_tokenizer(checkpoint_dir, local_files_only=True)
    else:
        classifier = None
        tokenizer = None

    return RetrievalReasoningPipeline(
        retriever=retriever,
        classifier=classifier,
        tokenizer=tokenizer,
        entailment_weight=float(os.getenv("NLI_ENTAILMENT_WEIGHT", "0.65")),
    )


@app.on_event("startup")
def startup_event() -> None:
    global pipeline
    # Build shared retrieval/model state once at startup instead of per request.
    pipeline = build_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(query: str, top_k: int = 5, candidate_k: int | None = None, rerank: bool = True):
    if pipeline is None:
        return {"error": "pipeline not initialized"}
    started_at = time.perf_counter()
    results = pipeline.retrieve_and_rerank(query, top_k=top_k, candidate_k=candidate_k, rerank=rerank)
    return {
        "query": query,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "reranking_enabled": any(result.reranking_enabled for result in results),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "results": RetrievalReasoningPipeline.serialize(results),
    }

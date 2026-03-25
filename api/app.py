from __future__ import annotations

import os

from fastapi import FastAPI

from data import load_nli_dataset
from models import BertNLIClassifier, build_tokenizer
from pipeline import RetrievalReasoningPipeline
from retrieval import FaissSentenceRetriever


app = FastAPI(title="NLP Retrieval Reasoning API")
pipeline: RetrievalReasoningPipeline | None = None


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
        classifier = BertNLIClassifier.load_from_checkpoint(checkpoint_dir)
        tokenizer = build_tokenizer(checkpoint_dir, local_files_only=True)
    else:
        classifier = None
        tokenizer = None

    return RetrievalReasoningPipeline(retriever=retriever, classifier=classifier, tokenizer=tokenizer)


@app.on_event("startup")
def startup_event() -> None:
    global pipeline
    # Build shared retrieval/model state once at startup instead of per request.
    pipeline = build_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(query: str, top_k: int = 5):
    if pipeline is None:
        return {"error": "pipeline not initialized"}
    return {"results": RetrievalReasoningPipeline.serialize(pipeline.retrieve_and_rerank(query, top_k=top_k))}

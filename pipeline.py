from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from retrieval import RetrievedDocument


@dataclass
class RankedResult:
    doc_id: str
    text: str
    retrieval_score: float
    entailment_score: float | None
    normalized_retrieval_score: float
    final_score: float
    reranking_enabled: bool
    score_breakdown: dict[str, float]
    metadata: dict[str, Any]


class RetrievalReasoningPipeline:
    def __init__(
        self,
        retriever,
        classifier=None,
        tokenizer=None,
        max_length: int = 256,
        entailment_weight: float = 0.65,
        candidate_multiplier: int = 3,
    ) -> None:
        self.retriever = retriever
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entailment_weight = entailment_weight
        self.candidate_multiplier = candidate_multiplier

    @property
    def reranking_enabled(self) -> bool:
        return self.classifier is not None and self.tokenizer is not None

    def _entailment_index(self) -> int | None:
        if self.classifier is None:
            return None
        for index, label in self.classifier.id_to_label.items():
            if "entail" in label.lower():
                return index
        return None

    @staticmethod
    def _normalize_scores(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)
        if abs(maximum - minimum) < 1e-12:
            return [1.0 for _ in scores]
        return [(score - minimum) / (maximum - minimum) for score in scores]

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        rerank: bool = True,
    ) -> list[RankedResult]:
        use_reranker = rerank and self.reranking_enabled
        candidate_k = candidate_k or max(top_k, top_k * self.candidate_multiplier if use_reranker else top_k)
        documents: list[RetrievedDocument] = self.retriever.retrieve(query, top_k=candidate_k)
        normalized_scores = self._normalize_scores([doc.score for doc in documents])

        if not use_reranker:
            return [
                RankedResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    retrieval_score=doc.score,
                    entailment_score=None,
                    normalized_retrieval_score=normalized_score,
                    final_score=normalized_score,
                    reranking_enabled=False,
                    score_breakdown={"retrieval": normalized_score, "reranker": 0.0},
                    metadata=doc.metadata,
                )
                for doc, normalized_score in zip(documents[:top_k], normalized_scores[:top_k])
            ]

        entailment_index = self._entailment_index()
        probabilities = self.classifier.predict_proba(
            tokenizer=self.tokenizer,
            premises=[doc.text for doc in documents],
            hypotheses=[query] * len(documents),
            max_length=self.max_length,
        )

        results = []
        for doc, probs, normalized_score in zip(documents, probabilities, normalized_scores):
            entailment_score = probs[entailment_index] if entailment_index is not None else max(probs)
            final_score = (1.0 - self.entailment_weight) * normalized_score + self.entailment_weight * entailment_score
            results.append(
                RankedResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    retrieval_score=doc.score,
                    entailment_score=entailment_score,
                    normalized_retrieval_score=normalized_score,
                    final_score=final_score,
                    reranking_enabled=True,
                    score_breakdown={
                        "retrieval": (1.0 - self.entailment_weight) * normalized_score,
                        "reranker": self.entailment_weight * entailment_score,
                    },
                    metadata=doc.metadata,
                )
            )

        results.sort(key=lambda item: item.final_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def serialize(results: list[RankedResult]) -> list[dict[str, Any]]:
        return [asdict(result) for result in results]

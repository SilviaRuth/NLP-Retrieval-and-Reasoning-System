from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from retrieval import RetrievedDocument


@dataclass
class RankedResult:
    doc_id: str
    text: str
    retrieval_score: float
    entailment_score: float | None
    final_score: float
    metadata: dict[str, Any]


class RetrievalReasoningPipeline:
    def __init__(
        self,
        retriever,
        classifier=None,
        tokenizer=None,
        max_length: int = 256,
        entailment_weight: float = 0.65,
    ) -> None:
        self.retriever = retriever
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entailment_weight = entailment_weight

    def _entailment_index(self) -> int | None:
        if self.classifier is None:
            return None
        for index, label in self.classifier.id_to_label.items():
            if "entail" in label.lower():
                return index
        return None

    def retrieve_and_rerank(self, query: str, top_k: int = 5) -> list[RankedResult]:
        documents: list[RetrievedDocument] = self.retriever.retrieve(query, top_k=top_k)
        if self.classifier is None or self.tokenizer is None:
            return [
                RankedResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    retrieval_score=doc.score,
                    entailment_score=None,
                    final_score=doc.score,
                    metadata=doc.metadata,
                )
                for doc in documents
            ]

        entailment_index = self._entailment_index()
        probabilities = self.classifier.predict_proba(
            tokenizer=self.tokenizer,
            premises=[doc.text for doc in documents],
            hypotheses=[query] * len(documents),
            max_length=self.max_length,
        )

        results = []
        for doc, probs in zip(documents, probabilities):
            entailment_score = probs[entailment_index] if entailment_index is not None else max(probs)
            final_score = (1.0 - self.entailment_weight) * doc.score + self.entailment_weight * entailment_score
            results.append(
                RankedResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    retrieval_score=doc.score,
                    entailment_score=entailment_score,
                    final_score=final_score,
                    metadata=doc.metadata,
                )
            )

        results.sort(key=lambda item: item.final_score, reverse=True)
        return results

    @staticmethod
    def serialize(results: list[RankedResult]) -> list[dict[str, Any]]:
        return [asdict(result) for result in results]

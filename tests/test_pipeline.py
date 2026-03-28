import unittest

from pipeline import RetrievalReasoningPipeline
from retrieval.faiss_retriever import RetrievedDocument


class DummyRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        del query, top_k
        return [
            RetrievedDocument(doc_id="1", text="weak lexical match", score=0.95, metadata={"label": "neutral"}),
            RetrievedDocument(doc_id="2", text="strong entailment evidence", score=0.65, metadata={"label": "entailment"}),
        ]


class DummyClassifier:
    id_to_label = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def predict_proba(self, tokenizer, premises, hypotheses, max_length=256, batch_size=16):
        del tokenizer, hypotheses, max_length, batch_size
        outputs = []
        for premise in premises:
            if "strong entailment" in premise:
                outputs.append([0.05, 0.10, 0.85])
            else:
                outputs.append([0.10, 0.70, 0.20])
        return outputs


class PipelineTests(unittest.TestCase):
    def test_reranker_can_change_order(self) -> None:
        pipeline = RetrievalReasoningPipeline(
            retriever=DummyRetriever(),
            classifier=DummyClassifier(),
            tokenizer=object(),
            entailment_weight=0.8,
        )

        results = pipeline.retrieve_and_rerank("evidence query", top_k=2, candidate_k=2, rerank=True)

        self.assertEqual(results[0].doc_id, "2")
        self.assertTrue(results[0].reranking_enabled)
        self.assertAlmostEqual(results[0].score_breakdown["reranker"], 0.68)

    def test_dense_only_mode_exposes_normalized_scores(self) -> None:
        pipeline = RetrievalReasoningPipeline(retriever=DummyRetriever())

        results = pipeline.retrieve_and_rerank("evidence query", top_k=2, candidate_k=2, rerank=False)

        self.assertFalse(results[0].reranking_enabled)
        self.assertEqual(results[0].score_breakdown["reranker"], 0.0)
        self.assertGreaterEqual(results[0].final_score, results[1].final_score)


if __name__ == "__main__":
    unittest.main()

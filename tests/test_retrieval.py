import unittest
from unittest.mock import patch

from retrieval import FaissSentenceRetriever


class RetrievalTests(unittest.TestCase):
    @patch("retrieval.faiss_retriever.SentenceTransformer", None)
    def test_retriever_falls_back_to_tfidf(self) -> None:
        retriever = FaissSentenceRetriever(use_faiss=False)
        retriever.fit(
            [
                "A soccer game with multiple men playing on a field.",
                "A chef slices vegetables in the kitchen.",
                "Two dogs jump for a frisbee.",
            ],
            doc_ids=["soccer", "kitchen", "dogs"],
        )

        results = retriever.retrieve("Men are playing soccer outdoors.", top_k=2)

        self.assertEqual(retriever.backend, "tfidf")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "soccer")


if __name__ == "__main__":
    unittest.main()

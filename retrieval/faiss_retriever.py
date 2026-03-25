from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
from typing import Any, Sequence

import numpy as np

from utils.io import ensure_dir

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FaissSentenceRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_faiss: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.local_files_only = local_files_only
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.index = None
        self.vectorizer = None
        self.backend = "uninitialized"
        self.model = None

    def _load_dense_model(self):
        if SentenceTransformer is None:
            return None
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        return self.model

    def fit(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> "FaissSentenceRetriever":
        self.documents = list(documents)
        self.doc_ids = list(doc_ids) if doc_ids is not None else [str(index) for index in range(len(documents))]
        self.metadata = list(metadata) if metadata is not None else [{} for _ in self.documents]

        dense_model = self._load_dense_model()
        if dense_model is not None:
            embeddings = dense_model.encode(self.documents, convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
            if self.use_faiss and faiss is not None:
                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.index.add(self.embeddings.astype(np.float32))
                self.backend = "faiss"
            else:
                self.backend = "numpy"
            return self

        if TfidfVectorizer is None:
            raise ImportError("Either sentence-transformers or scikit-learn must be installed for retrieval")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.embeddings = self.vectorizer.fit_transform(self.documents)
        self.backend = "tfidf"
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if not self.documents:
            raise ValueError("Retriever has not been fit on any documents")

        if self.backend == "faiss":
            query_vector = self._load_dense_model().encode([query], convert_to_numpy=True, show_progress_bar=False)
            query_vector = query_vector / np.clip(np.linalg.norm(query_vector, axis=1, keepdims=True), 1e-12, None)
            scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
            ranked_indices = indices[0].tolist()
            ranked_scores = scores[0].tolist()
        elif self.backend == "numpy":
            query_vector = self._load_dense_model().encode([query], convert_to_numpy=True, show_progress_bar=False)
            query_vector = query_vector / np.clip(np.linalg.norm(query_vector, axis=1, keepdims=True), 1e-12, None)
            similarities = np.matmul(self.embeddings, query_vector[0])
            ranked_indices = np.argsort(similarities)[::-1][:top_k].tolist()
            ranked_scores = similarities[ranked_indices].tolist()
        elif self.backend == "tfidf":
            query_vector = self.vectorizer.transform([query])
            similarities = (self.embeddings @ query_vector.T).toarray().ravel()
            ranked_indices = np.argsort(similarities)[::-1][:top_k].tolist()
            ranked_scores = similarities[ranked_indices].tolist()
        else:
            raise ValueError("Retriever backend is not initialized")

        return [
            RetrievedDocument(
                doc_id=self.doc_ids[index],
                text=self.documents[index],
                score=float(ranked_scores[offset]),
                metadata=self.metadata[index],
            )
            for offset, index in enumerate(ranked_indices)
        ]

    def save(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        config = {
            "model_name": self.model_name,
            "use_faiss": self.use_faiss,
            "local_files_only": self.local_files_only,
            "backend": self.backend,
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "metadata": self.metadata,
        }
        with Path(output_dir, "retriever_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

        if self.backend in {"faiss", "numpy"} and self.embeddings is not None:
            np.save(Path(output_dir, "embeddings.npy"), self.embeddings)
        if self.backend == "tfidf" and self.vectorizer is not None:
            with Path(output_dir, "tfidf.pkl").open("wb") as handle:
                pickle.dump(self.vectorizer, handle)

    @classmethod
    def load(cls, output_dir: str | Path) -> "FaissSentenceRetriever":
        output_dir = Path(output_dir)
        with Path(output_dir, "retriever_config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        retriever = cls(
            model_name=config["model_name"],
            use_faiss=config["use_faiss"],
            local_files_only=config["local_files_only"],
        )
        retriever.documents = config["documents"]
        retriever.doc_ids = config["doc_ids"]
        retriever.metadata = config["metadata"]
        retriever.backend = config["backend"]

        if retriever.backend in {"faiss", "numpy"}:
            retriever.embeddings = np.load(Path(output_dir, "embeddings.npy"))
            if retriever.backend == "faiss" and faiss is not None:
                retriever.index = faiss.IndexFlatIP(retriever.embeddings.shape[1])
                retriever.index.add(retriever.embeddings.astype(np.float32))
        elif retriever.backend == "tfidf":
            with Path(output_dir, "tfidf.pkl").open("rb") as handle:
                retriever.vectorizer = pickle.load(handle)
            retriever.embeddings = retriever.vectorizer.transform(retriever.documents)

        return retriever

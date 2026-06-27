import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import CONFIG, resolve_path

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self):
        cfg = CONFIG["rag"]
        self.top_k = cfg["top_k"]
        self.index_path = resolve_path(cfg["faiss_index_path"])
        self.kb_path = resolve_path(cfg["knowledge_base_path"])
        self.model = SentenceTransformer(cfg["embedding_model"])
        self.documents: List[Dict] = []
        self.index: faiss.IndexFlatIP | None = None
        self._load_or_build()

    def _load_or_build(self):
        index_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "documents.pkl"

        if index_file.exists() and docs_file.exists():
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(str(index_file))
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)
        else:
            logger.info("Building FAISS index from knowledge base")
            self._build_index()

    def _build_index(self):
        docs = []
        for json_file in self.kb_path.glob("*.json"):
            with open(json_file) as f:
                docs.extend(json.load(f))

        if not docs:
            logger.warning("No documents found in knowledge base")
            return

        self.documents = docs
        texts = [f"{d['title']}\n{d['content']}" for d in docs]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Built FAISS index with {len(docs)} documents")

    def retrieve(self, query: str, category: str | None = None) -> str:
        if self.index is None or not self.documents:
            return ""

        q_embedding = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_embedding, min(self.top_k * 2, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[idx]
            if category and doc.get("category") != category:
                continue
            results.append((score, doc))
            if len(results) >= self.top_k:
                break

        if not results:
            # Fall back to no category filter
            for score, idx in zip(scores[0], indices[0]):
                results.append((score, self.documents[idx]))
                if len(results) >= self.top_k:
                    break

        context_parts = []
        for score, doc in results:
            context_parts.append(f"[Source: {doc['title']}]\n{doc['content']}")

        return "\n\n---\n\n".join(context_parts)


_kb_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance

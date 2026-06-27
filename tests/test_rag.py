import pytest
from unittest.mock import patch, MagicMock


def test_knowledge_base_retrieve_returns_string():
    from src.rag.knowledge_base import KnowledgeBase
    with patch.object(KnowledgeBase, '_load_or_build'):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb.top_k = 3
        kb.documents = []
        kb.index = None
        kb.model = MagicMock()

    result = kb.retrieve("test query")
    assert isinstance(result, str)


def test_knowledge_base_documents_loaded():
    import json
    from pathlib import Path
    kb_path = Path("src/data/knowledge_base")
    all_docs = []
    for f in kb_path.glob("*.json"):
        with open(f) as fp:
            all_docs.extend(json.load(fp))
    assert len(all_docs) >= 15, "Expected at least 15 knowledge base articles"
    for doc in all_docs:
        assert "title" in doc
        assert "content" in doc
        assert "category" in doc

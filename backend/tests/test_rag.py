from pathlib import Path
from types import SimpleNamespace
import os

import pytest

from config import config
from agents.rag_agent.rag_agent import RAGAgent

DATA_DOC = Path(__file__).parent / "data" / "rag" / "text_medical_doc.txt"


@pytest.fixture(scope="function")
def rag_agent_with_kb(monkeypatch):
    """RAGAgent with real vector store and the test medical document ingested."""
    print("[test_rag] setup rag_agent_with_kb")
    
    # Ensure tests run in local mode (no remote QDRANT_HOST) to exercise data/* storage
    if hasattr(config, 'QDRANT_HOST') and config.QDRANT_HOST:
        monkeypatch.setattr(config, 'QDRANT_HOST', None)
    
    agent = RAGAgent()

    # Stub LLM to avoid external chat calls while keeping retrieval real
    agent.llm = SimpleNamespace(
        invoke=lambda prompt: SimpleNamespace(content="stubbed answer from LLM")
    )

    ingest_result = agent.ingest_document(str(DATA_DOC))
    assert ingest_result.get("success"), f"Ingest failed: {ingest_result}"
    
    # Verify local data storage happened (if in local mode)
    if not config.QDRANT_HOST:
        doc_name = DATA_DOC.name
        docs_db_path = Path(config.DOCS_DB_PATH) / doc_name
        parsed_docs_path = Path(config.PARSED_DOCS_PATH) / f"{DATA_DOC.stem}.json"
        
        # Check that files were created in data/* directories
        assert docs_db_path.exists(), f"Expected document copy at {docs_db_path}"
        assert parsed_docs_path.exists(), f"Expected parsed JSON at {parsed_docs_path}"
        print(f"[test_rag] verified local storage: {docs_db_path.name}, {parsed_docs_path.name}")
    
    return agent


def test_rag_workflow_hybrid(rag_agent_with_kb):
    """Exercise query expansion + hybrid retrieval + (optional) reranking."""
    print("[test_rag] start hybrid")
    agent = rag_agent_with_kb

    result = agent.process_query(
        "what is hypertension",
        use_hybrid=True,
        use_query_expansion=True,
        use_reranking=True,
    )

    assert result["status"] == "success"
    assert result["agent"] == "RAG_AGENT"
    assert result["retrieval_method"] == "hybrid"
    # Query expansion should be on in this path
    assert result["query_expanded"] is True
    # Reranking may fail gracefully depending on environment; just assert flag present
    assert isinstance(result["reranked"], bool)
    assert isinstance(result["confidence"], (float, int))
    assert isinstance(result["sources"], list)


def test_rag_workflow_dense(rag_agent_with_kb):
    """Exercise dense-only path without query expansion and reranking."""
    print("[test_rag] start dense")
    agent = rag_agent_with_kb

    result = agent.process_query(
        "what is hypertension",
        use_hybrid=False,
        use_query_expansion=False,
        use_reranking=False,
    )

    assert result["status"] == "success"
    assert result["retrieval_method"] == "dense"
    assert result["query_expanded"] is False
    assert result["reranked"] is False
    assert isinstance(result["confidence"], (float, int))
    assert isinstance(result["sources"], list)

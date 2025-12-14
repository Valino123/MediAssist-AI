from pathlib import Path
from types import SimpleNamespace
import os
import numpy as np

import pytest

from config import config
from agents.rag_agent.rag_agent import RAGAgent
from agents.rag_agent.vector_store import get_global_vector_store

DATA_DOC = Path(__file__).parent / "data" / "rag" / "text_medical_doc.txt"


def fake_encode(texts):
    """Mock embedding model encode method to return fake embeddings for testing."""
    # Return 1536-dimensional embeddings to match Azure text-embedding-3-small
    if isinstance(texts, str):
        texts = [texts]
    # Generate deterministic fake embeddings based on text hash
    embeddings = []
    for text in texts:
        # Create a simple deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.rand(1536).astype(np.float32).tolist()
        embeddings.append(embedding)
    # Always return a list of embeddings (list of lists)
    return embeddings


@pytest.fixture(scope="function")
def rag_agent_with_kb(monkeypatch):
    """RAGAgent with real vector store and the test medical document ingested."""
    print("[test_rag] setup rag_agent_with_kb")
    
    # Ensure tests run in local mode (no remote QDRANT_HOST) to exercise data/* storage
    if hasattr(config, 'QDRANT_HOST') and config.QDRANT_HOST:
        monkeypatch.setattr(config, 'QDRANT_HOST', None)
    
    # Mock the AzureTextEmbeddingModel.encode method to avoid API calls in CI
    from agents.rag_agent import vector_store as vs_module
    
    def mock_encode(self, texts):
        return fake_encode(texts)
    
    monkeypatch.setattr(vs_module.AzureTextEmbeddingModel, "encode", mock_encode)
    
    # Properly close and reset the global vector store
    # Close the module-level cache and its client first
    if vs_module.vector_store is not None:
        try:
            if hasattr(vs_module.vector_store, 'client') and vs_module.vector_store.client is not None:
                try:
                    vs_module.vector_store.client.close()
                except AttributeError:
                    # QdrantClient might not have close() in some versions
                    pass
        except Exception:
            pass
    vs_module.vector_store = None
    
    # Reset singleton state
    vs_module.VectorStore._instance = None
    vs_module.VectorStore._initialized = False
    
    # Delete existing collection to ensure clean state with correct dimensions
    try:
        from qdrant_client import QdrantClient
        temp_client = QdrantClient(path=config.VECTOR_DB_PATH)
        collections = temp_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if "medical_documents" in collection_names:
            temp_client.delete_collection("medical_documents")
            print("[test_rag] deleted existing collection for clean test state")
        temp_client.close()
    except Exception as e:
        print(f"[test_rag] warning: could not delete collection: {e}")
    
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

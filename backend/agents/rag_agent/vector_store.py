import os
import json 
import logging
import threading
from typing import List, Dict, Union, Any, Optional 
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid 

from config import config, logger

# Import BM25Retriever with error handling
try:
    from .bm25_retriever import BM25Retriever
except ImportError as e:
    logger.warning(f"BM25Retriever not available: {e}")
    BM25Retriever = None

class AzureTextEmbeddingModel:
    def __init__(self):
        """Initialize Azure OpenAI text embedding model"""
        try:
            self.client = AzureOpenAI(
                api_key=config.EMBEDDING_API_KEY,
                api_version=config.EMBEDDING_API_VERSION,
                azure_endpoint=config.EMBEDDING_AZURE_ENDPOINT
            )
            self.deployment = config.EMBEDDING_DEPLOYMENT
            self.model = config.EMBEDDING_MODEL
            logger.info(f"Azure OpenAI embedding model initialized: {self.deployment}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI embedding model: {str(e)}")
            raise
    
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for input texts"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # Validate input
            if not texts or not any(text.strip() for text in texts):
                logger.warning("Empty or whitespace-only texts provided for embedding")
                return []
            
            response = self.client.embeddings.create(
                model=self.deployment,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []

    

class VectorStore:
    _instance = None
    _initialized = False
    _lock = threading.Lock() # Add a class-level lock
    
    def __new__(cls):
        """Singleton pattern to prevent multiple instances"""
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if self._initialized:
            return
            
        try:
            # Initialize Qdrant client with connection management
            self.client = QdrantClient(path=config.VECTOR_DB_PATH)
            self.collection_name = "medical_documents"
            
            # Initialize embedding model
            # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_model = AzureTextEmbeddingModel()

            # Initialize BM25 retriever if available
            if BM25Retriever is not None:
                self.bm25_retriever = BM25Retriever()
                self.document_texts = []  # Store original texts for BM25
            else:
                self.bm25_retriever = None
                self.document_texts = []
                logger.warning("BM25Retriever not available, hybrid search will use dense search only")

            # Create collection if it doesn't exist
            self._create_collection()
            
            self._initialized = True
            logger.info("VectorStore initialized (singleton) with hybrid retrieval support")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}")
            # Clean up on failure
            if hasattr(self, 'client') and self.client:
                self.client = None
            raise
    
    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name, 
                    vectors_config=VectorParams(
                        # size=384, # all-MiniLM-L6-v2 embedding size
                        size = 1536,# text-embedding-3-small embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts)
            # return embeddings.tolist()
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store"""
        try:
            points = []

            for doc in documents:
                embedding = self.generate_embeddings([doc["text"]])[0]

                point = PointStruct(
                    id = str(uuid.uuid4()),
                    vector = embedding,
                    payload = {
                        "text": doc["text"],
                        "doc_id": doc["doc_id"],
                        "chunk_index": doc.get("chunk_index", 0),
                        "metadata": doc.get("metadata", {})
                    }                
                )
                points.append(point)
            
            # Insert points into collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            # Also add to BM25 retriever if available
            if self.bm25_retriever is not None:
                self.bm25_retriever.add_documents(documents)
            
            # Store texts for hybrid retrieval
            for doc in documents:
                self.document_texts.append(doc["text"])

            logger.info(f"Added {len(points)} documents to vector store and BM25 index")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def search_similar(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents using dense embeddings only"""
        if limit is None:
            limit = config.MAX_RETRIEVAL_DOCS 
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]

            # Search in collection
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
            )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["text"],
                    "doc_id": result.payload["doc_id"],
                    "chunk_index": result.payload["chunk_index"],
                    "score": result.score,
                    "metadata": result.payload["metadata"],
                    "retrieval_method": "dense"
                })
                logger.info(f"Document {result.payload['doc_id']} found with score {result.score}")
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

    def hybrid_search(self, query: str, limit: int = None, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse retrieval"""
        if limit is None:
            limit = config.MAX_RETRIEVAL_DOCS
            
        try:
            # Dense retrieval
            dense_results = self.search_similar(query, limit * 2)  # Get more for reranking
            
            # BM25 retrieval
            bm25_results = []
            if self.bm25_retriever is not None:
                bm25_results = self.bm25_retriever.search(query, limit * 2)
            
            # Combine and rerank results
            combined_results = self._combine_results(dense_results, bm25_results, alpha)
            
            logger.info(f"Hybrid search returned {len(combined_results[:limit])} results (dense: {len(dense_results)}, bm25: {len(bm25_results)})")
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to dense search only
            return self.search_similar(query, limit)
    
    def _combine_results(self, dense_results: List[Dict], bm25_results: List[Dict], alpha: float) -> List[Dict]:
        """Combine dense and BM25 results using weighted scoring"""
        try:
            # Normalize scores to [0, 1]
            if dense_results:
                max_dense = max(r["score"] for r in dense_results)
                for r in dense_results:
                    r["normalized_dense_score"] = r["score"] / max_dense if max_dense > 0 else 0
            
            if bm25_results:
                max_bm25 = max(r["score"] for r in bm25_results)
                for r in bm25_results:
                    r["normalized_bm25_score"] = r["score"] / max_bm25 if max_bm25 > 0 else 0
            
            # Create combined scoring
            combined = {}
            
            # Add dense results
            for r in dense_results:
                key = r["text"][:100]  # Use text as key for deduplication
                combined[key] = {
                    "text": r["text"],
                    "doc_id": r.get("doc_id"),
                    "chunk_index": r.get("chunk_index"),
                    "metadata": r.get("metadata", {}),
                    "dense_score": r.get("normalized_dense_score", 0),
                    "bm25_score": 0,
                    "combined_score": r.get("normalized_dense_score", 0) * alpha,
                    "retrieval_method": "hybrid"
                }
            
            # Add/update with BM25 results
            for r in bm25_results:
                key = r["text"][:100]
                if key in combined:
                    combined[key]["bm25_score"] = r.get("normalized_bm25_score", 0)
                    combined[key]["combined_score"] += r.get("normalized_bm25_score", 0) * (1 - alpha)
                else:
                    combined[key] = {
                        "text": r["text"],
                        "doc_id": r.get("doc_id"),
                        "chunk_index": r.get("chunk_index"),
                        "metadata": r.get("metadata", {}),
                        "dense_score": 0,
                        "bm25_score": r.get("normalized_bm25_score", 0),
                        "combined_score": r.get("normalized_bm25_score", 0) * (1 - alpha),
                        "retrieval_method": "hybrid"
                    }
            
            # Sort by combined score
            sorted_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            # Return dense results as fallback
            return dense_results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "name": collection_info.config.name,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._create_collection()
            
            # Also clear BM25 index if available
            if self.bm25_retriever is not None:
                self.bm25_retriever.clear()
            self.document_texts = []
            
            logger.info("Collection and BM25 index cleared and recreated")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

    def close(self):
        """Close the database connection and reset singleton"""
        try:
            if hasattr(self, 'client') and self.client:
                # QdrantClient doesn't have an explicit close method, but we can set it to None
                # The connection will be cleaned up when the object is garbage collected
                self.client = None
                logger.info("Vector store connection closed")
            
            # Reset singleton state
            VectorStore._instance = None
            VectorStore._initialized = False
            
        except Exception as e:
            logger.error(f"Error closing vector store: {str(e)}")
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance (Thread-Safe)"""
        # First check without a lock for performance
        if cls._instance is None:
            # If no instance, acquire the lock
            with cls._lock:
                # Double-check if another thread created an instance
                # while this thread was waiting for the lock.
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing or restart)"""
        if cls._instance:
            cls._instance.close()
        cls._instance = None
        cls._initialized = False

# Global vector store handle (lazy-initialized via init_global_vector_store in app lifespan)
vector_store: Optional[VectorStore] = None

def init_global_vector_store() -> bool:
    """Initialize the module-global vector_store if not already set.
    Returns True if initialized in this call, False if it already existed.
    """
    global vector_store
    if vector_store is None:
        vector_store = VectorStore.get_instance()
        return True
    return False

def close_global_vector_store() -> bool:
    """Close and clear the module-global vector_store if present.
    Returns True if closed, False if it was already None.
    """
    global vector_store
    if vector_store is not None:
        vector_store.close()
        vector_store = None
        return True
    return False

def get_global_vector_store() -> VectorStore:
    """Return the global vector store, initializing if necessary."""
    init_global_vector_store()
    # mypy: vector_store can be None only before init; after init it's set
    return vector_store  # type: ignore[return-value]
            
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from config import config, logger

# Import with error handling
try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    logger.warning(f"sentence-transformers not available: {e}")
    CrossEncoder = None

class CrossEncoderReranker:
    """Cross-encoder reranker for improving document relevance scoring"""
    
    def __init__(self, model_name: str = None):
        """Initialize the cross-encoder reranker"""
        try:
            if CrossEncoder is None:
                logger.warning("CrossEncoder not available, reranking will be disabled")
                self.model = None
                self.model_available = False
                return
            
            # Use configurable model name or default
            if model_name is None:
                model_name = getattr(config, 'CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            self.model_name = model_name
            self.model = CrossEncoder(model_name)
            self.model_available = True
            
            logger.info(f"CrossEncoderReranker initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoderReranker: {str(e)}")
            self.model = None
            self.model_available = False
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder"""
        try:
            if not self.model_available or not self.model:
                logger.warning("Cross-encoder not available, returning original documents")
                return documents
            
            if not documents or not query:
                logger.warning("Empty query or documents provided for reranking")
                return documents
            
            # Use configurable top_k or default
            if top_k is None:
                top_k = getattr(config, 'RERANK_TOP_K', 20)
            
            # Limit documents to rerank for efficiency
            docs_to_rerank = documents[:top_k]
            
            # Create query-document pairs
            query_doc_pairs = []
            for doc in docs_to_rerank:
                # Extract text content for reranking
                doc_text = doc.get("text", "")
                if doc_text:
                    query_doc_pairs.append([query, doc_text])
                else:
                    logger.warning(f"Document {doc.get('doc_id', 'unknown')} has no text content")
            
            if not query_doc_pairs:
                logger.warning("No valid query-document pairs for reranking")
                return documents
            
            # Get relevance scores from cross-encoder
            logger.info(f"Reranking {len(query_doc_pairs)} documents with cross-encoder")
            scores = self.model.predict(query_doc_pairs)
            
            # Add cross-encoder scores to documents
            for i, doc in enumerate(docs_to_rerank):
                if i < len(scores):
                    doc["cross_encoder_score"] = float(scores[i])
                else:
                    doc["cross_encoder_score"] = 0.0
            
            # Sort documents by cross-encoder score (descending)
            reranked_docs = sorted(docs_to_rerank, key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
            
            # Add remaining documents that weren't reranked
            if len(documents) > top_k:
                remaining_docs = documents[top_k:]
                for doc in remaining_docs:
                    doc["cross_encoder_score"] = 0.0  # No reranking score
                reranked_docs.extend(remaining_docs)
            
            logger.info(f"Cross-encoder reranking completed. Top score: {reranked_docs[0].get('cross_encoder_score', 0):.3f}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return documents  # Return original documents as fallback
    
    def rerank_batch(self, queries_docs: List[tuple]) -> List[List[Dict[str, Any]]]:
        """Rerank multiple query-document pairs in batch"""
        try:
            if not self.model_available or not self.model:
                logger.warning("Cross-encoder not available, returning original documents")
                return [docs for _, docs in queries_docs]
            
            results = []
            for query, documents in queries_docs:
                reranked = self.rerank(query, documents)
                results.append(reranked)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch cross-encoder reranking: {str(e)}")
            return [docs for _, docs in queries_docs]  # Return original documents as fallback
    
    def get_reranking_stats(self) -> Dict[str, Any]:
        """Get statistics about the reranker"""
        return {
            "model_available": self.model_available,
            "model_name": getattr(self, 'model_name', None),
            "enabled": getattr(config, 'ENABLE_CROSS_ENCODER_RERANKING', True)
        }
    
    def clear_cache(self):
        """Clear any cached data (placeholder for future caching implementation)"""
        # Future implementation for caching
        pass

# Global cross-encoder reranker instance
cross_encoder_reranker = CrossEncoderReranker()

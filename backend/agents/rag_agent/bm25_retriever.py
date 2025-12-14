import numpy as np
from typing import List, Dict, Any, Optional
import re
import logging

# Import with error handling
try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"Warning: rank_bm25 not available: {e}")
    BM25Okapi = None

try:
    import jieba  # For Chinese text segmentation
except ImportError as e:
    print(f"Warning: jieba not available: {e}")
    jieba = None

try:
    from config import config, logger
except ImportError:
    # Fallback for standalone usage
    import logging
    logger = logging.getLogger(__name__)
    class Config:
        ENABLE_HYBRID_RETRIEVAL = True
        HYBRID_ALPHA = 0.7
    config = Config()

class BM25Retriever:
    """BM25 sparse retrieval implementation for hybrid search"""
    
    def __init__(self, documents: List[str] = None):
        """Initialize BM25 with preprocessed documents"""
        self.documents = documents or []
        self.processed_docs = []
        self.bm25 = None
        self.document_metadata = []  # Store metadata for each document
        
        if self.documents:
            self._initialize_bm25()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 (tokenization, cleaning)"""
        try:
            # Remove special characters and normalize
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            
            # For Chinese text, use jieba segmentation if available
            # For English, simple word splitting
            if jieba and any('\u4e00' <= char <= '\u9fff' for char in text):
                tokens = list(jieba.cut(text))
            else:
                tokens = text.split()
            
            # Filter out very short tokens and common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            filtered_tokens = [token for token in tokens if len(token) > 1 and token not in stop_words]
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return []
    
    def _initialize_bm25(self):
        """Initialize BM25 with current documents"""
        try:
            if BM25Okapi is None:
                logger.warning("BM25Okapi not available, skipping BM25 initialization")
                self.bm25 = None
                return
                
            self.processed_docs = [self._preprocess_text(doc) for doc in self.documents]
            # Filter out empty documents
            valid_docs = [doc for doc in self.processed_docs if doc]
            if valid_docs:
                self.bm25 = BM25Okapi(valid_docs)
                logger.info(f"BM25 initialized with {len(valid_docs)} documents")
            else:
                logger.warning("No valid documents found for BM25 initialization")
                self.bm25 = None
        except Exception as e:
            logger.error(f"Error initializing BM25: {str(e)}")
            self.bm25 = None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to BM25 index"""
        try:
            new_texts = []
            new_metadata = []
            
            for doc in documents:
                text = doc.get("text", "")
                if text.strip():
                    new_texts.append(text)
                    new_metadata.append({
                        "doc_id": doc.get("doc_id", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "metadata": doc.get("metadata", {})
                    })
            
            # Add to existing documents
            self.documents.extend(new_texts)
            self.document_metadata.extend(new_metadata)
            
            # Reinitialize BM25 with all documents
            self._initialize_bm25()
            
            logger.info(f"Added {len(new_texts)} documents to BM25 index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to BM25: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25"""
        try:
            if BM25Okapi is None:
                logger.warning("BM25Okapi not available, returning empty results")
                return []
                
            if not self.bm25 or not self.documents:
                logger.warning("BM25 not initialized or no documents available")
                return []
            
            processed_query = self._preprocess_text(query)
            if not processed_query:
                logger.warning("Query preprocessing resulted in empty tokens")
                return []
            
            scores = self.bm25.get_scores(processed_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    if idx < len(self.document_metadata):
                        metadata = self.document_metadata[idx]
                    else:
                        metadata = {"doc_id": f"bm25_{idx}", "chunk_index": 0, "metadata": {}}
                    
                    results.append({
                        "text": self.documents[idx],
                        "score": float(scores[idx]),
                        "index": int(idx),
                        "doc_id": metadata.get("doc_id", f"bm25_{idx}"),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "metadata": metadata.get("metadata", {}),
                        "retrieval_method": "bm25"
                    })
            
            logger.info(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index"""
        return {
            "total_documents": len(self.documents),
            "processed_documents": len(self.processed_docs),
            "bm25_initialized": self.bm25 is not None,
            "average_doc_length": np.mean([len(doc) for doc in self.processed_docs]) if self.processed_docs else 0
        }
    
    def clear(self):
        """Clear all documents from BM25 index"""
        self.documents = []
        self.processed_docs = []
        self.document_metadata = []
        self.bm25 = None
        logger.info("BM25 index cleared")

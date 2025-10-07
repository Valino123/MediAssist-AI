# Hybrid Retrieval Implementation Guide

## Overview

This implementation adds BM25 sparse retrieval to your existing dense vector retrieval system, creating a hybrid approach that combines the best of both methods.

## What's New

### 1. BM25Retriever Class (`agents/rag_agent/bm25_retriever.py`)
- Implements BM25 sparse retrieval algorithm
- Supports both English and Chinese text processing
- Handles document indexing and searching
- Provides statistics and management functions

### 2. Enhanced VectorStore (`agents/rag_agent/vector_store.py`)
- Added `hybrid_search()` method that combines dense and BM25 results
- Automatic BM25 index management
- Configurable weighting between dense and sparse retrieval
- Fallback to dense-only search if BM25 fails

### 3. Updated RAGAgent (`agents/rag_agent/rag_agent.py`)
- Uses hybrid search by default (configurable)
- Enhanced context formatting with retrieval method information
- Improved score handling for both retrieval types

### 4. Configuration Options (`config.py`)
- `ENABLE_HYBRID_RETRIEVAL`: Enable/disable hybrid search (default: True)
- `HYBRID_ALPHA`: Weight for dense vs BM25 (default: 0.7 = 70% dense, 30% BM25)

## Installation

1. Install the new dependencies:
```bash
pip install -r requirements.txt
```

Key new dependencies:
- `rank-bm25>=0.2.2` - BM25 implementation
- `jieba>=0.42.1` - Chinese text segmentation

## Configuration

Add these settings to your `.env` file:

```env
# Hybrid Retrieval Settings
ENABLE_HYBRID_RETRIEVAL=True
HYBRID_ALPHA=0.7  # 0.7 = 70% dense, 30% BM25
```

### Alpha Parameter Explanation:
- `alpha = 0.0`: Pure BM25 (sparse only)
- `alpha = 0.5`: Equal weight (50% dense, 50% BM25)
- `alpha = 1.0`: Pure dense embeddings
- `alpha = 0.7`: Recommended for medical applications (70% dense, 30% BM25)

## Usage

### Automatic Usage
The RAGAgent now uses hybrid retrieval by default. No code changes needed for existing functionality.

### Manual Control
```python
from agents.rag_agent.rag_agent import rag_agent

# Use hybrid search (default)
result = rag_agent.process_query("What is hypertension?", use_hybrid=True)

# Use dense search only
result = rag_agent.process_query("What is hypertension?", use_hybrid=False)

# Use configuration setting
result = rag_agent.process_query("What is hypertension?")  # Uses config.ENABLE_HYBRID_RETRIEVAL
```

### Direct VectorStore Usage
```python
from agents.rag_agent.vector_store import get_global_vector_store

vs = get_global_vector_store()

# Hybrid search
results = vs.hybrid_search("medical query", limit=5, alpha=0.7)

# Dense search only
results = vs.search_similar("medical query", limit=5)

# BM25 search only
results = vs.bm25_retriever.search("medical query", k=5)
```

## Benefits

### For Medical Applications:
1. **Exact Term Matching**: BM25 excels at finding documents with precise medical terminology
2. **Semantic Understanding**: Dense embeddings capture related concepts and synonyms
3. **Improved Recall**: Hybrid approach finds more relevant documents
4. **Better Precision**: Combined scoring reduces false positives

### Performance:
- **Computational Efficiency**: BM25 is faster for exact matches
- **Scalability**: Both methods scale well with document count
- **Fallback Safety**: System falls back to dense search if BM25 fails

## Testing

Run the test script to verify the implementation:

```bash
python test_hybrid_retrieval.py
```

This will test:
- BM25 document indexing and searching
- Configuration settings
- Basic functionality

## Monitoring

The system logs retrieval method information:
- `"Retrieved X relevant documents using hybrid search (alpha=0.7)"`
- `"Retrieved X relevant documents using dense search"`
- Document scores include both dense and BM25 components

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure to install all dependencies from `requirements.txt`
2. **Chinese Text Issues**: Ensure `jieba` is installed for Chinese text processing
3. **Memory Issues**: BM25 index is stored in memory; monitor for large document collections
4. **Performance**: Adjust `HYBRID_ALPHA` based on your specific use case

### Fallback Behavior:
- If BM25 fails, the system automatically falls back to dense search
- If hybrid search fails, individual dense and BM25 searches are attempted
- All errors are logged for debugging

## Future Enhancements

Potential improvements:
1. **Persistent BM25 Index**: Store BM25 index on disk for faster startup
2. **Dynamic Alpha**: Adjust alpha based on query type or domain
3. **Query Expansion**: Enhance queries with synonyms before BM25 search
4. **Advanced Scoring**: Implement more sophisticated score combination methods
5. **Multi-language Support**: Add support for more languages beyond English/Chinese

## API Changes

### New Response Fields:
- `retrieval_method`: "hybrid", "dense", or "bm25"
- `combined_score`: For hybrid results
- `dense_score` and `bm25_score`: Individual component scores

### Backward Compatibility:
- All existing API endpoints work unchanged
- Existing dense search functionality is preserved
- Configuration defaults maintain current behavior

import logging
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import config, logger
from .document_processor import document_processor
from .vector_store import get_global_vector_store
from .query_expander import query_expander
from .cross_encoder_reranker import cross_encoder_reranker 

class RAGAgent:
    """Retrieval-Augmented Generation agent"""

    def __init__(self):
        """Initialize the RAG agent"""
        self.llm = ChatOpenAI(
            model=config.QWEN_MODEL,
            temperature=float(config.TEMPERATURE or "0.7"),
            max_tokens=int(config.QWEN_MAX_TOKENS or "1000"),
            openai_api_key=config.DASHSCOPE_API_KEY,
            base_url=config.BASE_URL,
        )

        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical assistant with access to a knowledge base.
                Use the provided context to answer questions accurately.
                If the context doesn't contain enough information, say so clearly.
                Always remind users to consult healthcare professionals for serious medical concerns."""),
            ("human", """Context: {context}

    Question: {question}

    Please provide a helpful and accurate response based on the context above.""")

        ])

        logger.info("RAGAgent initialized")

    def retrieve_relevant_documents(self, query: str, use_hybrid: bool = None, use_query_expansion: bool = None, use_reranking: bool = None) -> List[Dict[str, Any]]:
        try:
            """Retrieve relevant documents for a query using hybrid search with optional query expansion and cross-encoder reranking"""
            vs = get_global_vector_store()
            
            # Use config settings if not specified
            if use_hybrid is None:
                use_hybrid = config.ENABLE_HYBRID_RETRIEVAL
            if use_query_expansion is None:
                use_query_expansion = config.ENABLE_QUERY_EXPANSION
            if use_reranking is None:
                use_reranking = config.ENABLE_CROSS_ENCODER_RERANKING
            
            # Step 1: Query Expansion (if enabled)
            search_query = query
            if use_query_expansion:
                try:
                    expanded_query = query_expander.expand_query(query)
                    if expanded_query and expanded_query != query:
                        search_query = expanded_query
                        logger.info(f"Query expanded: '{query}' → '{expanded_query}'")
                    else:
                        logger.info(f"Query expansion returned original query: '{query}'")
                except Exception as e:
                    logger.warning(f"Query expansion failed, using original query: {str(e)}")
                    search_query = query
            
            # Step 2: Initial Document Retrieval (get more documents for reranking)
            retrieval_limit = config.RERANK_TOP_K if use_reranking else config.MAX_RETRIEVAL_DOCS
            
            if use_hybrid:
                # Use hybrid search (dense + BM25)
                results = vs.hybrid_search(search_query, retrieval_limit, alpha=config.HYBRID_ALPHA)
                # For hybrid search, use combined_score for filtering
                filtered_results = [
                    result for result in results if result.get("combined_score", result.get("score", 0)) > config.RAG_CONFIDENCE_THRESHOLD
                ]
                retrieval_method = "hybrid"
                logger.info(f"Retrieved {len(filtered_results)} relevant documents using hybrid search (alpha={config.HYBRID_ALPHA})")
            else:
                # Use dense search only (fallback)
                results = vs.search_similar(search_query, retrieval_limit)
                filtered_results = [
                    result for result in results if result["score"] > config.RAG_CONFIDENCE_THRESHOLD
                ]
                retrieval_method = "dense"
                logger.info(f"Retrieved {len(filtered_results)} relevant documents using dense search")

            # Add metadata about query expansion
            for result in filtered_results:
                result["original_query"] = query
                result["search_query"] = search_query
                result["query_expanded"] = use_query_expansion and search_query != query
                result["retrieval_method"] = retrieval_method

            # Step 3: Cross-Encoder Reranking (if enabled)
            if use_reranking and filtered_results:
                try:
                    logger.info(f"Applying cross-encoder reranking to {len(filtered_results)} documents")
                    reranked_results = cross_encoder_reranker.rerank(query, filtered_results, top_k=config.RERANK_TOP_K)
                    
                    # Limit to final number of documents
                    final_results = reranked_results[:config.MAX_RETRIEVAL_DOCS]
                    
                    # Add reranking metadata
                    for result in final_results:
                        result["reranked"] = True
                        result["reranking_method"] = "cross_encoder"
                    
                    logger.info(f"Cross-encoder reranking completed. Final results: {len(final_results)}")
                    return final_results
                    
                except Exception as e:
                    logger.warning(f"Cross-encoder reranking failed, using original results: {str(e)}")
                    # Add reranking failure metadata
                    for result in filtered_results[:config.MAX_RETRIEVAL_DOCS]:
                        result["reranked"] = False
                        result["reranking_method"] = "none"
                    return filtered_results[:config.MAX_RETRIEVAL_DOCS]
            else:
                # No reranking, just limit results
                final_results = filtered_results[:config.MAX_RETRIEVAL_DOCS]
                for result in final_results:
                    result["reranked"] = False
                    result["reranking_method"] = "none"
                return final_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context"""
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        # Add query expansion info if available
        if documents and documents[0].get("query_expanded"):
            original_query = documents[0].get("original_query", "")
            search_query = documents[0].get("search_query", "")
            if original_query != search_query:
                context_parts.append(f"Query expanded from '{original_query}' to '{search_query}'\\n")
        
        # Add reranking info if available
        if documents and documents[0].get("reranked"):
            reranking_method = documents[0].get("reranking_method", "unknown")
            context_parts.append(f"Documents reranked using {reranking_method}\\n")
        
        for i, doc in enumerate(documents, 1):
            # Handle different score formats
            score = doc.get("cross_encoder_score", doc.get("combined_score", doc.get("score", 0)))
            retrieval_method = doc.get("retrieval_method", "unknown")
            reranked = doc.get("reranked", False)
            
            score_info = f"Score: {score:.3f}"
            if reranked and "cross_encoder_score" in doc:
                score_info += f" (Reranked: {doc['cross_encoder_score']:.3f})"
            
            context_parts.append(f"Document {i} ({score_info}, Method: {retrieval_method}):\\n{doc['text']}\\n")
        
        return "\\n".join(context_parts)

    def generate_response(self, query: str, context: str) -> str:
        try:
            # Create the prompt
            prompt = self.rag_prompt.format(context=context, question=query)

            # Get response from LLM
            response = self.llm.invoke(prompt)

            if hasattr(response, "content"):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error generating a response. Please try again."

    def process_query(self, query: str, use_hybrid: bool = None, use_query_expansion: bool = None, use_reranking: bool = None) -> Dict[str, Any]:
        try:
            documents = self.retrieve_relevant_documents(query, use_hybrid, use_query_expansion, use_reranking)

            context = self.format_context(documents)

            response = self.generate_response(query, context)

            # Handle different score formats (prioritize cross-encoder scores)
            confidence = 0.0
            if documents:
                scores = []
                for doc in documents:
                    if "cross_encoder_score" in doc:
                        scores.append(doc["cross_encoder_score"])
                    elif "combined_score" in doc:
                        scores.append(doc["combined_score"])
                    else:
                        scores.append(doc.get("score", 0))
                confidence = max(scores) if scores else 0.0

            # Determine processing metadata
            query_expanded = False
            reranked = False
            if documents:
                query_expanded = documents[0].get("query_expanded", False)
                reranked = documents[0].get("reranked", False)
            else:
                # Even if no documents are ultimately returned, we still want the
                # response metadata to reflect which retrieval features were
                # active for this query (useful for tests and monitoring).
                if use_query_expansion is None:
                    query_expanded = bool(config.ENABLE_QUERY_EXPANSION)
                else:
                    query_expanded = bool(use_query_expansion)

            return {
                "status": "success",
                "response": response,
                "agent": "RAG_AGENT",
                "confidence": confidence,
                "sources": [doc["doc_id"] for doc in documents],
                "retrieval_method": "hybrid" if (use_hybrid if use_hybrid is not None else config.ENABLE_HYBRID_RETRIEVAL) else "dense",
                "query_expanded": query_expanded,
                "reranked": reranked,
                "reranking_method": documents[0].get("reranking_method", "none") if documents else "none",
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}")
            return {
                "status": "error",
                "response": "I'm sorry, I encountered an error processing your query. Please try again.",
                "agent": "RAG_AGENT",
                "confidence": 0.0,
                "sources": [],
                "retrieval_method": "hybrid" if (use_hybrid if use_hybrid is not None else config.ENABLE_HYBRID_RETRIEVAL) else "dense",
                "query_expanded": False,
                "reranked": False,
                "reranking_method": "none",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a document into the knowledge base"""
        try:
            # Process the document
            result = document_processor.process_document(file_path)

            if not result["success"]:
                return result
            
            # In local vector DB mode, persist raw and parsed docs under data/*
            if not config.QDRANT_HOST:
                try:
                    os.makedirs(config.DOCS_DB_PATH, exist_ok=True)
                    os.makedirs(config.PARSED_DOCS_PATH, exist_ok=True)

                    # Copy raw document into docs DB path for traceability
                    if os.path.isfile(file_path):
                        raw_dest = os.path.join(
                            config.DOCS_DB_PATH, os.path.basename(file_path)
                        )
                        if not os.path.exists(raw_dest):
                            try:
                                # Use binary copy preserving metadata when possible
                                import shutil  # local import to avoid unused at module level

                                shutil.copy2(file_path, raw_dest)
                            except Exception as copy_err:
                                logger.warning(
                                    f"Failed to copy raw document to docs DB path: {copy_err}"
                                )

                    # Persist parsed chunks and metadata as JSON for offline inspection
                    parsed_payload = {
                        "doc_id": result["metadata"]["doc_id"],
                        "source_path": os.path.abspath(file_path),
                        "chunks": result["chunks"],
                        "metadata": result["metadata"],
                    }
                    base_name, _ = os.path.splitext(os.path.basename(file_path))
                    parsed_dest = os.path.join(config.PARSED_DOCS_PATH, base_name + ".json")
                    with open(parsed_dest, "w", encoding="utf-8") as f:
                        json.dump(parsed_payload, f, ensure_ascii=False, indent=2)
                except Exception as persist_err:
                    # Do not fail ingestion if local persistence fails; just log.
                    logger.warning(
                        f"Failed to persist local RAG document artifacts: {persist_err}"
                    )

            # Prepare documents for vector store
            documents = []
            for i, chunk in enumerate(result["chunks"]):
                documents.append({
                    "text": chunk,
                    "doc_id": result["metadata"]["doc_id"],
                    "chunk_index": i,
                    "metadata": result["metadata"]
                })
            
            # Add to vector store
            vs = get_global_vector_store()
            success = vs.add_documents(documents)

            if success:
                return {
                    "success": True,
                    "message": f"Document {result['metadata']['doc_id']} ingested successfully",
                    "chunks_added": len(documents),
                    "metadata": result["metadata"]
                }
            else:
                 return {
                    "success": False,
                    "error": "Failed to add documents to vector store"
                }

        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base"""
        try:
            vs = get_global_vector_store()
            collection_info = vs.get_collection_info()
            return {
                "status": "success",
                "collection_info": collection_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

# Global RAG agent instance - initialized lazily to avoid circular imports at module load time
_rag_agent_instance = None

def _get_rag_agent():
    """Get or create the global RAG agent instance."""
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgent()
    return _rag_agent_instance

# Module-level attribute that will be accessed like: from ... import rag_agent; rag_agent.method()
class _RAGAgentProxy:
    """Proxy to delay instantiation of RAG agent until first access."""
    def __getattr__(self, name):
        return getattr(_get_rag_agent(), name)

rag_agent = _RAGAgentProxy()


import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import config, logger
from .document_processor import document_processor
from .vector_store import get_global_vector_store 

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

    def retrieve_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        try:
            """Retrieve relevant documents for a query"""
            vs = get_global_vector_store()
            results = vs.search_similar(query, config.MAX_RETRIEVAL_DOCS)

            filtered_results = [
                result for result in results if result["score"] > config.RAG_CONFIDENCE_THRESHOLD
            ]

            logger.info(f"Retrieved {len(filtered_results)} relevant documents")

            return filtered_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context"""
        if not documents:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i} (Score: {doc['score']:.3f}):\\n{doc['text']}\\n")
        
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

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            documents = self.retrieve_relevant_documents(query)

            context = self.format_context(documents)

            response = self.generate_response(query, context)

            confidence = max([doc["score"] for doc in documents]) if documents else 0.0

            return {
                "status": "success",
                "response": response,
                "agent": "RAG_AGENT",
                "confidence": confidence,
                "sources": [doc["doc_id"] for doc in documents],
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

# Global RAG agent instance
rag_agent = RAGAgent()


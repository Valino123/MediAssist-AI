import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import config, logger

class LLMQueryExpander:
    """LLM-based query expansion for medical information retrieval"""
    
    def __init__(self):
        """Initialize the query expander with the same LLM configuration as other agents"""
        try:
            # Use the same LLM configuration as RAGAgent and other agents
            self.llm = ChatOpenAI(
                model=config.QWEN_MODEL,
                temperature=config.QUERY_EXPANSION_TEMPERATURE,  # Use configurable temperature for expansion
                max_tokens=config.QUERY_EXPANSION_MAX_TOKENS,  # Use configurable max tokens for expansion
                openai_api_key=config.DASHSCOPE_API_KEY,
                base_url=config.BASE_URL,
            )
            
            # Create prompt template for query expansion
            self.expansion_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical information retrieval specialist. Your task is to expand medical queries with relevant synonyms, abbreviations, and related terms to improve search results.

Guidelines:
1. Add medical synonyms and alternative terms
2. Include common abbreviations (e.g., "MI" for "myocardial infarction")
3. Add lay terms alongside medical terms (e.g., "heart attack" for "myocardial infarction")
4. Include related conditions or symptoms
5. Keep the expansion concise and relevant
6. Maintain the original query intent
7. Focus on terms that would appear in medical documents

Examples:
- "hypertension" → "hypertension high blood pressure HTN elevated blood pressure"
- "chest pain" → "chest pain chest discomfort thoracic pain angina pectoris"
- "diabetes" → "diabetes diabetes mellitus DM sugar disease hyperglycemia"

Return only the expanded query terms, separated by spaces. Do not include explanations or additional text."""),
                ("human", "Original query: {query}\n\nExpanded query:")
            ])
            
            self.llm_available = True
            logger.info("LLMQueryExpander initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMQueryExpander: {str(e)}")
            self.llm = None
            self.llm_available = False
    
    def expand_query(self, query: str) -> str:
        """Expand a medical query with relevant terms"""
        try:
            if not self.llm_available or not self.llm:
                logger.warning("LLM not available for query expansion, returning original query")
                return query
            
            if not query or not query.strip():
                logger.warning("Empty query provided for expansion")
                return query
            
            # Create the expansion chain
            expansion_chain = self.expansion_prompt | self.llm
            
            # Get expansion from LLM
            response = expansion_chain.invoke({"query": query.strip()})
            
            if hasattr(response, "content"):
                expanded_query = response.content.strip()
            else:
                expanded_query = str(response).strip()
            
            # Validate and clean the expanded query
            if expanded_query and len(expanded_query) > len(query):
                # Remove any extra formatting or explanations
                expanded_query = self._clean_expanded_query(expanded_query)
                logger.info(f"Query expanded: '{query}' → '{expanded_query}'")
                return expanded_query
            else:
                logger.warning(f"LLM returned invalid expansion, using original query")
                return query
                
        except Exception as e:
            logger.error(f"Error expanding query '{query}': {str(e)}")
            return query
    
    def _clean_expanded_query(self, expanded_query: str) -> str:
        """Clean and validate the expanded query"""
        try:
            # Remove common prefixes that LLM might add
            prefixes_to_remove = [
                "Expanded query:",
                "Here's the expanded query:",
                "The expanded query is:",
                "Expanded terms:",
                "Synonyms:",
                "Related terms:"
            ]
            
            cleaned = expanded_query
            for prefix in prefixes_to_remove:
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            
            # Remove any quotes or special formatting
            cleaned = cleaned.strip('"\'`')
            
            # Ensure it's not empty and has reasonable length
            if cleaned and 5 <= len(cleaned) <= 500:  # Reasonable bounds
                return cleaned
            else:
                logger.warning(f"Cleaned query has invalid length: {len(cleaned)}")
                return expanded_query
                
        except Exception as e:
            logger.error(f"Error cleaning expanded query: {str(e)}")
            return expanded_query
    
    def expand_query_batch(self, queries: List[str]) -> List[str]:
        """Expand multiple queries in batch"""
        try:
            expanded_queries = []
            for query in queries:
                expanded = self.expand_query(query)
                expanded_queries.append(expanded)
            return expanded_queries
        except Exception as e:
            logger.error(f"Error in batch query expansion: {str(e)}")
            return queries  # Return original queries as fallback
    
    def get_expansion_stats(self) -> Dict[str, Any]:
        """Get statistics about the query expander"""
        return {
            "llm_available": self.llm_available,
            "model": config.QWEN_MODEL if self.llm_available else None,
            "temperature": config.QUERY_EXPANSION_TEMPERATURE,
            "max_tokens": config.QUERY_EXPANSION_MAX_TOKENS,
            "enabled": config.ENABLE_QUERY_EXPANSION
        }

# Global query expander instance
query_expander = LLMQueryExpander()

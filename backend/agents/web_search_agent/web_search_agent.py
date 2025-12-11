import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import config, logger
from .google_search import GoogleSearchProcessor
from .pubmed_search import PubMedSearchProcessor

class WebSearchAgent:
    """Agent for web search and real-time medical information"""

    def __init__(self):
        """Initialize the web search agent"""
        self.web_processor = GoogleSearchProcessor()
        self.pubmed_processor = PubMedSearchProcessor()

    def search_medical_info(self, query: str, search_type: str = "general") -> Dict[str, Any]:
        """Search for medical information"""
        try:
            logger.info(f"WebSearchAgent: Starting {search_type} search for query: '{query}'")
            
            if search_type == "literature":
                # Search PubMed for medical literature
                result = self.pubmed_processor.search(query)
            else:
                # Search general web for medical information
                result = self.web_processor.search(query)

            if result['status'] == 'success':
                # Process and format results
                formatted_result = self._format_search_results(result, search_type)
                
                logger.info(f"WebSearchAgent: {search_type} search successful for query: '{query}'")

                return {
                    'agent': 'WEB_SEARCH_PROCESSOR_AGENT',
                    'status': 'success',
                    'query': query,
                    'search_type': search_type,
                    'results': formatted_result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error(f"WebSearchAgent: {search_type} search failed for query: '{query}', error: {result['error']}")
                return {
                    'agent': 'WEB_SEARCH_PROCESSOR_AGENT',
                    'status': 'error',
                    'error': result['error'],
                    'query': query,
                    'search_type': search_type,
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return {
                'agent': 'WEB_SEARCH_PROCESSOR_AGENT',
                'status': 'error',
                'error': str(e),
                'query': query,
                'search_type': search_type,
                'timestamp': datetime.now().isoformat()
            }

    def _format_search_results(self, result: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        """Format search results for presentation"""
        try:
            if search_type == "literature":
                return self._format_pubmed_results(result)
            else:
                return self._format_web_results(result)

        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            return {'error': str(e)}

    def _format_pubmed_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format PubMed search results"""
        articles = result.get('results', [])

        if not articles:
            return {
                'summary': 'No medical literature found for your query.',
                'articles': [],
                'total_found': 0
            }

        # Create summary
        summary = f"Found {len(articles)} medical research articles related to your query."

        # Format articles
        formatted_articles = []
        for article in articles[:5]:  # Limit to 5 articles
            formatted_article = {
                'title': article.get('title', ''),
                'authors': ', '.join(article.get('authors', [])),
                'journal': article.get('journal', ''),
                'year': article.get('year', ''),
                'abstract': article.get('abstract', '')[:300] + '...' if len(article.get('abstract', '')) > 300 else article.get('abstract', ''),
                'url': article.get('url', ''),
                'pmid': article.get('pmid', '')
            }
            formatted_articles.append(formatted_article)

        return {
            'summary': summary,
            'articles': formatted_articles,
            'total_found': result.get('total_found', len(articles))
        }

    def _format_web_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format web search results"""
        results = result.get('results', [])
        answer = result.get('answer', '')

        if not results and not answer:
            return {
                'summary': 'No relevant information found for your query.',
                'sources': [],
                'answer': ''
            }

        # Create summary
        summary = f"Found {len(results)} relevant sources for your query."
        if answer:
            summary += f" Here's what I found: {answer}"

        # Format sources
        formatted_sources = []
        for source in results[:5]:  # Limit to 5 sources
            formatted_source = {
                'title': source.get('title', ''),
                'url': source.get('url', ''),
                'content': source.get('content', '')[:200] + '...' if len(source.get('content', '')) > 200 else source.get('content', ''),
                'domain': source.get('domain', ''),
                'score': source.get('score', 0.0)
            }
            formatted_sources.append(formatted_source)

        return {
            'summary': summary,
            'sources': formatted_sources,
            'answer': answer
        }

    def generate_search_response(self, query: str, search_type: str = "general") -> str:
        """Generate a formatted response from search results"""
        try:
            search_result = self.search_medical_info(query, search_type)

            if search_result['status'] == 'success':
                results = search_result['results']

                response = f"🔍 **Search Results for: {query}**\n\n"
                response += f"{results['summary']}\n\n"

                if search_type == "literature":
                    # Format PubMed results
                    for i, article in enumerate(results['articles'], 1):
                        response += f"**{i}. {article['title']}**\n"
                        response += f"Authors: {article['authors']}\n"
                        response += f"Journal: {article['journal']} ({article['year']})\n"
                        response += f"Abstract: {article['abstract']}\n"
                        response += f"Link: {article['url']}\n\n"
                else:
                    # Format web results
                    if results['answer']:
                        response += f"**Answer:** {results['answer']}\n\n"

                    for i, source in enumerate(results['sources'], 1):
                        response += f"**{i}. {source['title']}**\n"
                        response += f"Source: {source['domain']}\n"
                        response += f"Content: {source['content']}\n"
                        response += f"Link: {source['url']}\n\n"

                response += "⚠️ *Please note: This information is for educational purposes only and should not replace professional medical advice.*"

                return response
            else:
                return f"❌ **Search Error:** {search_result['error']}"

        except Exception as e:
            logger.error(f"Error generating search response: {str(e)}")
            return f"❌ **Search Error:** {str(e)}"
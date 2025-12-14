"""
Web Search Processor Agent Package
Handles real-time web search for medical information
"""

from .web_search_agent import WebSearchAgent
from .google_search import GoogleSearchProcessor
from .pubmed_search import PubMedSearchProcessor

__all__ = [
    'WebSearchAgent',
    'GoogleSearchProcessor',
    'PubMedSearchProcessor'
]
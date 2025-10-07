from typing import Dict, Any, List
from datetime import datetime
from urllib.parse import urlparse

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import config, logger

class GoogleSearchProcessor:
    """Handles web search using the Google Custom Search JSON API"""

    def __init__(self):
        """Initialize the Google Search processor"""
        self.api_key = config.GOOGLE_API_KEY
        self.cse_id = config.GOOGLE_CSE_ID
        self.timeout = config.SEARCH_TIMEOUT

        if not self.api_key or not self.cse_id:
            logger.warning("Google API Key or CSE ID not provided, web search disabled")
        
        # Build the service object for the Custom Search API
        try:
            self.service = build("customsearch", "v1", developerKey=self.api_key)
        except Exception as e:
            self.service = None
            logger.error(f"Failed to build Google Search service: {e}")

    def search(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """Search the web using the configured Custom Search Engine"""
        if not self.service:
            return {'status': 'error', 'error': 'Google Search service not initialized', 'results': []}

        try:
            num_results = max_results or config.SEARCH_RESULTS_LIMIT
            # The API allows a maximum of 10 results per request
            num_results = min(num_results, 10) 

            # Make the API request
            result = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=num_results
            ).execute()

            processed_results, answer = self._process_search_results(result)

            return {
                'status': 'success',
                'query': query,
                'results': processed_results,
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            }
        
        except HttpError as e:
            logger.error(f"Google API HTTP Error: {e.resp.status} {e.content}")
            error_content = e.content.decode('utf-8')
            return {'status': 'error', 'error': f"Google API Error: {error_content}", 'results': []}
        except Exception as e:
            logger.error(f"Error in Google search: {e}")
            return {'status': 'error', 'error': str(e), 'results': []}

    def _process_search_results(self, data: Dict[str, Any]) -> (List[Dict[str, Any]], str):
        """Process search results from the Google CSE API"""
        results = []
        search_items = data.get('items', [])
        
        answer = search_items[0].get('snippet', '') if search_items else ''

        for item in search_items:
            processed_result = {
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'content': item.get('snippet', ''),
                'score': 0.0, # Not provided by the API
                'published_date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', ''),
                'domain': self._extract_domain(item.get('link', ''))
            }
            results.append(processed_result)
        
        return results, answer

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        if not url:
            return ""
        try:
            return urlparse(url).netloc
        except Exception:
            return ""
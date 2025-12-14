import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import config, logger

class PubMedSearchProcessor:
    """Handles PubMed search for medical literature"""

    def __init__(self):
        """Initialize the PubMed search processor"""
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.timeout = config.SEARCH_TIMEOUT

    def search(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """Search PubMed for medical literature"""
        try:
            max_results = max_results or config.SEARCH_RESULTS_LIMIT

            # Search for article IDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance"
            }

            response = requests.get(search_url, params=search_params, timeout=self.timeout)

            if response.status_code == 200:
                search_data = response.json()
                pmids = search_data.get('esearchresult', {}).get('idlist', [])

                if pmids:
                    # Get article details
                    articles = self._get_article_details(pmids)

                    return {
                        'status': 'success',
                        'query': query,
                        'results': articles,
                        'total_found': search_data.get('esearchresult', {}).get('count', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'status': 'success',
                        'query': query,
                        'results': [],
                        'total_found': 0,
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'status': 'error',
                    'error': f"PubMed API error: {response.status_code}",
                    'results': []
                }

        except Exception as e:
            logger.error(f"Error in PubMed search: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'results': []
            }

    def _get_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information for PubMed articles"""
        try:
            if not pmids:
                return []

            # Fetch article details
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract"
            }

            response = requests.get(fetch_url, params=fetch_params, timeout=self.timeout)

            if response.status_code == 200:
                # Parse XML response (simplified)
                articles = self._parse_pubmed_xml(response.text)
                return articles
            else:
                return []

        except Exception as e:
            logger.error(f"Error getting article details: {str(e)}")
            return []

    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        articles = []
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            for article_elem in root.findall('.//PubmedArticle'):
                article = self._extract_article_info(article_elem)
                if article:
                    articles.append(article)
            return articles
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {str(e)}")
            return []

    def _extract_article_info(self, article_elem) -> Optional[Dict[str, Any]]:
        """Extract article information from XML Element"""
        try:
            title = article_elem.findtext('.//ArticleTitle', default="")
            abstract = article_elem.findtext('.//AbstractText', default="")
            
            authors = [
                f"{author.findtext('LastName', '')} {author.findtext('Initials', '')}"
                for author in article_elem.findall('.//Author')
            ]
            
            journal = article_elem.findtext('.//Journal/Title', default="")
            year = article_elem.findtext('.//PubDate/Year', default="")
            pmid = article_elem.findtext('.//PMID', default="")

            if title:
                return {
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'journal': journal,
                    'year': year,
                    'pmid': pmid,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                }
            return None
        except Exception as e:
            logger.error(f"Error extracting article info: {str(e)}")
            return None
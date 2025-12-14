from types import SimpleNamespace

from agents.web_search_agent.web_search_agent import WebSearchAgent


def test_web_search_general(monkeypatch):
    """General web search should return formatted sources."""
    print("[test_web_search] start general")

    # Simple function (no `self`) because we attach it via SimpleNamespace.
    def fake_search(query):
        return {
            "status": "success",
            "results": [
                {
                    "title": "Result A",
                    "url": "http://a",
                    "content": "A",
                    "domain": "a.com",
                    "score": 0.9,
                }
            ],
            "answer": "summary",
        }

    monkeypatch.setattr(WebSearchAgent, "__init__", lambda self: None)
    agent = WebSearchAgent()
    agent.web_processor = SimpleNamespace(search=fake_search)
    agent.pubmed_processor = SimpleNamespace(search=lambda q: {"status": "success", "results": []})

    result = agent.search_medical_info("flu symptoms", search_type="general")

    assert result["status"] == "success"
    assert result["results"]["sources"][0]["title"] == "Result A"
    assert result["results"]["answer"] == "summary"


def test_web_search_literature(monkeypatch):
    """Literature search should format PubMed-style results."""
    print("[test_web_search] start literature")

    # Simple function (no `self`) because we attach it via SimpleNamespace.
    def fake_pubmed(query):
        return {
            "status": "success",
            "results": [
                {
                    "title": "Paper",
                    "authors": ["A"],
                    "journal": "J",
                    "year": "2024",
                    "abstract": "abs",
                    "url": "u",
                }
            ],
            "total_found": 1,
        }

    monkeypatch.setattr(WebSearchAgent, "__init__", lambda self: None)
    agent = WebSearchAgent()
    agent.web_processor = SimpleNamespace(search=lambda q: {"status": "success", "results": []})
    agent.pubmed_processor = SimpleNamespace(search=fake_pubmed)

    result = agent.search_medical_info("diabetes", search_type="literature")

    assert result["status"] == "success"
    assert result["results"]["articles"][0]["title"] == "Paper"

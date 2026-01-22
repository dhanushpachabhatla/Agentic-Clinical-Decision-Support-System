"""
Web Search Agent
----------------
Role:
- Retrieve non-indexed medical information
- Rare disease mentions
- Newly released guidelines not yet in PubMed

Rules:
- Must be used sparingly
- No reasoning
- No hallucination
- No LLM usage
- Returns raw evidence snippets only

Backend:
- Tavily Search API
"""

from typing import List, Dict
import os
from dotenv import load_dotenv
from tavily import TavilyClient


# ======================================================
# Configuration
# ======================================================

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise EnvironmentError("TAVILY_API_KEY not set")

_client = TavilyClient(api_key=TAVILY_API_KEY)


# ======================================================
# Public API
# ======================================================

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Performs controlled web search.

    Input:
        query: medical query string

    Output:
        [
          {
            "title": "...",
            "url": "...",
            "snippet": "..."
          }
        ]
    """

    if not query.strip():
        return []

    response = _client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results
    )

    results: List[Dict] = []

    for r in response.get("results", []):
        results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content")
        })

    return results


# ======================================================
# Manual Test
# ======================================================

if __name__ == "__main__":

    test_query = "new 2025 guideline management myocarditis"

    print("\n[INFO] Running Web Search Agent...\n")

    results = web_search(test_query, max_results=3)

    for r in results:
        print("=" * 60)
        print("Title:", r["title"])
        print("URL:", r["url"])
        print("Snippet:", (r["snippet"] or "")[:300])

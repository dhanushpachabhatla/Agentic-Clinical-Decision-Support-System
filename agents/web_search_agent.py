"""
Web Search Agent
----------------
Role:
- Retrieve non-indexed medical information
- Rare disease mentions
- Newly released guidelines not yet in PubMed

Rules (ENFORCED):
- No reasoning
- No hallucination
- No LLM usage
- Evidence-only output
- Short snippets only

Backend:
- Tavily Search API
"""

from typing import List, Dict, Optional
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

# Trusted medical / scientific domains
TRUSTED_DOMAINS = [
    "who.int",
    "cdc.gov",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "nejm.org",
    "thelancet.com",
    "nature.com",
    "science.org",
    "ema.europa.eu",
    "fda.gov"
]

MAX_SNIPPET_LENGTH = 400


# ======================================================
# Internal Helpers
# ======================================================

def _is_trusted_source(url: Optional[str]) -> bool:
    if not url:
        return False
    return any(domain in url for domain in TRUSTED_DOMAINS)


def _normalize_snippet(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.strip().replace("\n", " ")[:MAX_SNIPPET_LENGTH]


# ======================================================
# Public API
# ======================================================

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Performs controlled medical web search.

    Input:
        query: medical query string

    Output:
        [
          {
            "title": str,
            "url": str,
            "snippet": str,
            "source_type": "web",
            "trusted": bool
          }
        ]
    """

    if not query or not query.strip():
        return []

    try:
        response = _client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )
    except Exception as e:
        # Fail safely — orchestrator can retry or skip
        return [{
            "title": "Web search failed",
            "url": None,
            "snippet": str(e),
            "source_type": "web",
            "trusted": False
        }]

    results: List[Dict] = []
    seen_urls = set()

    for r in response.get("results", []):
        url = r.get("url")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        results.append({
            "title": r.get("title"),
            "url": url,
            "snippet": _normalize_snippet(r.get("content")),
            "source_type": "web",
            "trusted": _is_trusted_source(url)
        })

    return results


# ======================================================
# Manual Test
# ======================================================

if __name__ == "__main__":

    test_query = "new WHO guideline on dengue fever 2024"

    print("\n[INFO] Running Web Search Agent...\n")

    results = web_search(test_query, max_results=5)

    for r in results:
        print("=" * 70)
        print("Title   :", r["title"])
        print("URL     :", r["url"])
        print("Trusted :", r["trusted"])
        print("Snippet :", r["snippet"])

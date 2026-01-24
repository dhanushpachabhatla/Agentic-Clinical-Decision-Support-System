"""
Web Search Agent (LangGraph Node)
---------------------------------
Role:
- Retrieve non-indexed medical information
- Rare disease mentions
- Newly released clinical guidelines
- Uses Tavily Search via LangChain

Rules:
- No reasoning
- No hallucination
- No LLM usage
- Returns raw evidence snippets only

Can run standalone OR as LangGraph node
"""

from typing import Dict, Any, List
import os
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults


# ======================================================
# Configuration
# ======================================================

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise EnvironmentError("TAVILY_API_KEY not set in environment")


# Initialize Tavily tool
tavily_tool = TavilySearchResults(
    max_results=5,
    api_key=TAVILY_API_KEY
)


# ======================================================
# LangGraph Node Function
# ======================================================

def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible node.

    Expects:
        state["query"] -> clinician question

    Produces:
        state["web_search_results"] -> list of search snippets
    """

    query = state.get("query", "").strip()

    if not query:
        state["web_search_results"] = []
        return state

    print("[WebSearchAgent] Calling Tavily search...")

    results = tavily_tool.invoke({"query": query})

    formatted: List[Dict] = []

    for r in results:
        formatted.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content")
        })

    state["web_search_results"] = formatted
    return state


# ======================================================
# Standalone Function (optional simple call)
# ======================================================

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Direct function call without LangGraph.
    Useful for quick testing.
    """

    tool = TavilySearchResults(
        max_results=max_results,
        api_key=TAVILY_API_KEY
    )

    results = tool.invoke({"query": query})

    formatted = []
    for r in results:
        formatted.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content")
        })

    return formatted


# ======================================================
# Manual Test
# ======================================================

if __name__ == "__main__":

    # -------------------------------
    # Test 1: Standalone Function
    # -------------------------------
    print("\n===== Standalone Web Search Test =====\n")

    test_query = "2025 clinical guideline myocarditis management"

    results = web_search(test_query, max_results=3)

    for r in results:
        print("=" * 60)
        print("Title:", r["title"])
        print("URL:", r["url"])
        print("Snippet:", (r["snippet"] or "")[:300])

    # -------------------------------
    # Test 2: LangGraph Node Simulation
    # -------------------------------
    print("\n===== LangGraph Node Test =====\n")

    test_state = {
        "query": "new rare disease treatment guidelines 2025"
    }

    updated_state = web_search_node(test_state)

    for r in updated_state["web_search_results"]:
        print("=" * 60)
        print("Title:", r["title"])
        print("URL:", r["url"])
        print("Snippet:", (r["snippet"] or "")[:300])

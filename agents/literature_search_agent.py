"""
Literature Search Agent
----------------------
Responsibility:
- Fetch medical research papers, guidelines, evidence
- Query-based PubMed search
- No reasoning
- No hallucination
- No LLM usage
"""

from typing import List, Dict
from tools.pubmed import retrieve_evidence


# ======================================================
# Query-based Literature Search (PRIMARY)
# ======================================================

def literature_search(query: str) -> Dict:
    """
    Searches PubMed with AUTOMATIC QUERY EXPANSION.
    
    If user sends "aspirin", we convert it to:
    "aspirin AND (systematic review[pt] OR guideline[pt] OR mechanism)"
    """

    if not query or not query.strip():
        return {}

    # --- SMART QUERY EXPANSION ---
    # If the query is short (likely a single topic/drug), 
    # force PubMed to look for high-value evidence types.
    clean_query = query.strip()
    
    # Heuristic: If query is short (< 3 words), it's likely a topic, not a specific question.
    if len(clean_query.split()) < 3:
        # [pt] stands for Publication Type in PubMed syntax
        expanded_query = f"({clean_query}) AND (systematic review[pt] OR practice guideline[pt] OR review[pt])"
        print(f"[INFO] Expanded query: '{clean_query}' -> '{expanded_query}'")
    else:
        # If user asks a specific question ("does aspirin cause bleeding"), keep it as is
        expanded_query = clean_query

    pubmed_data = retrieve_evidence(
        query=expanded_query,
        top_k=5,
        years=5 # Reduced to 5 years for fresher guidelines
    )

    return {
        "query": expanded_query, # Return the actual query used
        "original_query": query,
        "pubmed": {
            "pmids": pubmed_data.get("pmids", []),
            "metadata": pubmed_data.get("metadata", {}),
            "fulltexts": pubmed_data.get("fulltexts", {})
        }
    }


# ======================================================
# Drug-based Literature (SECONDARY / SUPPORTING)
# ======================================================

def fetch_drug_literature(drug_names: List[str]) -> List[Dict]:
    """
    Fetches PubMed literature specific to drugs.
    """
    results: List[Dict] = []

    for drug in drug_names:
        # We explicitly add clinical terms to get better matches
        query = f"{drug} AND (adverse effects OR therapeutic use OR pharmacology)"
        
        print(f"[INFO] PubMed drug search: {query}")

        pubmed_data = retrieve_evidence(
            query=query,
            top_k=5,
            years=10
        )

        results.append({
            "drug": drug,
            "pubmed": {
                "pmids": pubmed_data.get("pmids", []),
                "metadata": pubmed_data.get("metadata", {}),
                "fulltexts": pubmed_data.get("fulltexts", {})
            }
        })

    return results


if __name__ == "__main__":

    query = "Recent studies on Imatinib resistance"

    result = literature_search(query)

    print("\nLiterature Search Output\n")
    print("Query:", result["query"])
    print("\n[PubMed]")
    print("PMIDs:", result["pubmed"]["pmids"])

    for pmid, meta in result["pubmed"]["metadata"].items():
        print("\n PMID:", pmid)
        print("Title:", meta.get("title"))
        print("Evidence Score:", meta.get("evidence_score"))
        print("Abstract Preview:", meta.get("abstract", ""))
        print("Links:", meta.get("links"))

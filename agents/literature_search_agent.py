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
    Searches PubMed using a free-form medical query.

    Input:
        "Imatinib resistance mechanisms in CML"

    Output:
        {
          "query": str,
          "pubmed": {
              "pmids": [...],
              "metadata": {...},
              "fulltexts": {...}
          }
        }
    """

    if not query or not query.strip():
        return {}

    print(f"[INFO] PubMed query search: {query}")

    pubmed_data = retrieve_evidence(
        query=query,
        top_k=5,
        years=10
    )

    return {
        "query": query,
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
    Used only when drug intelligence is required.
    """

    results: List[Dict] = []

    for drug in drug_names:

        print(f"[INFO] PubMed drug search: {drug}")

        query = f"{drug} adverse effects contraindications mechanism"

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

    query = "Mechanisms of Imatinib resistance in chronic myeloid leukemia"

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

"""
Drug Intelligence Agent
-----------------------
Responsibility:
- Retrieve deterministic drug intelligence
- PubChem → chemical properties
- PubMed → clinical literature evidence
- No reasoning
- No hallucination
- No LLM usage

Triggered when:
- Drug entities appear in query or clinical NLP output

Returns:
- Structured drug intelligence objects
"""

from typing import List, Dict

from tools.pubchem import get_drug_profile
from tools.pubmed import retrieve_evidence


# ======================================================
# Utility: Extract drugs from clinical NLP entities
# ======================================================

def extract_drug_names(entities: List[Dict]) -> List[str]:
    """
    Extracts unique medication names from clinical_nlp output.

    Input:
        clinical_nlp entities list

    Output:
        ["aspirin", "metformin"]
    """

    drugs = set()

    for e in entities:
        if e.get("type") == "medication":
            drugs.add(e["entity"].lower())

    return list(drugs)


# ======================================================
# Core Drug Intelligence Fetcher
# ======================================================

def fetch_drug_intelligence(drug_names: List[str]) -> List[Dict]:
    """
    Fetches PubChem + PubMed intelligence for a list of drugs.

    Input:
        ["aspirin", "metformin"]

    Output:
        [
          {
            "drug": "aspirin",
            "pubchem": {...},
            "pubmed": {...}
          }
        ]
    """

    results: List[Dict] = []

    for drug in drug_names:

        # --------------------------
        # PubChem
        # --------------------------
        print(f"[INFO] Calling PubChem for: {drug}")
        pubchem_profile = get_drug_profile(drug)

        # --------------------------
        # PubMed
        # --------------------------
        print(f"[INFO] Calling PubMed for: {drug}")
        pubmed_query = f"{drug} adverse effects contraindications mechanism"
        pubmed_data = retrieve_evidence(
            query=pubmed_query,
            top_k=5,
            years=10
        )

        # --------------------------
        # Assemble deterministic output
        # --------------------------
        results.append({
            "drug": drug,
            "pubchem": {
                "found": pubchem_profile.get("found", False),
                "cid": pubchem_profile.get("cid"),
                "properties": pubchem_profile.get("properties", {}),
                "raw_description": pubchem_profile.get("description", ""),
                "raw_warnings": pubchem_profile.get("warnings", "")
            },
            "pubmed": {
                "pmids": pubmed_data.get("pmids", []),
                "metadata": pubmed_data.get("metadata", {}),
                "fulltexts": pubmed_data.get("fulltexts", {})
            }
        })

    return results


# ======================================================
# Manual Test
# ======================================================

if __name__ == "__main__":

    # Simulated NLP entities
    sample_entities = [
        {"entity": "aspirin", "type": "medication"},
        {"entity": "metformin", "type": "medication"}
    ]

    drug_list = extract_drug_names(sample_entities)

    print("\nDetected drugs:", drug_list)

    profiles = fetch_drug_intelligence(drug_list)

    print("\nDrug Intelligence Output\n")

    for p in profiles:
        print("=" * 70)
        print("Drug:", p["drug"])

        print("\n[PubChem]")
        print("Found:", p["pubchem"]["found"])
        print("CID:", p["pubchem"]["cid"])
        print("Properties:", p["pubchem"]["properties"])

        print("\n[PubMed]")
        print("PMIDs:", p["pubmed"]["pmids"])

        for pmid, meta in p["pubmed"]["metadata"].items():
            print("\n PMID:", pmid)
            print("Title:", meta.get("title"))
            print("Evidence Score:", meta.get("evidence_score"))
            print("Abstract Preview:", meta.get("abstract", "")[:200])
            print("Links:", meta.get("links"))


"""
Drug Intelligence Agent
-----------------------
Responsibility:
- Retrieve deterministic drug intelligence
- PubChem → chemical properties
- No reasoning
- No hallucination
- No LLM usage
"""

from typing import List, Dict
from tools.pubchem import get_drug_profile


# ======================================================
# Utility: Extract drugs from clinical NLP entities
# ======================================================

def extract_drug_names(entities: List[Dict]) -> List[str]:
    """
    Extracts unique medication names from clinical_nlp output.
    """

    drugs = set()

    for e in entities:
        if e.get("type") == "medication":
            drugs.add(e["entity"].lower())

    return list(drugs)


# ======================================================
# Core Drug Intelligence Fetcher (PubChem ONLY)
# ======================================================

def fetch_drug_intelligence(drug_names: List[str]) -> List[Dict]:
    """
    Fetches PubChem intelligence for a list of drugs.
    """

    results: List[Dict] = []

    for drug in drug_names:

        print(f"[INFO] Calling PubChem for: {drug}")
        pubchem_profile = get_drug_profile(drug)

        results.append({
            "drug": drug,
            "pubchem": {
                "found": pubchem_profile.get("found", False),
                "cid": pubchem_profile.get("cid"),
                "properties": pubchem_profile.get("properties", {}),
                "raw_description": pubchem_profile.get("description", ""),
                "raw_warnings": pubchem_profile.get("warnings", "")
            }
        })

    return results


# ======================================================
# Manual Test
# ======================================================

if __name__ == "__main__":

    sample_entities = [
        {"entity": "aspirin", "type": "medication"}
    ]

    drug_list = extract_drug_names(sample_entities)
    profiles = fetch_drug_intelligence(drug_list)

    for p in profiles:
        print("=" * 70)
        print("Drug:", p["drug"])
    
        print("\n[PubChem]")
        print("Found:", p["pubchem"]["found"])
        print("CID:", p["pubchem"]["cid"])
        print("Properties:", p["pubchem"]["properties"])


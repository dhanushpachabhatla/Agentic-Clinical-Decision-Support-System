"""
PubChem Agent
-------------
Responsibility:
- Retrieve deterministic drug intelligence from PubChem
- No reasoning
- No hallucination
- No LLM usage

Used for:
- Drug context
- Contraindication awareness
- Mechanism lookup
"""

import requests
from typing import Dict, Optional

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# --------------------------------------------------
# Internal helpers
# --------------------------------------------------

def _get(url: str) -> Dict:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------
# Drug name → CID
# --------------------------------------------------

def get_cid(drug_name: str) -> Optional[int]:
    """
    Resolve drug name to PubChem CID
    """
    url = f"{BASE_URL}/compound/name/{drug_name}/cids/JSON"
    try:
        data = _get(url)
        return data["IdentifierList"]["CID"][0]
    except Exception:
        return None


# --------------------------------------------------
# Core chemical properties
# --------------------------------------------------

def get_basic_properties(cid: int) -> Dict:
    """
    Fetch basic chemical properties
    """
    props = [
        "MolecularFormula",
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount"
    ]

    prop_str = ",".join(props)
    url = f"{BASE_URL}/compound/cid/{cid}/property/{prop_str}/JSON"

    try:
        data = _get(url)
        return data["PropertyTable"]["Properties"][0]
    except Exception:
        return {}


# --------------------------------------------------
# Drug description / mechanism
# --------------------------------------------------

def get_description(cid: int) -> str:
    """
    Fetch textual drug description / mechanism
    """
    url = f"{BASE_URL}/compound/cid/{cid}/description/JSON"

    try:
        data = _get(url)
        info = data.get("InformationList", {}).get("Information", [])
        if info:
            return info[0].get("Description", "")
    except Exception:
        pass

    return ""


# --------------------------------------------------
# Drug warnings (where available)
# --------------------------------------------------

def get_warnings(cid: int) -> str:
    """
    Fetch safety / warning information if present
    """
    url = f"{BASE_URL}/compound/cid/{cid}/xrefs/JSON"

    try:
        data = _get(url)
        return str(data)
    except Exception:
        return ""


# --------------------------------------------------
# High-level API
# --------------------------------------------------

def get_drug_profile(drug_name: str) -> Dict:
    """
    High-level drug lookup
    """
    cid = get_cid(drug_name)

    if not cid:
        return {
            "drug": drug_name,
            "found": False
        }

    return {
        "drug": drug_name,
        "found": True,
        "cid": cid,
        "properties": get_basic_properties(cid),
        "description": get_description(cid),
        "warnings": get_warnings(cid)
    }


# --------------------------------------------------
# Manual test
# --------------------------------------------------

if __name__ == "__main__":
    drug = "aspirin"
    profile = get_drug_profile(drug)

    print("\nDrug:", profile["drug"])
    print("Found:", profile["found"])
    print("CID:", profile.get("cid"))
    print("Properties:", profile.get("properties"))
    print("Description:", (profile.get("description") or "")[:300])

# The API couldnt fetch the Description, let the LLM do the job


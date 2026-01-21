import time
import requests
from typing import List, Dict, Optional

# ======================================================
# CONFIGURATION
# ======================================================

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMC_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"

API_KEY = None  # Optional but recommended
EMAIL = "your_email@example.com"  # REQUIRED by NCBI policy
TOOL = "agentic-cdss-evidence-agent"

HEADERS = {
    "User-Agent": f"{TOOL} ({EMAIL})"
}

RATE_LIMIT_SLEEP = 0.34  # ~3 req/sec without API key
CURRENT_YEAR = 2025


# ======================================================
# EVIDENCE TIERS
# ======================================================

EVIDENCE_TIERS = {
    "guideline": 1.0,
    "meta-analysis": 0.9,
    "systematic review": 0.85,
    "review": 0.8,
    "randomized controlled trial": 0.7,
    "clinical trial": 0.65,
    "case reports": 0.3,
}


# ======================================================
# INTERNAL UTILS
# ======================================================

def _sleep():
    time.sleep(RATE_LIMIT_SLEEP)


def _request(url: str, params: Dict) -> requests.Response:
    params["tool"] = TOOL
    params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY

    response = requests.get(url, params=params, headers=HEADERS, timeout=20)
    response.raise_for_status()
    _sleep()
    return response


def infer_evidence_score(pubtypes: List[str]) -> float:
    for pt in pubtypes:
        pt_lower = pt.lower()
        for key, score in EVIDENCE_TIERS.items():
            if key in pt_lower:
                return score
    return 0.4  # observational / unknown


# ======================================================
# PUBMED SEARCH (GUIDELINE FIRST)
# ======================================================

def search_pubmed(
    query: str,
    retmax: int = 10,
    years: Optional[int] = None
) -> List[str]:
    """
    Priority-based PubMed search:
    1. Guidelines
    2. Reviews / Meta-analyses
    3. Trials
    4. Case reports (fallback)
    """

    base = f"({query})"

    if years:
        base += f" AND {CURRENT_YEAR - years}:{CURRENT_YEAR}[pdat]"

    priority_queries = [
        f"{base} AND guideline[pt]",
        f"{base} AND (meta-analysis[pt] OR systematic review[pt] OR review[pt])",
        f"{base} AND (randomized controlled trial[pt] OR clinical trial[pt])",
        f"{base} AND case reports[pt]",
    ]

    pmids = []

    for q in priority_queries:
        params = {
            "db": "pubmed",
            "term": q,
            "retmode": "json",
            "retmax": retmax,
        }

        r = _request(BASE_URL + "esearch.fcgi", params)
        ids = r.json()["esearchresult"]["idlist"]

        for pid in ids:
            if pid not in pmids:
                pmids.append(pid)

        if len(pmids) >= retmax:
            break

    return pmids[:retmax]


# ======================================================
# METADATA
# ======================================================

def fetch_metadata(pmids: List[str]) -> Dict[str, Dict]:
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }

    r = _request(BASE_URL + "esummary.fcgi", params)
    data = r.json()["result"]

    metadata = {}

    for pmid in pmids:
        doc = data.get(pmid)
        if not doc:
            continue

        pubtypes = doc.get("pubtype", [])

        metadata[pmid] = {
            "title": doc.get("title"),
            "journal": doc.get("fulljournalname"),
            "year": doc.get("pubdate", "")[:4],
            "article_types": pubtypes,
            "evidence_score": infer_evidence_score(pubtypes),
        }

    return metadata


# ======================================================
# ABSTRACTS
# ======================================================

def fetch_abstracts(pmids: List[str]) -> Dict[str, str]:
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }

    r = _request(BASE_URL + "efetch.fcgi", params)

    return {pmid: r.text for pmid in pmids}


# ======================================================
# PMID → PMCID
# ======================================================

def map_pmid_to_pmcid(pmids: List[str]) -> Dict[str, str]:
    if not pmids:
        return {}

    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": ",".join(pmids),
        "retmode": "json",
    }

    r = _request(BASE_URL + "elink.fcgi", params)
    mapping = {}

    for linkset in r.json().get("linksets", []):
        pmid = linkset.get("ids", [None])[0]
        links = (
            linkset.get("linksetdbs", [{}])[0]
            .get("links", [])
        )
        if pmid and links:
            mapping[pmid] = f"PMC{links[0]}"

    return mapping


# ======================================================
# PMC FULL TEXT
# ======================================================

def fetch_pmc_fulltext(pmcid: str) -> str:
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{pmcid.replace('PMC', '')}",
        "metadataPrefix": "pmc",
    }

    r = requests.get(PMC_BASE, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    _sleep()
    return r.text


# ======================================================
# HIGH-LEVEL PIPELINE
# ======================================================

def retrieve_evidence(
    query: str,
    top_k: int = 5,
    years: int = 10
) -> Dict:
    pmids = search_pubmed(query, retmax=top_k, years=years)
    metadata = fetch_metadata(pmids)
    abstracts = fetch_abstracts(pmids)
    pmc_map = map_pmid_to_pmcid(pmids)

    fulltexts = {}
    for pmid, pmcid in pmc_map.items():
        try:
            fulltexts[pmid] = fetch_pmc_fulltext(pmcid)
        except Exception:
            continue

    return {
        "pmids": pmids,
        "metadata": metadata,
        "abstracts": abstracts,
        "fulltexts": fulltexts,
    }


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":
    result = retrieve_evidence(
        query="Head Ache",
        top_k=5,
        years=10
    )

    print("\nPMIDs:", result["pmids"])

    for pmid, meta in result["metadata"].items():
        print(
            f"\n{pmid} | {meta['title']} "
            f"(Score: {meta['evidence_score']})"
        )

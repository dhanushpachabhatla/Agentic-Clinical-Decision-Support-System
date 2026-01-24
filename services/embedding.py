"""
Embedding Agent
----------------
Responsibility:
- Convert structured Clinical NLP JSON output into embeddings
- NO reasoning
- NO interpretation
- Deterministic transformation only

Embedding Model:
- Google Gemini: text-embedding-004
"""

import os
import json
from typing import Dict, List, Union

# =====================================================
# DEPENDENCY CHECK
# =====================================================

try:
    from google import genai
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: google-genai\n"
        "Install with: pip install google-genai"
    ) from e


# =====================================================
# CONFIG
# =====================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "Set it using:\n"
        "  $env:GEMINI_API_KEY='your_key_here'  (PowerShell)"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "text-embedding-004"


# =====================================================
# PUBLIC API
# =====================================================

def embed_clinical_json(
    clinical_output: Union[Dict, str]
) -> Dict:
    """
    Generate embeddings for Clinical NLP output.
    """

    data = _load_json(clinical_output)

    entities = data.get("entities", [])
    metadata = data.get("doc_metadata", {})

    records = []

    for ent in entities:
        text = _entity_to_text(ent)
        if not text.strip():
            continue

        vector = _embed_text(text)

        records.append({
            "entity": ent.get("entity"),
            "type": ent.get("type"),
            "normalized": ent.get("normalized"),
            "embedding": vector
        })

    return {
        "doc_metadata": metadata,
        "embedding_model": EMBEDDING_MODEL,
        "embeddings": records
    }


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _load_json(inp: Union[Dict, str]) -> Dict:
    if isinstance(inp, dict):
        return inp

    if isinstance(inp, str):
        with open(inp, "r", encoding="utf-8") as f:
            return json.load(f)

    raise TypeError("Input must be dict or JSON file path")


def _entity_to_text(ent: Dict) -> str:
    """
    Deterministic textual representation of an entity.
    """

    parts = [
        f"Entity: {ent.get('entity')}",
        f"Type: {ent.get('type')}"
    ]

    if ent.get("value") is not None:
        parts.append(f"Value: {ent.get('value')}")

    if ent.get("unit"):
        parts.append(f"Unit: {ent.get('unit')}")

    if ent.get("section"):
        parts.append(f"Section: {ent.get('section')}")

    return " | ".join(parts)


def _embed_text(text: str) -> List[float]:
    """
    Correct Gemini embedding call (google-genai).
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


# =====================================================
# STANDALONE TEST (WITH EMBEDDING DISPLAY)
# =====================================================

def _standalone_test():
    sample_clinical_output = {
        "doc_metadata": {
            "source": "sample_note.txt",
            "date": "2024-08-14",
            "doc_type": "lab_report"
        },
        "entities": [
            {
                "entity": "SGOT",
                "type": "lab",
                "normalized": "SGOT",
                "value": "162",
                "unit": "U/L",
                "section": "laboratory_results"
            },
            {
                "entity": "ALBUMIN",
                "type": "lab",
                "normalized": "ALBUMIN",
                "value": "3.7",
                "unit": "g/dL",
                "section": "laboratory_results"
            }
        ]
    }

    output = embed_clinical_json(sample_clinical_output)

    print("\n[INFO] Embedding test successful\n")

    for idx, emb in enumerate(output["embeddings"], start=1):
        vector = emb["embedding"]

        print("=" * 70)
        print(f"Entity {idx}: {emb['entity']}")
        print(f"Type    : {emb['type']}")
        print(f"Dim     : {len(vector)}")
        print(f"Preview : {vector[:10]}")  # first 10 values only
        print("=" * 70)

    print("\n[INFO] Total embeddings:", len(output["embeddings"]))


if __name__ == "__main__":
    _standalone_test()

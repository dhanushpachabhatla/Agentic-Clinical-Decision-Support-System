"""
Pinecone Vector Store Agent
---------------------------
Responsibility:
- Store embeddings in Pinecone
- NO reasoning
- NO interpretation
- Deterministic upsert only
"""

import os
from typing import Dict, List
from pinecone import Pinecone


# =====================================================
# CONFIG (loaded from .env)
# =====================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_URL = os.getenv("PINECONE_INDEX_URL")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment")

if not PINECONE_INDEX_URL:
    raise RuntimeError("PINECONE_INDEX_URL not found in environment")


# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# NOTE: host= is REQUIRED when using serverless indexes
index = pc.Index(host=PINECONE_INDEX_URL)


# =====================================================
# PUBLIC API
# =====================================================

def upsert_embeddings(embedding_output: Dict) -> Dict:
    """
    Upsert embeddings into Pinecone.

    Input:
    - Output from embedding.py

    Output:
    {
        "upserted_vectors": int,
        "index": str
    }
    """

    metadata = embedding_output.get("doc_metadata", {})
    embeddings = embedding_output.get("embeddings", [])

    vectors: List[Dict] = []

    for idx, item in enumerate(embeddings):
        vector_id = _build_vector_id(metadata, item, idx)

        # ---------------------------------------------
        # Metadata (MUST be Pinecone-safe)
        # - No None values
        # - Only str, int, float, bool, list[str]
        # ---------------------------------------------
        meta = {
            "entity": item.get("entity"),
            "type": item.get("type"),
            "normalized": item.get("normalized"),
            "source": metadata.get("source"),
            "date": metadata.get("date"),
            "doc_type": metadata.get("doc_type"),
        }

        #  CRITICAL FIX: remove None values
        meta = {k: v for k, v in meta.items() if v is not None}

        vectors.append({
            "id": vector_id,
            "values": item["embedding"],
            "metadata": meta
        })

    if not vectors:
        return {
            "upserted_vectors": 0,
            "index": PINECONE_INDEX_URL
        }

    # Perform upsert
    index.upsert(vectors=vectors)

    return {
        "upserted_vectors": len(vectors),
        "index": PINECONE_INDEX_URL
    }


# =====================================================
# HELPERS
# =====================================================

def _build_vector_id(metadata: Dict, item: Dict, idx: int) -> str:
    """
    Deterministic vector ID.
    Format:
    source:normalized_entity:index
    """
    source = metadata.get("source", "document")
    entity = item.get("normalized", "ENTITY")
    return f"{source}:{entity}:{idx}"


# =====================================================
# STANDALONE TEST
# =====================================================

def _standalone_test():
    """
    Run this file directly to test Pinecone upsert.
    """

    sample_embedding_output = {
        "doc_metadata": {
            "source": "sample_note.txt",
            "date": "2024-08-14",
            "doc_type": "lab_report"
        },
        "embedding_model": "text-embedding-004",
        "embeddings": [
            {
                "entity": "SGOT",
                "type": "lab",
                "normalized": "SGOT",
                "embedding": [0.01] * 768
            },
            {
                "entity": "ALBUMIN",
                "type": "lab",
                "normalized": "ALBUMIN",
                "embedding": [0.02] * 768
            }
        ]
    }

    result = upsert_embeddings(sample_embedding_output)

    print("\n[INFO] Pinecone upsert result:")
    print(result)


if __name__ == "__main__":
    _standalone_test()

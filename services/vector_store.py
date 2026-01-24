"""
Vector Store (Qdrant)
--------------------
Responsibility:
- Persist and retrieve embedding vectors
- Support Hybrid RAG (vector + metadata filtering)

Designed for:
- Clinical RAG
- Explainability
- Temporal & section-based filtering
"""

from typing import List, Dict, Optional
import asyncio
from services.embedding import embed_chunks
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


# -----------------------------
# Configuration
# -----------------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "clinical_rag_chunks"
VECTOR_DIMENSION = 768  # MUST match embedding output
DISTANCE_METRIC = Distance.COSINE


# -----------------------------
# Client
# -----------------------------

# _client = QdrantClient(
#     host=QDRANT_HOST,
#     port=QDRANT_PORT,
# )
_client = QdrantClient(path="./qdrant_data")



# -----------------------------
# Collection setup
# -----------------------------

def init_collection(force_recreate: bool = False) -> None:
    """
    Initializes the Qdrant collection.
    """

    collections = _client.get_collections().collections
    existing = {c.name for c in collections}

    if COLLECTION_NAME in existing:
        if not force_recreate:
            return
        _client.delete_collection(COLLECTION_NAME)

    _client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIMENSION,
            distance=DISTANCE_METRIC,
        ),
    )


# -----------------------------
# Insert embeddings
# -----------------------------

def upsert_embeddings(records: List[Dict]) -> None:
    """
    Inserts or updates embedding records.
    Uses UUIDs for Qdrant compatibility (local mode).
    """

    points: List[PointStruct] = []

    for r in records:
        qdrant_id = uuid.uuid4().hex  # valid UUID

        payload = r["metadata"].copy()
        payload["chunk_id"] = r["id"]  # preserve logical ID

        points.append(
            PointStruct(
                id=qdrant_id,
                vector=r["vector"],
                payload=payload,
            )
        )

    if points:
        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )



# -----------------------------
# Retrieval
# -----------------------------

def query_similar(
    query_vector: List[float],
    top_k: int = 5,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Hybrid similarity search with optional metadata filters.
    """

    qdrant_filter = _build_filter(filters) if filters else None

    results = _client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[],
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_filter,
    )

    return [
        {
            "id": r.payload.get("chunk_id"),
            "score": r.score,
            "payload": r.payload,
        }
        for r in results.points
    ]



# -----------------------------
# Filter builder
# -----------------------------

def _build_filter(filters: Dict) -> Filter:
    """
    Converts simple filter dict into Qdrant Filter.
    """

    conditions = []

    for key, value in filters.items():
        if value is None:
            continue

        conditions.append(
            FieldCondition(
                key=key,
                match=MatchValue(value=value),
            )
        )

    return Filter(must=conditions)


# -----------------------------
# MAIN (Integration Test)
# -----------------------------

async def main():
    print("[INFO] Initializing Qdrant collection...")
    init_collection(force_recreate=True)

    # ---- Sample RAG chunk (realistic) ----
    sample_chunks = [
        {
            "chunk_id": "test_chunk_1",
            "text": (
                "Patient presents with acute chest pain radiating to the left arm. "
                "Cardiac troponin is elevated at 1.2 ng/mL, raising concern for myocardial injury."
            ),
            "entities": ["CHEST_PAIN", "TROPONIN"],
            "entity_types": ["symptom", "lab"],
            "section": "investigation",
            "date": "2024-08-14",
            "source": "er_note_1.txt",
            "metadata": {
                "doc_type": "clinical_note",
                "negated_entities": [],
                "labs": [
                    {
                        "lab": "TROPONIN",
                        "value": "1.2",
                        "unit": "ng/mL",
                        "context": "Cardiac troponin is elevated at 1.2 ng/mL"
                    }
                ]
            }
        }
    ]

    print("[INFO] Embedding chunks using Gemini...")
    embedded_records = await embed_chunks(sample_chunks)

    if not embedded_records:
        print("[ERROR] No embeddings generated.")
        return

    print(
        f"[INFO] Generated embeddings with dimension: "
        f"{len(embedded_records[0]['vector'])}"
    )

    print("[INFO] Upserting embeddings into Qdrant...")
    upsert_embeddings(embedded_records)

    # ---- Query using same embedding (sanity check) ----
    query_vector = embedded_records[0]["vector"]

    print("[INFO] Querying vector store...")
    results = query_similar(
        query_vector=query_vector,
        top_k=3,
        filters={"section": "investigation"}
    )

    print("\nSearch Results:\n")
    for r in results:
        print(f"ID    : {r['id']}")
        print(f"Score : {r['score']:.4f}")
        print(f"Meta  : {r['payload']}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())

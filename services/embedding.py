"""
Embedding Agent (Gemini - google.genai)
--------------------------------------
Responsibility:
- Convert RAG chunks into dense embeddings
- Use Gemini embedding models via google.genai
- Produce vector records for Hybrid RAG

Guarantees:
- No chunk modification
- No retrieval logic
- No reasoning
"""

from typing import List, Dict
import os
import asyncio
import time
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY1")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY1 not set")

EMBEDDING_MODEL = "text-embedding-004"
OUTPUT_DIM = 3072
TASK_TYPE = "RETRIEVAL_DOCUMENT"

# Rate safety
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


# -----------------------------
# Client
# -----------------------------

_client = genai.Client(api_key=GEMINI_API_KEY)


# -----------------------------
# Public API
# -----------------------------

async def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Converts RAG chunks into embedding records.

    Input:
        [
          {
            "chunk_id": "...",
            "text": "...",
            "metadata": {...}
          }
        ]

    Output:
        [
          {
            "id": "chunk_id",
            "vector": [...],
            "metadata": {...}
          }
        ]
    """

    if not chunks:
        return []

    texts: List[str] = []
    index_map: List[int] = []

    for idx, chunk in enumerate(chunks):
        text = _clean_text(chunk.get("text", ""))
        if text:
            texts.append(text)
            index_map.append(idx)

    if not texts:
        return []

    embeddings = await _embed_texts(texts)
    print(f"[INFO] Gemini embedding dimension: {len(embeddings[0])}")


    records: List[Dict] = []
    for emb, chunk_idx in zip(embeddings, index_map):
        chunk = chunks[chunk_idx]

        records.append({
            "id": chunk["chunk_id"],
            "vector": emb,
            "metadata": _build_metadata(chunk),
        })

    return records


# -----------------------------
# Embedding helpers
# -----------------------------

async def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batched embedding call with retries.
    """

    for attempt in range(MAX_RETRIES):
        try:
            result = _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=EmbedContentConfig(
                    task_type=TASK_TYPE,
                    output_dimensionality=OUTPUT_DIM
                )
            )

            return [e.values for e in result.embeddings]

        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg and attempt < MAX_RETRIES - 1:
                print(
                    f"[WARN] Gemini rate-limited. "
                    f"Retrying in {RETRY_DELAY}s (attempt {attempt + 1})..."
                )
                await asyncio.sleep(RETRY_DELAY)
                continue

            raise RuntimeError(f"[ERROR] Gemini embedding failed: {e}")

    return []


def _clean_text(text: str) -> str:
    """
    Lightweight normalization before embedding.
    """
    return " ".join(text.split()).strip()


def _build_metadata(chunk: Dict) -> Dict:
    """
    Builds retrieval-friendly metadata.
    """

    meta = chunk.get("metadata", {})

    return {
        "source": chunk.get("source"),
        "date": chunk.get("date"),
        "section": chunk.get("section"),
        "doc_type": meta.get("doc_type"),
        "entities": chunk.get("entities", []),
        "entity_types": chunk.get("entity_types", []),
        "negated_entities": meta.get("negated_entities", []),
        "labs": meta.get("labs", []),
    }


# -----------------------------
# MAIN (Manual Test)
# -----------------------------

async def main():
    sample_chunks = [
        {
            "chunk_id": "test_chunk_1",
            "text": (
                "Patient presents with acute chest pain radiating to the left arm. "
                "Cardiac troponin is elevated, raising concern for myocardial injury."
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

    records = await embed_chunks(sample_chunks)

    print("\nEmbedded Records:\n")
    for r in records:
        print(f"ID: {r['id']}")
        print(f"Vector length: {len(r['vector'])}")
        print(f"Metadata keys: {list(r['metadata'].keys())}")
        print("-" * 60)

    
# -----------------------------
# Query Embedding (for retrieval)
# -----------------------------

async def embed_query(query: str) -> List[float]:
    """
    Embeds a single query string for retrieval.

    Input:
        "Is elevated troponin indicative of myocardial infarction?"

    Output:
        [float, float, ...]
    """

    if not query or not query.strip():
        return []

    cleaned = _clean_text(query)

    embeddings = await _embed_texts([cleaned])

    return embeddings[0]


if __name__ == "__main__":
    asyncio.run(main())

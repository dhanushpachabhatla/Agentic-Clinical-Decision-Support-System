"""
Retriever Agent (Entity-centric, Pinecone)
------------------------------------------
Responsibility:
- Retrieve relevant clinical ENTITIES from Pinecone
- Support:
    1. Summary retrieval (dashboard)
    2. Q/A retrieval (chat)

Rules:
- NO LLM usage
- NO reasoning
- NO interpretation
- Deterministic retrieval only

Works with:
- services.embedding (Gemini)
- services.upsert (Pinecone schema)
"""

from typing import List, Dict, Optional
from pinecone import Pinecone
import os

# Gemini embedding
from services.embedding import _embed_text  # internal but deterministic


# =====================================================
# PINECONE CONFIG
# =====================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_URL = os.getenv("PINECONE_INDEX_URL")

if not PINECONE_API_KEY or not PINECONE_INDEX_URL:
    raise RuntimeError("Pinecone environment variables missing")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_URL)


# =====================================================
# RETRIEVAL CONFIG
# =====================================================

QA_TOP_K = 5
SUMMARY_TOP_K = 20


# =====================================================
# RETRIEVER AGENT
# =====================================================

class RetrieverAgent:
    """
    Entity-centric retriever.
    Orchestrator decides WHICH method to call.
    """

    # -------------------------------------------------
    # Mode 1: Q/A Retrieval (High Precision)
    # -------------------------------------------------

    def retrieve_for_qa(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = QA_TOP_K
    ) -> List[Dict]:
        """
        Retrieve entities most relevant to a user question.
        """

        if not query or not query.strip():
            return []

        query_vector = _embed_text(query)

        pinecone_filter = self._build_filter(filters)

        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter
        )

        return self._postprocess(response.matches)


    # -------------------------------------------------
    # Mode 2: Summary Retrieval (High Recall)
    # -------------------------------------------------

    def retrieve_for_summary(
        self,
        filters: Optional[Dict] = None,
        top_k: int = SUMMARY_TOP_K
    ) -> List[Dict]:
        """
        Retrieve broad entity coverage for dashboard summary.
        """

        anchor_query = (
            "clinical summary diagnosis labs medications findings history"
        )

        query_vector = _embed_text(anchor_query)

        pinecone_filter = self._build_filter(filters)

        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter
        )

        return self._postprocess(response.matches)


# =====================================================
# INTERNAL HELPERS
# =====================================================

    def _build_filter(self, filters: Optional[Dict]) -> Optional[Dict]:
        """
        Convert simple filters into Pinecone filter.
        """
        if not filters:
            return None

        return {
            k: {"$eq": v}
            for k, v in filters.items()
            if v is not None
        }


    def _postprocess(self, matches) -> List[Dict]:
        """
        Normalize Pinecone matches.
        """

        results = []

        for m in matches:
            meta = m.metadata or {}

            results.append({
                "id": m.id,
                "score": m.score,
                "entity": meta.get("entity"),
                "normalized": meta.get("normalized"),
                "type": meta.get("type"),
                "source": meta.get("source"),
                "date": meta.get("date"),
                "doc_type": meta.get("doc_type"),
            })

        # Sort explicitly (Pinecone already does, but be safe)
        results.sort(key=lambda x: x["score"], reverse=True)

        return results


# =====================================================
# MANUAL TEST
# =====================================================

if __name__ == "__main__":

    retriever = RetrieverAgent()

    print("\n[TEST] Q/A Retrieval\n")

    qa_results = retriever.retrieve_for_qa(
        query="Is elevated SGOT indicative of liver injury?",
        filters={"doc_type": "lab_report"}
    )

    for r in qa_results:
        print("=" * 70)
        print("Entity :", r["entity"])
        print("Type   :", r["type"])
        print("Score  :", r["score"])
        print("Source :", r["source"])
        print("Date   :", r["date"])

    print("\n[TEST] Summary Retrieval\n")

    summary_results = retriever.retrieve_for_summary(
        filters={"doc_type": "lab_report"}
    )

    for r in summary_results:
        print("=" * 70)
        print("Entity :", r["entity"])
        print("Type   :", r["type"])
        print("Score  :", r["score"])

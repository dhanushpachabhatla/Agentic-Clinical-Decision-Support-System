"""
Retriever Agent
----------------
Responsibility:
- Retrieve relevant RAG chunks from vector store
- Support TWO retrieval modes:
    1. Summary retrieval (dashboard)
    2. Q/A retrieval (chat)

Rules:
- NO embeddings generation (delegated)
- NO reasoning
- NO LLM usage
- NO tool calling
- Deterministic retrieval only

Uses:
- vector_store.query_similar
- embedding_agent.embed_query
"""

from typing import List, Dict, Optional
from services.embedding import embed_query
from services.vector_store import query_similar


# ======================================================
# Configuration
# ======================================================

SUMMARY_TOP_K = 20
QA_TOP_K = 5

SUMMARY_SECTION_PRIORITY = [
    "diagnosis",
    "history",
    "investigation",
    "assessment",
    "treatment",
    "plan"
]


# ======================================================
# Public API
# ======================================================

class RetrieverAgent:
    """
    Central retrieval capability.
    Orchestrator decides WHICH method to call.
    """

    # --------------------------------------------------
    # Mode 1: Dashboard / Summary Retrieval
    # --------------------------------------------------

    async def retrieve_for_summary(
        self,
        patient_id: Optional[str] = None,
        sections: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieves a broad, stable set of chunks for dashboard summary.

        Characteristics:
        - High recall
        - Section-aware
        - Time-aware
        - Deterministic
        """

        # Use a generic summary anchor query
        query_text = "patient clinical summary medical history findings"

        query_vector = await embed_query(query_text)

        filters = {}

        if sections:
            # NOTE: Qdrant filter supports single values.
            # Multiple sections handled via multiple calls.
            results = []
            for section in sections:
                section_results = query_similar(
                    query_vector=query_vector,
                    top_k=SUMMARY_TOP_K // len(sections),
                    filters={"section": section}
                )
                results.extend(section_results)

            return self._postprocess_results(results)

        if date:
            filters["date"] = date

        results = query_similar(
            query_vector=query_vector,
            top_k=SUMMARY_TOP_K,
            filters=filters if filters else None
        )

        return self._postprocess_results(results)


    # --------------------------------------------------
    # Mode 2: Q/A Retrieval
    # --------------------------------------------------

    async def retrieve_for_qa(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = QA_TOP_K,
    ) -> List[Dict]:
        """
        Retrieves precise chunks for question answering.

        Characteristics:
        - High precision
        - Query-driven
        - Token-budget aware
        """

        query_vector = await embed_query(query)

        results = query_similar(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters
        )

        return self._postprocess_results(results)


# ======================================================
# Internal Helpers
# ======================================================

    def _postprocess_results(self, results: List[Dict]) -> List[Dict]:
        """
        Normalizes and sorts retrieval output.
        """

        cleaned = []

        for r in results:
            payload = r.get("payload", {})

            cleaned.append({
                "chunk_id": r.get("id"),
                "score": r.get("score"),
                "section": payload.get("section"),
                "date": payload.get("date"),
                "source": payload.get("source"),
                "doc_type": payload.get("doc_type"),
                "negated_entities": payload.get("negated_entities", []),
                "labs": payload.get("labs", []),
            })

        # Sort by score (descending)
        cleaned.sort(key=lambda x: x["score"], reverse=True)

        return cleaned


# ======================================================
# Manual Test
# ======================================================

if __name__ == "__main__":
    import asyncio

    retriever = RetrieverAgent()

    async def test():

        print("\n[TEST] Q/A Retrieval\n")

        qa_results = await retriever.retrieve_for_qa(
            query="Is elevated troponin indicative of myocardial infarction?",
            filters={"section": "investigation"}
        )

        for r in qa_results:
            print("=" * 70)
            print("Chunk ID:", r["chunk_id"])
            print("Score:", r["score"])
            print("Section:", r["section"])
            print("Date:", r["date"])

        print("\n[TEST] Summary Retrieval\n")

        summary_results = await retriever.retrieve_for_summary(
            sections=["diagnosis", "investigation"]
        )

        for r in summary_results:
            print("=" * 70)
            print("Chunk ID:", r["chunk_id"])
            print("Score:", r["score"])
            print("Section:", r["section"])

    asyncio.run(test())

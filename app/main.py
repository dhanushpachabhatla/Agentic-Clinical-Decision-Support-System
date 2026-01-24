from dotenv import load_dotenv
load_dotenv()  # MUST be first – loads .env for entire pipeline

from pathlib import Path

from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator


def main():
    """
    Entry point for the Clinical Decision Support pipeline.

    STRICT:
    - No medical reasoning
    - No data mutation
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # -------------------------------------------------
    # INPUT FILES
    # -------------------------------------------------
    file_paths = [
        PROJECT_ROOT / "samples" / "report_1_medic.png",
    ]

    # Convert to strings + validate
    file_paths = [str(p) for p in file_paths if p.exists()]

    if not file_paths:
        print("[ERROR] No valid input files found.")
        return

    # -------------------------------------------------
    # INITIALIZE STATE + ORCHESTRATOR
    # -------------------------------------------------
    state = ClinicalState(file_paths=file_paths)
    orchestrator = ClinicalOrchestrator()

    # -------------------------------------------------
    # STEP 1: INGESTION
    # -------------------------------------------------
    state = orchestrator.run_ingestion(state)
    print(f"\n[INFO] Ingested {len(state.raw_documents)} document(s)")

    if not state.raw_documents:
        print("[ERROR] No documents ingested. Exiting.")
        return

    # -------------------------------------------------
    # STEP 2: CLINICAL NLP + DATE NORMALIZATION
    # -------------------------------------------------
    state = orchestrator.run_clinical_nlp(state)
    print(f"[INFO] NLP processed {len(state.nlp_results)} document(s)")
    print(f"[INFO] Normalized NLP results: {len(state.normalized_nlp_results)}")

    if not state.normalized_nlp_results:
        print("[ERROR] No normalized NLP results produced. Exiting.")
        return

    # Show canonical normalized date (if available)
    if state.normalized_date:
        print(f"[INFO] Canonical document date: {state.normalized_date}")

    # -------------------------------------------------
    # STEP 3: EMBEDDING (USES NORMALIZED DATA)
    # -------------------------------------------------
    state = orchestrator.run_embedding(state)
    print(f"[INFO] Generated embeddings for {len(state.embedding_results)} document(s)")

    if not state.embedding_results:
        print("[ERROR] No embeddings generated. Exiting.")
        return

    # -------------------------------------------------
    # STEP 4: VECTOR STORE (PINECONE)
    # -------------------------------------------------
    state = orchestrator.run_vector_upsert(state)
    print(f"[INFO] Upserted vectors for {len(state.vector_store_results)} document(s)\n")

    # -------------------------------------------------
    # OUTPUT SUMMARY
    # -------------------------------------------------
    for idx, item in enumerate(state.embedding_results, start=1):
        print("=" * 80)
        print(f"Document {idx}")
        print(f"Source           : {item.get('source')}")
        print(f"Embedding Model  : {item.get('embedding_model')}")
        print(f"Num Vectors      : {item.get('num_embeddings')}")
        print("-" * 80)

        # Safe preview of first vector only
        if item.get("embeddings"):
            first = item["embeddings"][0]
            print("First Entity :", first.get("entity"))
            print("Vector Dim  :", len(first.get("embedding", [])))
            print("Preview     :", first.get("embedding", [])[:10])

        print()

    # -------------------------------------------------
    # WARNINGS / ERRORS
    # -------------------------------------------------
    if state.errors:
        print("\n[WARNINGS]")
        for e in state.errors:
            print("-", e)


if __name__ == "__main__":
    main()

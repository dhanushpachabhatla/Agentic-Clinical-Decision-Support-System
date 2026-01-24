# app/main.py

from pathlib import Path
import json

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

    file_paths = [str(p) for p in file_paths if p.exists()]

    if not file_paths:
        print("[ERROR] No valid input files found.")
        return

    # -------------------------------------------------
    # INITIALIZE STATE
    # -------------------------------------------------
    state = ClinicalState(file_paths=file_paths)
    orchestrator = ClinicalOrchestrator()

    # -------------------------------------------------
    # STEP 1: INGESTION
    # -------------------------------------------------
    state = orchestrator.run_ingestion(state)
    print(f"\n[INFO] Ingested {len(state.raw_documents)} document(s)\n")

    if not state.raw_documents:
        print("[ERROR] No documents ingested. Exiting.")
        return

    # -------------------------------------------------
    # STEP 2: CLINICAL NLP
    # -------------------------------------------------
    state = orchestrator.run_clinical_nlp(state)
    print(f"[INFO] NLP processed {len(state.nlp_results)} document(s)\n")

    # -------------------------------------------------
    # STEP 3: EMBEDDING
    # -------------------------------------------------
    state = orchestrator.run_embedding(state)
    print(f"[INFO] Generated embeddings for {len(state.embedding_results)} document(s)\n")

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    for idx, item in enumerate(state.embedding_results, start=1):
        print("=" * 80)
        print(f"Embedded Document {idx}")
        print(f"Source           : {item.get('source')}")
        print(f"Embedding Model  : {item.get('embedding_model')}")
        print(f"Num Vectors      : {item.get('num_embeddings')}")
        print("-" * 80)

        # Preview first embedding only (safe)
        if item.get("embeddings"):
            preview = item["embeddings"][0]
            print("First Entity     :", preview.get("entity"))
            print("Vector Dim       :", len(preview.get("embedding", [])))
            print("Vector Preview   :", preview.get("embedding", [])[:10])

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

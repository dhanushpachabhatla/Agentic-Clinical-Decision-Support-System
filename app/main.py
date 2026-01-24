# app/main.py

from pathlib import Path
import json

from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator


def main():
    """
    Entry point for the Clinical Decision Support pipeline.

    Responsibilities:
    - Define input files
    - Initialize pipeline state
    - Run orchestrator steps
    - Display results and errors

    STRICT:
    - No medical reasoning
    - No data mutation
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # -------------------------------------------------
    # INPUT FILES (add/remove as needed)
    # -------------------------------------------------
    file_paths = [
        # PROJECT_ROOT / "samples" / "image.png",
        PROJECT_ROOT / "samples" / "report_1_medic.png",
        # PROJECT_ROOT / "samples" / "report_2_medic.png",
        # PROJECT_ROOT / "samples" / "Chest_Pain.pdf",
    ]

    # Convert to strings + validate existence
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

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    print(f"\n[INFO] NLP processed {len(state.nlp_results)} document(s)\n")

    for idx, item in enumerate(state.nlp_results, start=1):
        print("=" * 80)
        print(f"Processed Document {idx}")
        print(f"Source      : {item.get('source')}")
        print(f"Saved JSON  : {item.get('output_path')}")
        print("-" * 80)

        # Pretty print NLP result
        print(json.dumps(item.get("result", {}), indent=2))
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

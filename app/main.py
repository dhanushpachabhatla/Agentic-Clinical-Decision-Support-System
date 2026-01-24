# app/main.py

from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator
from pathlib import Path
import json


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    file_paths = [
        PROJECT_ROOT / "samples" / "report_1_medic.png",
        # PROJECT_ROOT / "samples" / "report_2_medic.png",
        # PROJECT_ROOT / "samples" / "Chest_Pain.pdf",
    ]

    file_paths = [str(p) for p in file_paths if p.exists()]

    if not file_paths:
        print("[ERROR] No valid input files found.")
        return

    # Initialize state
    state = ClinicalState(file_paths=file_paths)

    orchestrator = ClinicalOrchestrator()

    # Step 1: Ingestion
    state = orchestrator.run_ingestion(state)

    print(f"\n[INFO] Ingested {len(state.raw_documents)} document(s)\n")

    # Step 2: Clinical NLP
    state = orchestrator.run_clinical_nlp(state)

    # ---- OUTPUT ----

    for idx, result in enumerate(state.nlp_results, start=1):
        print("=" * 80)
        print(f"Processed Document {idx}")
        print("-" * 80)
        print(json.dumps(result, indent=2))
        print()

    if state.errors:
        print("\n[WARNINGS]")
        for e in state.errors:
            print("-", e)


if __name__ == "__main__":
    main()

# app/orchestrator.py

from app.state import ClinicalState
from services.ingestion import ingest_files
from services.clinical_nlp import extract_and_process


class ClinicalOrchestrator:
    """
    Coordinates pipeline execution.
    No medical reasoning here.
    """

    def run_ingestion(self, state: ClinicalState) -> ClinicalState:
        """
        Step 1: Ingestion
        """
        state.current_step = "ingestion"

        try:
            documents = ingest_files(state.file_paths)
            state.raw_documents = documents
        except Exception as e:
            state.add_error(f"Ingestion failed: {str(e)}")

        return state

    def run_clinical_nlp(self, state: ClinicalState) -> ClinicalState:
        """
        Step 2: Clinical NLP (extraction → reasoning → normalization)
        """
        state.current_step = "clinical_nlp"

        if not state.raw_documents:
            state.add_error("No documents available for NLP processing")
            return state

        try:
            for doc in state.raw_documents:
                result = extract_and_process([doc])
                state.nlp_results.append(result)

        except Exception as e:
            state.add_error(f"Clinical NLP failed: {str(e)}")

        return state

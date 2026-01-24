# app/orchestrator.py

from app.state import ClinicalState
from services.ingestion import ingest_files
from services.clinical_nlp import extract_and_process, save_result_json
from services.embedding import embed_clinical_json


class ClinicalOrchestrator:
    """
    Coordinates pipeline execution.

    Responsibilities:
    - Call agents in correct order
    - Pass state between stages
    - Handle failures gracefully
    - Persist outputs (but not interpret them)

    STRICT:
    - No medical reasoning
    - No data mutation
    """

    # -------------------------------------------------
    # STEP 1: INGESTION
    # -------------------------------------------------
    def run_ingestion(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "ingestion"

        try:
            documents = ingest_files(state.file_paths)

            if not documents:
                state.add_error("Ingestion produced no documents")
            else:
                state.raw_documents = documents

        except Exception as e:
            state.add_error(f"Ingestion failed: {str(e)}")

        return state

    # -------------------------------------------------
    # STEP 2: CLINICAL NLP
    # -------------------------------------------------
    def run_clinical_nlp(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "clinical_nlp"

        if not state.raw_documents:
            state.add_error("No documents available for NLP processing")
            return state

        for idx, doc in enumerate(state.raw_documents):
            try:
                # Run NLP on ONE document at a time
                result = extract_and_process([doc])

                # Persist NLP output
                output_path = save_result_json(result)

                state.nlp_results.append({
                    "source": doc.get("source"),
                    "output_path": output_path,
                    "result": result
                })

            except Exception as e:
                state.add_error(
                    f"NLP failed for document {doc.get('source', idx)}: {str(e)}"
                )

        return state

    # -------------------------------------------------
    # STEP 3: EMBEDDING
    # -------------------------------------------------
    def run_embedding(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "embedding"

        if not state.nlp_results:
            state.add_error("No NLP results available for embedding")
            return state

        for idx, item in enumerate(state.nlp_results):
            try:
                # Use NLP JSON result directly
                embedding_output = embed_clinical_json(item["result"])

                state.embedding_results.append({
                    "source": item.get("source"),
                    "embedding_model": embedding_output.get("embedding_model"),
                    "num_embeddings": len(embedding_output.get("embeddings", [])),
                    "embeddings": embedding_output.get("embeddings")
                })

            except Exception as e:
                state.add_error(
                    f"Embedding failed for document {item.get('source', idx)}: {str(e)}"
                )

        return state

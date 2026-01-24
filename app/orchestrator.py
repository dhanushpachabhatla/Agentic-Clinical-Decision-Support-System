# app/orchestrator.py

from app.state import ClinicalState
from services.ingestion import ingest_files
from services.clinical_nlp import extract_and_process, save_result_json
from services.embedding import embed_clinical_json
from services.upsert import upsert_embeddings

# ✅ NEW: date updater (runs between NLP and embedding)
from services.date_normalizer import update_dates_consistently


class ClinicalOrchestrator:
    """
    Coordinates pipeline execution.
    """

    # -----------------------------
    # STEP 1: INGESTION
    # -----------------------------
    def run_ingestion(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "ingestion"

        try:
            docs = ingest_files(state.file_paths)
            if not docs:
                state.add_error("Ingestion produced no documents")
            else:
                state.raw_documents = docs
        except Exception as e:
            state.add_error(f"Ingestion failed: {str(e)}")

        return state

    # -----------------------------
    # STEP 2: CLINICAL NLP
    # -----------------------------
    def run_clinical_nlp(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "clinical_nlp"

        if not state.raw_documents:
            state.add_error("No documents available for NLP")
            return state

        for doc in state.raw_documents:
            try:
                # ---- Raw NLP extraction
                raw_result = extract_and_process([doc])

                state.nlp_results.append({
                    "source": doc.get("source"),
                    "result": raw_result
                })

                # ---- Date normalization (AFTER NLP)
                normalized_result = update_dates_consistently(raw_result)

                # Persist normalized JSON (not raw)
                path = save_result_json(normalized_result)

                state.normalized_nlp_results.append({
                    "source": doc.get("source"),
                    "output_path": path,
                    "result": normalized_result
                })

                # Store canonical date once (for observability)
                if state.normalized_date is None:
                    state.normalized_date = normalized_result.get(
                        "doc_metadata", {}
                    ).get("date")

            except Exception as e:
                state.add_error(
                    f"NLP failed for {doc.get('source')}: {str(e)}"
                )

        return state

    # -----------------------------
    # STEP 3: EMBEDDING
    # -----------------------------
    def run_embedding(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "embedding"

        if not state.normalized_nlp_results:
            state.add_error("No normalized NLP results for embedding")
            return state

        for item in state.normalized_nlp_results:
            try:
                emb = embed_clinical_json(item["result"])

                state.embedding_results.append({
                    "source": item.get("source"),
                    "embedding_model": emb.get("embedding_model"),
                    "embeddings": emb.get("embeddings"),
                    "num_embeddings": len(emb.get("embeddings", []))
                })

            except Exception as e:
                state.add_error(
                    f"Embedding failed for {item.get('source')}: {str(e)}"
                )

        return state

    # -----------------------------
    # STEP 4: VECTOR UPSERT
    # -----------------------------
    def run_vector_upsert(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "vector_upsert"

        if not state.embedding_results:
            state.add_error("No embeddings available for upsert")
            return state

        for item in state.embedding_results:
            try:
                result = upsert_embeddings(item)

                state.vector_store_results.append({
                    "source": item.get("source"),
                    "upserted_vectors": result.get("upserted_vectors"),
                    "index": result.get("index")
                })

            except Exception as e:
                state.add_error(
                    f"Vector upsert failed for document {item.get('source')}: {str(e)}"
                )

        return state

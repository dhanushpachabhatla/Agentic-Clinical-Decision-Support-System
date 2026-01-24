# app/state.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ClinicalState:
    """
    Shared mutable state for a single pipeline execution.

    This object is passed across orchestrator steps.
    Each stage appends outputs but NEVER mutates previous results.
    """

    # -------------------------------------------------
    # Input
    # -------------------------------------------------
    file_paths: List[str]

    # -------------------------------------------------
    # Ingestion output (deterministic, no LLM)
    # -------------------------------------------------
    raw_documents: List[Dict] = field(default_factory=list)

    # -------------------------------------------------
    # NLP output (clinical_nlp.py)
    # One entry per document
    # -------------------------------------------------
    nlp_results: List[Dict] = field(default_factory=list)

    # -------------------------------------------------
    # Embedding output (embedding.py)
    # One entry per NLP result
    # -------------------------------------------------
    embedding_results: List[Dict] = field(default_factory=list)

    # -------------------------------------------------
    # Pipeline bookkeeping
    # -------------------------------------------------
    errors: List[str] = field(default_factory=list)
    current_step: Optional[str] = None

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def add_error(self, msg: str) -> None:
        """
        Record non-fatal pipeline errors.
        Pipeline should continue whenever possible.
        """
        self.errors.append(msg)

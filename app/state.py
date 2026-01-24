# app/state.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ClinicalState:
    """
    Shared mutable state for a single pipeline execution.
    """

    # Input
    file_paths: List[str]

    # Ingestion output
    raw_documents: List[Dict] = field(default_factory=list)

    # NLP output
    nlp_results: List[Dict] = field(default_factory=list)

    # Pipeline bookkeeping
    errors: List[str] = field(default_factory=list)
    current_step: Optional[str] = None

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

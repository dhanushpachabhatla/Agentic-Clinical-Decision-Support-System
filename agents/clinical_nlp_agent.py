"""
Clinical NLP Agent (Layered)
----------------------------
Layer 1: Deterministic heuristic extraction
Layer 2: Biomedical NER (scispaCy)
NO diagnosis
NO reasoning
NO hallucination
"""

from typing import List, Dict
import re

# -----------------------------
# OPTIONAL: Model-based NER
# -----------------------------

try:
    import spacy
    _NLP = spacy.load("en_ner_bc5cdr_md")
    MODEL_NER_AVAILABLE = True
except Exception:
    _NLP = None
    MODEL_NER_AVAILABLE = False


# -----------------------------
# Public API
# -----------------------------

def extract_entities(documents: List[Dict]) -> List[Dict]:
    extracted: List[Dict] = []

    for doc in documents:
        text = doc["text"]

        # ---- Layer 1: Heuristics ----
        heuristic_entities = []
        for extractor in [
            _extract_symptoms,
            _extract_conditions,
            _extract_labs,
            _extract_medications,
            _extract_procedures,
        ]:
            heuristic_entities.extend(extractor(text))

        # Normalize heuristic entities
        seen = set()
        for ent in heuristic_entities:
            key = (ent["entity"].lower(), ent["type"])
            seen.add(key)

            extracted.append(_build_entity(
                ent=ent,
                doc=doc,
                source="heuristic"
            ))

        # ---- Layer 2: Model-based NER ----
        if MODEL_NER_AVAILABLE:
            model_entities = _extract_model_entities(text)

            for ent in model_entities:
                key = (ent["entity"].lower(), ent["type"])
                if key in seen:
                    continue  # already captured by heuristics

                # Do not let model override known symptom/procedure entities
                if key[0] in [e["entity"].lower() for e in heuristic_entities]:
                    continue

                extracted.append(_build_entity(
                    ent=ent,
                    doc=doc,
                    source="model"
                ))

    return extracted


# -----------------------------
# Entity builders
# -----------------------------

def _build_entity(ent: Dict, doc: Dict, source: str) -> Dict:
    return {
        "entity": ent["entity"],
        "type": ent["type"],
        "normalized": _normalize(ent["entity"]),
        "negated": ent.get("negated", False),
        "value": ent.get("value"),
        "unit": ent.get("unit"),
        "context": ent["context"],
        "section": _infer_section(ent["context"]),
        "date": doc.get("date"),
        "source": doc.get("source"),
        "extraction_source": source,  # heuristic | model
    }

def _extract_labs(text: str) -> List[Dict]:
    """
    Extracts lab name, numeric value, and unit.
    Example:
        Hemoglobin: 10.2 g/dL
        WBC = 12000 /mm3
    """

    lab_patterns = {
        "hemoglobin": r"(hemoglobin|hb)\s*[:=]?\s*(\d+\.?\d*)\s*(g/dl|gm/dl)?",
        "wbc": r"(wbc|white blood cell)\s*[:=]?\s*(\d+)\s*(/mm3|x10\^3)?",
        "esr": r"(esr)\s*[:=]?\s*(\d+)\s*(mm/hr)?",
        "platelets": r"(platelet[s]?)\s*[:=]?\s*(\d+)\s*(/mm3|x10\^3)?",
    }

    results = []

    for lab, pattern in lab_patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = match.start()
            end = match.end()

            results.append({
                "entity": lab,
                "type": "lab",
                "value": match.group(2),
                "unit": match.group(3),
                "negated": _is_negated(text, start),
                "context": _extract_sentence_context(text, start),
            })

    return results

# -----------------------------
# Model-based NER extractor
# -----------------------------

def _extract_model_entities(text: str) -> List[Dict]:
    results = []
    doc = _NLP(text)

    for ent in doc.ents:
        raw = ent.text.strip()

        # Skip negation-like phrases
        if raw.lower().startswith(("no ", "denies", "without")):
            continue

        # Only allow true disease / drug candidates
        if ent.label_ == "DISEASE":
            etype = "condition"
        elif ent.label_ == "CHEMICAL":
            etype = "medication"
        else:
            continue

        # Skip very short / generic phrases
        if len(raw.split()) < 2:
            continue

        start = ent.start_char
        results.append({
            "entity": raw,
            "type": etype,
            "negated": _is_negated(text, start),
            "context": _extract_sentence_context(text, start),
        })

    return results



# -----------------------------
# Negation detection (sentence-bounded)
# -----------------------------

NEGATION_CUES = [
    "no", "not", "denies", "denied", "without",
    "absence of", "negative for", "free of"
]

def _is_negated(text: str, start: int) -> bool:
    sentence_start = max(
        text.rfind(".", 0, start),
        text.rfind("\n", 0, start)
    )
    if sentence_start == -1:
        sentence_start = 0

    window = text[sentence_start:start].lower()
    return any(re.search(rf"\b{cue}\b", window) for cue in NEGATION_CUES)


# -----------------------------
# Sentence-bounded context
# -----------------------------

def _extract_sentence_context(text: str, start: int) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    char_count = 0

    for i, sent in enumerate(sentences):
        sent_len = len(sent)
        if char_count <= start <= char_count + sent_len:
            prev_sent = sentences[i - 1] if i > 0 else ""
            next_sent = sentences[i + 1] if i < len(sentences) - 1 else ""
            return " ".join(s for s in [prev_sent, sent, next_sent] if s).strip()
        char_count += sent_len + 1

    return text[max(0, start - 80):start + 80]


# -----------------------------
# Section inference
# -----------------------------

def _infer_section(context: str) -> str:
    ctx = context.lower()

    if "denies" in ctx or "no " in ctx:
        return "review_of_systems"
    if "history of" in ctx:
        return "past_medical_history"
    if "treated with" in ctx or "started on" in ctx:
        return "treatment"
    if "scan" in ctx or "performed" in ctx:
        return "investigation"

    return "unspecified"


# -----------------------------
# Heuristic extractors (unchanged)
# -----------------------------

def _keyword_entity_extractor(text: str, keywords: List[str], entity_type: str) -> List[Dict]:
    lowered = text.lower()
    results = []

    for kw in keywords:
        for match in re.finditer(rf"\b{re.escape(kw)}\b", lowered):
            start, end = match.start(), match.end()
            results.append({
                "entity": kw,
                "type": entity_type,
                "negated": _is_negated(text, start),
                "context": _extract_sentence_context(text, start),
            })

    return results


def _extract_symptoms(text: str) -> List[Dict]:
    symptoms = [
        "chest pain", "shortness of breath", "fever", "fatigue",
        "weight loss", "cough", "headache", "palpitations",
        "nausea", "vomiting", "dizziness"
    ]
    return _keyword_entity_extractor(text, symptoms, "symptom")


def _extract_conditions(text: str) -> List[Dict]:
    conditions = [
        "diabetes", "hypertension", "tuberculosis", "cancer",
        "pneumonia", "asthma", "anemia", "myocardial infarction"
    ]
    return _keyword_entity_extractor(text, conditions, "condition")


def _extract_medications(text: str) -> List[Dict]:
    medications = [
        "paracetamol", "aspirin", "metformin",
        "insulin", "amoxicillin", "atorvastatin"
    ]
    return _keyword_entity_extractor(text, medications, "medication")


def _extract_procedures(text: str) -> List[Dict]:
    procedures = [
        "ct scan", "x-ray", "ecg", "echocardiogram",
        "angiography", "biopsy"
    ]
    return _keyword_entity_extractor(text, procedures, "procedure")


# -----------------------------
# Normalization
# -----------------------------

def _normalize(entity: str) -> str:
    return entity.upper().replace(" ", "_")



# -----------------------------
# MAIN (Manual Test)
# -----------------------------

def main():
    sample_docs = [
        {
            "text": """
            Patient presents with chest pain and shortness of breath.
            Denies fever and no cough.
            History of diabetes and hypertension.
            Hemoglobin: 10.2 g/dL
            WBC = 12000 /mm3
            ESR: 45 mm/hr
            CT scan of chest performed.
            Treated with aspirin and metformin.
            """,
            "doc_type": "clinical_note",
            "date": "2024-08-14",
            "source": "sample_note.txt",
        }
    ]

    entities = extract_entities(sample_docs)

    print("\nExtracted Entities:\n")
    for e in entities:
        print(e)


if __name__ == "__main__":
    main()

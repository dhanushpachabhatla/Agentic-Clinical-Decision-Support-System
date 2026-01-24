"""
Clinical + Lab NLP Pipeline with LLM Reasoning, Normalization
AND Residual Text Conclusion (STRICT & FAIL-CLOSED)
-----------------------------------------------------------
Stage 0: Document type detection
Stage 1: Deterministic extraction (labs / clinical)
Stage 2: LLM reasoning → KEEP / DROP entities
Stage 2.5: Deterministic residual text extraction
Stage 3: LLM normalization
Stage 4: LLM residual reasoning → conclusion_text

STRICT:
- NO diagnosis
- NO interpretation
- NO hallucination
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv()  # Load .env for API keys, etc.

# =====================================================
# OPTIONAL: scispaCy (ONLY for clinical notes)
# =====================================================

try:
    import spacy
    _NLP = spacy.load("en_ner_bc5cdr_md")
    MODEL_NER_AVAILABLE = True
except Exception:
    _NLP = None
    MODEL_NER_AVAILABLE = False


# =====================================================
# GROQ CLIENT
# =====================================================

from groq import Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))



# =====================================================
# PUBLIC PIPELINE
# =====================================================

def extract_and_process(documents: List[Dict]) -> Dict:
    extracted = extract_entities(documents)

    # LLM KEEP / DROP
    filtered = llm_reason_and_filter_entities(extracted)

    # Deterministic residual text
    residual_text = build_residual_text(
        documents[0]["text"],
        filtered
    )

    payload = {
        "doc_metadata": {
            "source": documents[0].get("source"),
            "date": documents[0].get("date"),
            "doc_type": documents[0].get("doc_type")
        },
        "entities": filtered,
        "residual_text": residual_text
    }

    normalized = llm_normalize_entities(payload)

    # Residual reasoning (safe)
    normalized["conclusion_text"] = llm_reason_over_residual(residual_text)

    return normalized


# =====================================================
# DOCUMENT TYPE DETECTION
# =====================================================

def detect_doc_type(text: str) -> str:
    lab_markers = [
        "bilirubin", "sgot", "sgpt", "alkaline",
        "albumin", "globulin", "serum", "lft"
    ]
    score = sum(1 for k in lab_markers if k in text.lower())
    return "lab_report" if score >= 2 else "clinical_note"


# =====================================================
# ENTITY EXTRACTION
# =====================================================

def extract_entities(documents: List[Dict]) -> List[Dict]:
    entities = []

    for doc in documents:
        text = doc["text"]
        doc_type = detect_doc_type(text)
        doc["doc_type"] = doc_type

        if doc_type == "lab_report":
            raw = extract_labs_from_rows(text)
            source = "lab_parser"
        else:
            raw = extract_clinical_entities(text)
            source = "clinical_nlp"

        for ent in raw:
            entities.append(build_entity(ent, doc, source))

    return entities


# =====================================================
# LAB PARSER (OCR-TOLERANT)
# =====================================================

JUNK_PREFIXES = [
    "iso", "i so", "regn", "mci", "hospital",
    "specimen", "facility", "note", "end of report"
]

KNOWN_LABS = [
    "bilirubin", "sgot", "sgpt", "alkaline",
    "albumin", "globulin", "protein", "ratio", "gt"
]

UNIT_MAP = {
    "mgdl": "mg/dL",
    "mg/dl": "mg/dL",
    "u/l": "U/L",
    "iul": "IU/L",
    "gnvdl": "g/dL",
    "giivdl": "g/dL",
    "omdt": "g/dL"
}


def extract_labs_from_rows(text: str) -> List[Dict]:
    results = []

    for line in text.splitlines():
        clean = re.sub(r"[^\w\s./%-]", " ", line)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 6:
            continue

        lower = clean.lower()
        if any(lower.startswith(j) for j in JUNK_PREFIXES):
            continue
        if not any(k in lower for k in KNOWN_LABS):
            continue

        match = re.match(
            r"([A-Z][A-Z ()\-\.]+?)\s+([\d]+[.,]?[\d]*)\s*([HL]?)\s*([a-zA-Z/%]+)?",
            clean,
            flags=re.IGNORECASE
        )

        if not match:
            continue

        name, value, _, unit = match.groups()
        name = re.sub(r"\(.*?\)", "", name).strip()

        if unit:
            u = unit.lower().replace(".", "")
            unit = UNIT_MAP.get(u, unit)

        start = text.find(line)
        end = start + len(line)

        results.append({
            "entity": name,
            "type": "lab",
            "value": value.replace(",", "."),
            "unit": unit,
            "negated": False,
            "context": line.strip(),
            "span": (start, end)
        })

    return results


# =====================================================
# CLINICAL NLP
# =====================================================

def extract_clinical_entities(text: str) -> List[Dict]:
    entities = []

    for extractor in [
        extract_symptoms,
        extract_conditions,
        extract_medications,
        extract_procedures
    ]:
        entities.extend(extractor(text))

    if MODEL_NER_AVAILABLE:
        entities.extend(extract_model_entities(text))

    return entities


def extract_model_entities(text: str) -> List[Dict]:
    results = []
    doc = _NLP(text)

    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            etype = "condition"
        elif ent.label_ == "CHEMICAL":
            etype = "medication"
        else:
            continue

        if len(ent.text.split()) < 2:
            continue

        results.append({
            "entity": ent.text,
            "type": etype,
            "negated": False,
            "context": extract_sentence(text, ent.start_char),
            "span": (ent.start_char, ent.end_char)
        })

    return results


# =====================================================
# HEURISTIC EXTRACTORS
# =====================================================

def keyword_extractor(text, keywords, etype):
    out = []
    for kw in keywords:
        for m in re.finditer(rf"\b{re.escape(kw)}\b", text.lower()):
            out.append({
                "entity": kw,
                "type": etype,
                "negated": is_negated(text, m.start()),
                "context": extract_sentence(text, m.start()),
                "span": (m.start(), m.start() + len(kw))
            })
    return out


def extract_symptoms(text):
    return keyword_extractor(text, ["fever", "cough", "fatigue"], "symptom")


def extract_conditions(text):
    return keyword_extractor(text, ["diabetes", "hypertension"], "condition")


def extract_medications(text):
    return keyword_extractor(text, ["paracetamol", "metformin"], "medication")


def extract_procedures(text):
    return keyword_extractor(text, ["ct scan", "x-ray"], "procedure")


# =====================================================
# ENTITY BUILDER
# =====================================================

def build_entity(ent: Dict, doc: Dict, source: str) -> Dict:
    return {
        "entity": ent["entity"],
        "type": ent["type"],
        "normalized": normalize(ent["entity"]),
        "value": ent.get("value"),
        "unit": ent.get("unit"),
        "negated": ent.get("negated", False),
        "context": ent["context"],
        "section": infer_section(ent["context"]),
        "date": doc.get("date"),
        "source": doc.get("source"),
        "extraction_source": source,
        "span": ent.get("span")
    }


# =====================================================
# UTILITIES
# =====================================================

NEGATION = ["no", "denies", "without", "negative"]

def is_negated(text, start):
    window = text[max(0, start - 80):start].lower()
    return any(n in window for n in NEGATION)


def extract_sentence(text, start):
    sents = re.split(r'(?<=[.!?])\s+', text)
    count = 0
    for s in sents:
        if count <= start <= count + len(s):
            return s.strip()
        count += len(s) + 1
    return text[start:start + 80]


def infer_section(ctx):
    c = ctx.lower()
    if any(k in c for k in ["bilirubin", "sgot", "sgpt", "alkaline"]):
        return "laboratory_results"
    return "unspecified"


def normalize(e):
    return re.sub(r"\s+", "_", e.strip().upper())


# =====================================================
# RESIDUAL TEXT BUILDER (DETERMINISTIC)
# =====================================================

def build_residual_text(original_text: str, kept_entities: List[Dict]) -> str:
    spans = [e["span"] for e in kept_entities if e.get("span")]
    spans = sorted(spans, key=lambda x: x[0])

    cursor = 0
    chunks = []

    for start, end in spans:
        if cursor < start:
            chunks.append(original_text[cursor:start])
        cursor = max(cursor, end)

    if cursor < len(original_text):
        chunks.append(original_text[cursor:])

    return "\n".join(c.strip() for c in chunks if c.strip())


# =====================================================
# LLM: ENTITY VALIDATION (FAIL-CLOSED)
# =====================================================

def llm_reason_and_filter_entities(entities: List[Dict]) -> List[Dict]:
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical data validator.\n"
                    "Decide whether each extracted entity is VALID or INVALID.\n\n"
                    "Rules:\n"
                    "- Do NOT add entities\n"
                    "- Do NOT infer diagnosis\n"
                    "- DROP headers, IDs, OCR noise\n"
                    "- Output JSON only\n\n"
                    '{ "validated_entities": [ { "entity": "...", "decision": "KEEP" | "DROP" } ] }'
                )
            },
            {
                "role": "user",
                "content": json.dumps(entities)
            }
        ],
        temperature=0,
        max_completion_tokens=2048
    )

    try:
        verdict = json.loads(completion.choices[0].message.content)
    except Exception:
        return []

    decisions = {
        v["entity"]: v["decision"]
        for v in verdict.get("validated_entities", [])
        if v.get("decision") in {"KEEP", "DROP"}
    }

    return [e for e in entities if decisions.get(e["entity"]) == "KEEP"]


# =====================================================
# LLM: NORMALIZATION
# =====================================================

def llm_normalize_entities(payload: Dict) -> Dict:
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical data normalizer.\n"
                    "- Do NOT add entities\n"
                    "- Do NOT infer diagnosis\n"
                    "- Normalize names, units, sections only\n"
                    "- Output VALID JSON"
                )
            },
            {
                "role": "user",
                "content": json.dumps(payload)
            }
        ],
        temperature=0,
        max_completion_tokens=2048
    )

    return json.loads(completion.choices[0].message.content)


# =====================================================
# LLM: RESIDUAL REASONING (SAFE CONCLUSION)
# =====================================================

def llm_reason_over_residual(residual_text: str) -> str:
    if not residual_text.strip():
        return ""

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical document sanitizer.\n"
                    "Rewrite residual text into a neutral conclusion.\n\n"
                    "STRICT:\n"
                    "- NO diagnosis\n"
                    "- NO interpretation\n"
                    "- NO medical advice\n"
                    "- NO new facts\n"
                    "- Preserve factual statements only\n\n"
                    "Output plain text only."
                )
            },
            {
                "role": "user",
                "content": residual_text
            }
        ],
        temperature=0,
        max_completion_tokens=512
    )

    return completion.choices[0].message.content.strip()


# =====================================================
# SAVE OUTPUT
# =====================================================

def save_result_json(output: Dict, base_dir: str = "results") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, base_dir)
    os.makedirs(results_dir, exist_ok=True)

    source = output["doc_metadata"].get("source", "document")
    source = os.path.splitext(os.path.basename(source))[0]

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{source}_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return path


# =====================================================
# MANUAL TEST
# =====================================================

if __name__ == "__main__":
    docs = [{
        "text": """
i SO : 9001-2008

SARVODAYA HOSPITAL Dr. (Capt) Atul per (Rott) MB Path
KJ-7, Kavi Nagar, Ghaziabad (U-P.) Regn No. MCI 3426

mi

[10 Investigation Observed Value Unit Biological Ref interval
IOCHEMISTRY

- VER FUNCTION TEST (LFT)

BILIRUBIN TOTAL es mg/dl 0,30- 1,20

CONJUGATED (D. BILIRUBIN) 7.79 H mg/dl 0.00 - 0.30
UNCONJUGATED (1.0.BILIRUBIN) 1.63 H mg/dl 0.00 - 0.70
SGOT 162 H WAL 0.00 - 46.00
SGPT 86 H WWAL 0.00 - 49.00
ALKALINE PHOSPHATASE 396 H U/L 42.00 - 128.00
TOTAL PROTEIN 6.2 gnvdl 6.20 - 3.00
ALBUMIN 3.7 L omdt 3.80 - 5.40
GLOBULIN 2.5 giivdl 1,50 - 3.60
AWG RATIO — 1.48 1.0-2.0
GAMMAT-GT 263 H IU/L 11,00 - 50.00
wo,

Specimen : SERUM
™ END OF REPORT **

FACILITIES : FOR HORMONES ASSAYS, FNAC, HISTOPATHOLOGY, BONE MARROW ASPIRATION & BIOPSY WITH MICRO PHOTOGRAPHS.
NOTE : ABOVE MENTIONED FINDINGS ARE A PROFESSIONAL J AD NOT A FINAL DIAGNGSIS, ALL LABORATORY TESTS & OTHER
INVESTIGATION RESULTS ARE TO BE CORELATED CLINIC-PATHOLOGICALLY, DISCREPANCIES, IF ANY, NECESSITATE REVIEW/REPEAT OF THE TESTS.

CLINICAL CORELATION IS MANDATORY
""",
        "date": "2024-08-14",
        "source": "sample_note.txt"
    }]

    final_output = extract_and_process(docs)
    path = save_result_json(final_output)

    print(json.dumps(final_output, indent=2))
    print(f"\n[INFO] Saved to {path}")

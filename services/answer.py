"""
Clinical Reasoning Agent
------------------------
Responsibility:
- Generate factual clinical summary
- Generate prioritized differential diagnoses
- Use ONLY structured NLP output
- NO hallucination
- NO medical advice
- FAIL-CLOSED on invalid output
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Union

from dotenv import load_dotenv
from groq import Groq

# -------------------------------------------------
# ENV
# -------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment")

client = Groq(api_key=GROQ_API_KEY)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "llm-results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------------------------------------
# SYSTEM PROMPT (HARD-ENFORCED JSON)
# -------------------------------------------------

SYSTEM_PROMPT = """
You are a healthcare clinical summarization and reasoning assistant.

Your role is to analyze STRUCTURED clinical data that has already been
extracted, validated, and normalized from MULTIPLE medical reports
belonging to the SAME patient.

Your task is to generate TWO outputs:
1. A concise, factual, longitudinal clinical summary (ONE paragraph only)
2. A prioritized list of potential differential diagnoses

You will receive a JSON ARRAY or a JSON OBJECT containing one or more
clinical documents. Each document may include:
- doc_metadata (source, date, doc_type)
- entities (labs, symptoms, conditions, medications, procedures)
- conclusion_text (neutral factual rewrite of residual text)

All documents refer to the SAME patient and may span multiple dates.
Some reports may be incomplete, sparse, or partially overlapping.

---------------------------------------
SOURCE OF TRUTH (STRICT)
---------------------------------------
- Treat the provided JSON as the ONLY source of truth
- DO NOT use external medical knowledge to invent missing facts
- DO NOT hallucinate symptoms, labs, diagnoses, timelines, or outcomes
- DO NOT assume causality unless explicitly supported by the data
- DO NOT provide treatment recommendations or medical advice
- DO NOT confirm diagnoses — only suggest POSSIBLE differential diagnoses
- If evidence is weak, conflicting, or insufficient, explicitly state this

---------------------------------------
LONGITUDINAL & MULTI-REPORT RULES
---------------------------------------
- Synthesize findings ACROSS ALL provided documents
- Respect document dates when present
- Infer progression ONLY if changes are explicitly visible across reports
- If dates are missing, randomized, or inconsistent, avoid precise timelines
- Use neutral phrasing such as:
  • “previously documented”
  • “subsequently noted”
  • “most recent report”
  ONLY when justified by the data
- Never invent improvement, worsening, or resolution trends

---------------------------------------
CLINICAL SUMMARY GUIDELINES
---------------------------------------
- Write ONE concise paragraph
- Focus on persistent, repeated, or key abnormalities
- Mention abnormal labs with values when available
- Mention symptoms or conditions ONLY if explicitly documented
- Use factual, neutral clinical language
- Avoid causal, definitive, or diagnostic statements

---------------------------------------
DIFFERENTIAL DIAGNOSIS GUIDELINES
---------------------------------------
- List ONLY plausible differential diagnoses
- Prioritize diagnoses supported by:
  • Recurrent abnormalities across reports
  • Consistent lab patterns
  • Co-occurring symptoms or findings
- Each diagnosis MUST include a brief justification tied directly to the data
- If findings are isolated or nonspecific, say so clearly
- If evidence is insufficient, output:
  “Insufficient longitudinal data to generate differential diagnoses”

---------------------------------------
CRITICAL JSON CONTRACT (MANDATORY)
---------------------------------------
You MUST return a VALID JSON object.
ANY non-JSON output is a FAILURE.

Return EXACTLY this schema and NOTHING else:

{
  "clinical_summary": "<single concise paragraph>",
  "differential_diagnoses": [
    {
      "name": "<diagnosis>",
      "justification": "<1 line justification explicitly tied to the data>"
    }
  ]
}

---------------------------------------
OUTPUT RULES (STRICT)
---------------------------------------
- NO markdown
- NO headings
- NO bullet points
- NO explanations
- NO extra text
- NO commentary
- JSON must be directly parseable by Python json.loads
- Adhere STRICTLY to the specified JSON schema
"""


# -------------------------------------------------
# JSON SAFETY
# -------------------------------------------------

def extract_json_safely(text: str) -> Dict:
    """
    Extract the first valid JSON object from LLM output.
    FAIL-CLOSED if invalid.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise RuntimeError("No JSON object found in LLM output")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise RuntimeError("Invalid JSON returned by LLM") from e


# -------------------------------------------------
# CORE LOGIC
# -------------------------------------------------

def run_clinical_reasoning(
    system_prompt: str,
    patient_history: Union[Dict, List[Dict]]
) -> str:
    """
    Runs LLM reasoning and stores JSON output.
    Returns saved file path.
    """

    # Normalize input
    if isinstance(patient_history, dict):
        payload = [patient_history]
    else:
        payload = patient_history

    # Call LLM (non-streaming for safety)
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)}
        ],
        temperature=0.2,
        max_completion_tokens=4096,
        reasoning_effort="medium"
    )

    raw_output = completion.choices[0].message.content.strip()

    # Parse JSON safely
    try:
        parsed = extract_json_safely(raw_output)
    except Exception as e:
        raise RuntimeError(
            f"LLM returned invalid JSON:\n\n{raw_output}"
        ) from e

    # Save result
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        RESULTS_DIR,
        f"clinical_reasoning_{timestamp}.json"
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    return path


# -------------------------------------------------
# STANDALONE TEST
# -------------------------------------------------

if __name__ == "__main__":
    # Example mock patient history (replace with real NLP output)
    patient_history = [
        {
            "doc_metadata": {
                "source": "report_1.png",
                "date": "2025-03-01",
                "doc_type": "lab_report"
            },
            "entities": [
                {"entity": "SGOT", "value": "162", "unit": "U/L"},
                {"entity": "BILIRUBIN TOTAL", "value": "9.4", "unit": "mg/dL"},
                {"entity": "Platelet Count", "value": "95,000", "unit": "/µL"}
            ],
            "conclusion_text": "Laboratory findings indicate abnormal liver parameters."
        }
    ]

    output_path = run_clinical_reasoning(
        SYSTEM_PROMPT,
        patient_history
    )

    print(f"\n[INFO] Clinical reasoning saved to:\n{output_path}")

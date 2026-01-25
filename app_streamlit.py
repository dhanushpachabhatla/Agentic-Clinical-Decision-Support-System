import streamlit as st
import tempfile
import asyncio
from pathlib import Path
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# -------------------------------------------------
# ENV INITIALIZATION
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "reasoning_result" not in st.session_state:
    st.session_state.reasoning_result = None

# -------------------------------------------------
# PIPELINE IMPORTS
# -------------------------------------------------
from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator

# -------------------------------------------------
# ASYNC HELPER
# -------------------------------------------------
def run_async(coro):
    return asyncio.run(coro)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Med-Insight-AI | Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Med-Insight-AI")
    st.caption("Clinical Decision Support System")
    st.divider()

    view_mode = st.radio(
        "Select View",
        ["Analysis Report", "Interactive Chat"]
    )

    st.divider()
    st.markdown("### Capabilities")
    st.markdown(
        """
        - Multi-document ingestion  
        - Deterministic clinical NLP  
        - Evidence-aware LLM reasoning  
        """
    )

    st.divider()
    st.caption("Version 2.0.0")

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.title("Med-Insight-AI")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload clinical reports (PDF or images)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload one or more clinical documents to begin.")
    st.stop()

# -------------------------------------------------
# RUN ANALYSIS
# -------------------------------------------------
if not st.session_state.analysis_complete:
    if st.button("Run Clinical Analysis", use_container_width=True):
        with st.spinner("Running clinical pipeline..."):
            temp_dir = tempfile.TemporaryDirectory()
            file_paths = []

            for file in uploaded_files:
                path = Path(temp_dir.name) / file.name
                with open(path, "wb") as f:
                    f.write(file.read())
                file_paths.append(str(path))

            orchestrator = ClinicalOrchestrator()
            state = ClinicalState(file_paths=file_paths)

            state = orchestrator.run_ingestion(state)
            state = orchestrator.run_clinical_nlp(state)
            state = orchestrator.run_embedding(state)
            state = orchestrator.run_vector_upsert(state)
            state = orchestrator.run_reasoning(state)

            st.session_state.orchestrator = orchestrator
            st.session_state.reasoning_result = state.reasoning_result
            st.session_state.analysis_complete = True

            st.rerun()

# -------------------------------------------------
# ANALYSIS REPORT VIEW
# -------------------------------------------------
if st.session_state.analysis_complete and view_mode == "Analysis Report":
    st.header("Clinical Analysis Report")

    raw_result = st.session_state.reasoning_result

    # ---- SAFE PARSING (NO CRASH GUARANTEE) ----
    if isinstance(raw_result, dict):
        result = raw_result
    elif isinstance(raw_result, str):
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            st.error("Reasoning output is not valid JSON.")
            st.stop()
    else:
        st.error(f"Unsupported reasoning output type: {type(raw_result)}")
        st.stop()

    # ---- CLINICAL SUMMARY ----
    st.subheader("Clinical Summary")
    with st.container(border=True):
        st.markdown(
            result.get(
                "clinical_summary",
                "No clinical summary available."
            )
        )

    # ---- DIFFERENTIAL DIAGNOSES ----
    st.subheader("Differential Diagnoses")

    differentials = result.get("differential_diagnoses", [])

    if not differentials:
        st.info("No differential diagnoses generated.")
    else:
        for dx in differentials:
            with st.container(border=True):
                st.markdown(f"**{dx.get('name', 'Unknown diagnosis')}**")
                st.markdown(
                    dx.get(
                        "justification",
                        "No justification provided."
                    )
                )

# -------------------------------------------------
# CHAT VIEW
# -------------------------------------------------
if st.session_state.analysis_complete and view_mode == "Interactive Chat":
    st.header("Clinical Assistant")
    st.caption("Ask questions about the uploaded documents or analysis.")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(
            "user" if isinstance(msg, HumanMessage) else "assistant"
        ):
            st.markdown(msg.content)

    # User input
    if prompt := st.chat_input("Enter a clinical question"):
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = run_async(
                    st.session_state.orchestrator.answer_user_query(
                        prompt,
                        st.session_state.messages[:-1]
                    )
                )

                answer = response.get("answer", "No response generated.")
                st.markdown(answer)

                if response.get("sources"):
                    with st.expander("Source Documents"):
                        for src in response["sources"]:
                            st.markdown(f"- {src.get('title', 'Unknown source')}")

        st.session_state.messages.append(AIMessage(content=answer))

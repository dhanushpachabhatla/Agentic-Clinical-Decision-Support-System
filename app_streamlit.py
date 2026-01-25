import streamlit as st
import tempfile
import asyncio
from pathlib import Path
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# -------------------------------------------------
# ENV & STATE INITIALIZATION
# -------------------------------------------------
load_dotenv()

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
    """Helper to run async functions in Streamlit's sync environment."""
    return asyncio.run(coro)

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="MediMind AI – Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 MediMind AI")
    st.caption("Agentic Clinical Decision Support")
    st.divider()
    
    # Toggle between Chat and Report
    view_mode = st.radio("Select View:", ["📄 Analysis Report", "💬 Interactive Chat"])
    
    st.divider()
    st.markdown("### 📌 Capabilities")
    st.markdown("• Multi-report ingestion\n• Deterministic clinical NLP\n• LLM-based reasoning")
    st.divider()
    st.caption("Version: v2.0.0 (Chat Enabled)")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
st.title("🧠 MediMind AI")

uploaded_files = st.file_uploader(
    "📤 Upload Clinical Reports (PDF / Images)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload reports to begin.")
    st.stop()

# Save files and init Orchestrator
if not st.session_state.analysis_complete:
    if st.button("🚀 Run Clinical Analysis", use_container_width=True):
        with st.spinner("Processing pipeline..."):
            temp_dir = tempfile.TemporaryDirectory()
            file_paths = []
            for file in uploaded_files:
                temp_path = Path(temp_dir.name) / file.name
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                file_paths.append(str(temp_path))

            # Initialize and Run Pipeline
            orch = ClinicalOrchestrator()
            state = ClinicalState(file_paths=file_paths)
            
            # Step-by-step execution mirroring your main.py
            state = orch.run_ingestion(state)
            state = orch.run_clinical_nlp(state)
            state = orch.run_embedding(state)
            state = orch.run_vector_upsert(state)
            state = orch.run_reasoning(state)

            # Store in session state
            st.session_state.orchestrator = orch
            st.session_state.reasoning_result = state.reasoning_result
            st.session_state.analysis_complete = True
            st.rerun()

# -------------------------------------------------
# MAIN INTERFACE (REPORT VS CHAT)
# -------------------------------------------------
if st.session_state.analysis_complete:
    
    if view_mode == "📄 Analysis Report":
        st.header("📄 Clinical Reasoning Results")
        with st.container(border=True):
            st.markdown(st.session_state.reasoning_result or "No reasoning produced.")

    else:
        st.header("💬 Clinical Assistant Chat")
        st.caption("Ask questions about the uploaded documents or the clinical summary.")

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
                st.markdown(message.content)

        # Chat Input
        if prompt := st.chat_input("Ask about the patient's history..."):
            # Add User Message
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Call the async orchestrator method
                    response = run_async(
                        st.session_state.orchestrator.answer_user_query(
                            prompt, 
                            st.session_state.messages[:-1] # Exclude latest prompt
                        )
                    )
                    
                    answer = response['answer']
                    st.markdown(answer)
                    
                    if response.get('sources'):
                        with st.expander("View Sources"):
                            for s in response['sources']:
                                st.write(f"- {s['title']}")

            # Update Session History
            st.session_state.messages.append(AIMessage(content=answer))